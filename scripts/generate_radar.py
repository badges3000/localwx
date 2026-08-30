#!/usr/bin/env python3
"""
DWD RADOLAN HD Vector-Style Niederschlagsradar Generator (100% DWD OpenData)
=============================================================================
Erzeugt gestochen scharfe Vektor-Stil Radar-Overlays im offiziellen
DWD WarnWetter / Kachelmann Look:

Features:
- Analytische Sub-Pixel Vektor-Isolinien (Distance-Field Rendering)
- Organisch abgerundete Kurven (Bicubic Splines) OHNE Pixel-Treppen
- Knackig scharfe Kanten (exakt 1-Pixel Antialiasing, KEIN Weichzeichner-Dunst)
- Leuchtende, gleichmäßige Vektor-Außenkonturlinien (Strokes)
- Exakte DWD DE1200 Koordinaten-Offsets (x_0 = -543.462 km, y_0 = -4808.645 km)
- DWD KONRAD3D Gewitter-Zellen Erkennung & Boden-Snapping
- Kompakte, 100% verlustfreie WebP-Kompression
"""

import os
import sys
import json
import time
import bz2
import tarfile
import io
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

# Exakte DWD RADOLAN DE1200 Bounding Box (Zielgitter 1400x1400)
RADAR_BOUNDS = [[45.68, 1.46], [55.86, 18.73]]


# ==============================================================================
# VEKTOR-FARBSTUFEN & KONTURLINIEN (DWD WARNWETTER STANDARD)
# ==============================================================================

# 6 meteorologische Vektor-Bänder: (Schwellenwert_Roh, Fill_RGBA, Stroke_RGBA, Stroke_Breite_px_bei_2400)
VECTOR_LEVELS = [
    # 1. Zarter Niesel / Feuchtesaum (val >= 2.0 -> 0.24 mm/h)
    (2.0, (22, 185, 235, 175), (56, 220, 250, 255), 3.2),
    # 2. Leichter bis mäßiger Landregen (val >= 6.0 -> 0.72 mm/h)
    (6.0, (34, 197, 94, 230), (74, 222, 128, 255), 3.2),
    # 3. Kräftiger Schauer (val >= 21.0 -> 2.52 mm/h) -> Strahlendes Goldgelb
    (21.0, (250, 204, 21, 255), (254, 240, 138, 255), 3.6),
    # 4. Starkregen (val >= 56.0 -> 6.72 mm/h) -> Kräftiges Warn-Orange
    (56.0, (249, 115, 22, 255), (253, 186, 116, 255), 3.8),
    # 5. Unwetter / Extremregen (val >= 141.0 -> 16.9 mm/h) -> Feuriges Alarmrot
    (141.0, (239, 68, 68, 255), (254, 202, 202, 255), 4.2),
    # 6. Hagelkern / Extremer Starkregen (val >= 281.0 -> > 33.7 mm/h) -> Leuchtendes Magenta/Weiß
    (281.0, (217, 70, 239, 255), (255, 255, 255, 255), 4.6)
]


# ==============================================================================
# KOORDINATENTRANSFORMATION & REPROJEKTION
# ==============================================================================

_WARP_COORDS = None

def get_reprojection_coords(target_h=2400, target_w=2400):
    global _WARP_COORDS
    if _WARP_COORDS is not None:
        return _WARP_COORDS

    lat_min, lat_max = RADAR_BOUNDS[0][0], RADAR_BOUNDS[1][0]
    lon_min, lon_max = RADAR_BOUNDS[0][1], RADAR_BOUNDS[1][1]

    R = 6370.040
    lat_ts = np.radians(60.0)
    lon_0 = np.radians(10.0)
    scale = R * (1.0 + np.sin(lat_ts))

    y_merc_max = np.log(np.tan(np.pi / 4.0 + np.radians(lat_max) / 2.0))
    y_merc_min = np.log(np.tan(np.pi / 4.0 + np.radians(lat_min) / 2.0))
    y_merc_grid = np.linspace(y_merc_max, y_merc_min, target_h)
    lats = np.degrees(2.0 * np.arctan(np.exp(y_merc_grid)) - np.pi / 2.0)

    lons = np.linspace(lon_min, lon_max, target_w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    phi = np.radians(lat_grid)
    lam = np.radians(lon_grid)

    m = scale / (1.0 + np.sin(phi))
    x_proj = m * np.cos(phi) * np.sin(lam - lon_0)
    y_proj = -m * np.cos(phi) * np.cos(lam - lon_0)

    # Exakte DWD DE1200 Gitter-Offsets:
    x_px = x_proj + 543.462
    y_px = y_proj + 4808.645

    _WARP_COORDS = (y_px, x_px)
    return _WARP_COORDS


def reproject_and_smooth_radar(grid_1200x1100):
    """
    High-DPI Bicubic-Spline Reprojektion (2400x2400) + stufenlose organische Rundung
    für absolut glatte, elegante Vektorkurven (wie in Illustrator / DWD WarnWetter).
    """
    try:
        from scipy.ndimage import map_coordinates, gaussian_filter

        y_coords, x_coords = get_reprojection_coords(2400, 2400)
        # Bicubic Spline (order=3) erzeugt stufenlose, geschwungene Kurven
        warped = map_coordinates(grid_1200x1100.astype(np.float32), [y_coords, x_coords], order=3, mode='constant', cval=0.0)
        # Organische Rundung (sigma=2.6 bei 2400x2400) für geschmeidige Vektor-Isolinien
        smoothed = gaussian_filter(warped, sigma=2.6)
        return np.maximum(0.0, smoothed)
    except ImportError:
        flipped = np.flipud(grid_1200x1100)
        img = Image.fromarray(flipped)
        return np.array(img.resize((2400, 2400), Image.BICUBIC), dtype=np.float32)


def remove_isolated_radar_clutter(val):
    if not np.any(val > 0):
        return val

    try:
        from scipy.ndimage import label, maximum as nd_max, sum as nd_sum, binary_dilation

        labeled_array, num_features = label(val > 0)
        if num_features == 0:
            return val

        indices = np.arange(1, num_features + 1)
        cluster_sizes = nd_sum(np.ones_like(val), labels=labeled_array, index=indices)
        cluster_maxs = nd_max(val, labels=labeled_array, index=indices)

        is_core_or_large = (cluster_maxs >= 4) | (cluster_sizes >= 150)
        valid_ids = indices[is_core_or_large]
        valid_mask = np.isin(labeled_array, valid_ids)

        expanded_zone = binary_dilation(valid_mask, iterations=15)
        overlap = nd_sum(expanded_zone.astype(int), labels=labeled_array, index=indices)
        is_valid_cluster = is_core_or_large | (overlap > 0)

        invalid_cluster_ids = indices[~is_valid_cluster]
        clean_val = val.copy()
        if len(invalid_cluster_ids) > 0:
            is_invalid_pixel = np.isin(labeled_array, invalid_cluster_ids)
            clean_val[is_invalid_pixel] = 0

        return clean_val
    except ImportError:
        return val


def parse_radolan_binary(data_bytes):
    etx_pos = data_bytes.find(b'\x03')
    if etx_pos == -1:
        etx_pos = data_bytes.find(b'\n\x00')
    if etx_pos == -1:
        return None, None

    header = data_bytes[:etx_pos].decode('latin1', errors='ignore')
    raw_data = data_bytes[etx_pos + 1:]

    width, height = 1100, 1200
    if '900x900' in header:
        width, height = 900, 900

    expected_16bit = width * height * 2
    if len(raw_data) >= expected_16bit:
        arr = np.frombuffer(raw_data[:expected_16bit], dtype=np.uint16).reshape((height, width))
        is_nodata = (arr & 0x6000) > 0
        val = (arr & 0x0FFF).astype(np.float32)
        val[is_nodata] = 0.0
        val[val < 2.0] = 0.0

        val = remove_isolated_radar_clutter(val)
        return header, val

    return None, None


# ==============================================================================
# ANALYTISCHE VEKTOR-STIL RENDERING ENGINE (DWD WARNWETTER LOOK)
# ==============================================================================

def render_vector_matrix_to_webp(field, output_path):
    """
    Analytic Vector-Style Rendering Engine:
    Rendert glatte, organisch abgerundete Vektor-Isolinien mit
    gestochen scharfer Sub-Pixel Kantenschärfe (kein Weichzeichner!).
    """
    h, w = field.shape
    if not np.any(field >= 1.5):
        empty_img = Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), mode='RGBA')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        empty_img.save(output_path, 'WEBP', lossless=True, quality=100, method=6)
        return

    # 1. Berechne den Sub-Pixel Gradientenbetrag |grad(field)| für analytische Distanz
    gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx * gx + gy * gy) + 1e-4

    # 2. RGBA Canvas in Float (0.0 .. 1.0)
    out_r = np.zeros((h, w), dtype=np.float32)
    out_g = np.zeros((h, w), dtype=np.float32)
    out_b = np.zeros((h, w), dtype=np.float32)
    out_a = np.zeros((h, w), dtype=np.float32)

    # 3. Rendere von niedrigster Stufe (Niesel) bis höchster Stufe (Hagel)
    for threshold, fill_rgba, stroke_rgba, stroke_w in VECTOR_LEVELS:
        if not np.any(field >= (threshold - 1.0)):
            continue

        # Vorzeichenbehaftete Subpixel-Distanz zur Isolinie (in Pixeln)
        # d > 0 = innerhalb des Polygons, d = 0 = Konturlinie, d < 0 = außerhalb
        d = (field - threshold) / grad_mag

        # A) Analytische Vektor-Flächenfüllung: Exakt 1-Pixel Sub-Pixel Schärfe an der Kante
        fill_cov = np.clip(d + 0.5, 0.0, 1.0)
        
        # B) Analytische leuchtende Vektor-Außenlinie mit definierter Breite stroke_w
        stroke_cov = np.clip(1.0 - np.abs(d) / (stroke_w / 2.0), 0.0, 1.0)

        # Normalisierte Farben
        f_r, f_g, f_b, f_a = [c / 255.0 for c in fill_rgba]
        s_r, s_g, s_b, s_a = [c / 255.0 for c in stroke_rgba]

        # Kombiniere Stroke und Fill
        s_weight = stroke_cov * s_a
        f_weight = np.maximum(0.0, fill_cov * f_a - s_weight * 0.7)
        total_w = s_weight + f_weight + 1e-6

        curr_r = (s_weight * s_r + f_weight * f_r) / total_w
        curr_g = (s_weight * s_g + f_weight * f_g) / total_w
        curr_b = (s_weight * s_b + f_weight * f_b) / total_w
        curr_a = np.maximum(fill_cov * f_a, stroke_cov * s_a)

        # Standard Alpha-Blend über den bestehenden Canvas
        inv_a = 1.0 - curr_a
        out_r = curr_r * curr_a + out_r * inv_a
        out_g = curr_g * curr_a + out_g * inv_a
        out_b = curr_b * curr_a + out_b * inv_a
        out_a = curr_a + out_a * inv_a

    # 4. In 8-Bit RGBA Bild konvertieren und verlustfrei speichern
    img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    img_rgba[..., 0] = np.clip(out_r * 255.0, 0, 255).astype(np.uint8)
    img_rgba[..., 1] = np.clip(out_g * 255.0, 0, 255).astype(np.uint8)
    img_rgba[..., 2] = np.clip(out_b * 255.0, 0, 255).astype(np.uint8)
    img_rgba[..., 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_rgba, mode='RGBA')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'WEBP', lossless=True, quality=100, method=6)


# ==============================================================================
# DWD RADOLAN PIPELINE & DOWNLOADS
# ==============================================================================

def get_available_dwd_rv_files():
    url = "https://opendata.dwd.de/weather/radar/composite/rv/"
    req = urllib.request.Request(url, headers={'User-Agent': 'localwx-Vector-Radar/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        
        pattern = r'href="(DE1200_RV(\d{10})\.tar\.bz2)"'
        matches = re.findall(pattern, html)
        
        files = []
        for filename, dt_str in matches:
            try:
                dt = datetime.strptime(f"20{dt_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                files.append((filename, dt))
            except Exception:
                pass
        
        files.sort(key=lambda x: x[1])
        return files
    except Exception as e:
        print(f"⚠️ Fehler beim Abrufen des DWD-Index: {e}")
        return []


def download_and_extract_tar_bz2(filename):
    url = f"https://opendata.dwd.de/weather/radar/composite/rv/{filename}"
    req = urllib.request.Request(url, headers={'User-Agent': 'localwx-Vector-Radar/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            compressed_bytes = resp.read()
        
        tar_bytes = bz2.decompress(compressed_bytes)
        tar_stream = io.BytesIO(tar_bytes)
        
        extracted_files = {}
        with tarfile.open(fileobj=tar_stream, mode="r:") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        extracted_files[member.name] = f.read()
        return extracted_files
    except Exception as e:
        print(f"⚠️ Fehler beim Download/Entpacken von {filename}: {e}")
        return None


def apply_temporal_consistency_filter(grid_list):
    n_frames = len(grid_list)
    if n_frames < 3:
        return grid_list

    stack = np.array(grid_list, dtype=np.float32)
    cleaned = np.copy(stack)

    for i in range(1, n_frames - 1):
        prev_f = stack[i - 1]
        curr_f = stack[i]
        next_f = stack[i + 1]

        single_frame_pop = (curr_f > 0) & (prev_f == 0) & (next_f == 0)
        cleaned[i][single_frame_pop] = 0

        gap_drop = (curr_f == 0) & (prev_f >= 6.0) & (next_f >= 6.0)
        cleaned[i][gap_drop] = (prev_f[gap_drop] + next_f[gap_drop]) / 2.0

    return [cleaned[i] for i in range(n_frames)]


# ==============================================================================
# DWD KONRAD3D GEWITTERZELLEN-ERKENNUNG MIT BODEN-SNAPPING
# ==============================================================================

def snap_cell_to_surface_radar(lat_aloft, lon_aloft, surface_grid, search_radius_km=35):
    if surface_grid is None:
        return lat_aloft, lon_aloft, 0.0, 0.0

    try:
        R = 6370.040
        lat_ts = np.radians(60.0)
        lon_0 = np.radians(10.0)
        scale = R * (1.0 + np.sin(lat_ts))

        phi = np.radians(lat_aloft)
        lam = np.radians(lon_aloft)
        m = scale / (1.0 + np.sin(phi))
        xp = m * np.cos(phi) * np.sin(lam - lon_0)
        yp = -m * np.cos(phi) * np.cos(lam - lon_0)

        cx = int(round(xp + 543.462))
        cy = int(round(yp + 4808.645))

        h, w = surface_grid.shape
        r_px = int(round(search_radius_km))
        xmin = max(0, cx - r_px)
        xmax = min(w - 1, cx + r_px)
        ymin = max(0, cy - r_px)
        ymax = min(h - 1, cy + r_px)

        window = surface_grid[ymin:ymax+1, xmin:xmax+1]
        max_val = np.max(window) if window.size > 0 else 0

        if max_val >= 15:
            threshold = max(15.0, 0.82 * max_val)
            mask = window >= threshold
            y_indices, x_indices = np.where(mask)
            weights = np.power(window[mask].astype(np.float64) - threshold + 1.0, 2.0)
            
            if np.sum(weights) > 0:
                x_ground = np.sum(x_indices * weights) / np.sum(weights) + xmin
                y_ground = np.sum(y_indices * weights) / np.sum(weights) + ymin

                xp_g = x_ground - 543.462
                yp_g = y_ground - 4808.645
                d = np.sqrt(xp_g * xp_g + yp_g * yp_g)
                phi_g = np.pi / 2.0 - 2.0 * np.arctan(d / scale)
                lam_g = lon_0 + np.arctan2(xp_g, -yp_g)
                
                lat_ground = np.degrees(phi_g)
                lon_ground = np.degrees(lam_g)

                d_lat = lat_ground - lat_aloft
                d_lon = lon_ground - lon_aloft
                
                dist_km = np.sqrt(((d_lat * 111.32) ** 2) + ((d_lon * 111.32 * np.cos(np.radians(lat_aloft))) ** 2))
                if dist_km <= 40.0:
                    return lat_ground, lon_ground, d_lat, d_lon
    except Exception as e:
        print(f"Hinweis beim Boden-Snapping: {e}")

    return lat_aloft, lon_aloft, 0.0, 0.0


def fetch_and_parse_konrad3d(output_dir, surface_grid=None):
    print("⚡ Rufe aktuelle DWD KONRAD3D Gewitter- & Hagelzell-Daten ab...")
    cells_data = {
        "reference_time": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cells": 0,
        "cells": []
    }
    
    try:
        url = "https://opendata.dwd.de/weather/radar/konrad3d/"
        req = urllib.request.Request(url, headers={'User-Agent': 'localwx-Radar/2.0'})
        with urllib.request.urlopen(req, timeout=12) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        
        files = re.findall(r'href="(KONRAD3D_[^"]+\.xml)"', html)
        if not files:
            with open(os.path.join(output_dir, "cells.json"), 'w', encoding='utf-8') as f:
                json.dump(cells_data, f, indent=2, ensure_ascii=False)
            return cells_data

        latest_file = sorted(files)[-1]
        file_url = f"https://opendata.dwd.de/weather/radar/konrad3d/{latest_file}"
        with urllib.request.urlopen(file_url, timeout=15) as resp:
            xml_bytes = resp.read()

        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_bytes)
        
        ref_time_elem = root.find('.//metadata/reference_time')
        ref_time = ref_time_elem.text if ref_time_elem is not None else datetime.now(timezone.utc).isoformat()
        cells_data["reference_time"] = ref_time

        parsed_cells = []
        for feat in root.findall('.//feature'):
            feat_id = feat.attrib.get('identifier', str(len(parsed_cells) + 1))
            
            centroid = feat.find('.//centroid_3d/geodetic_coordinate')
            if centroid is None:
                continue
            lat_elem = centroid.find('latitude')
            lon_elem = centroid.find('longitude')
            if lat_elem is None or lon_elem is None:
                continue
            lat = round(float(lat_elem.text), 5)
            lon = round(float(lon_elem.text), 5)
            
            height_elem = centroid.find('height_msl')
            height_m = round(float(height_elem.text), 0) if height_elem is not None else 0
            
            motion = feat.find('.//motion')
            speed = 0.0
            direction = 0.0
            if motion is not None:
                sp_elem = motion.find('.//speed')
                if sp_elem is not None and sp_elem.text:
                    try:
                        speed = round(float(sp_elem.text), 1)
                    except ValueError:
                        pass
                dir_elem = motion.find('.//direction')
                if dir_elem is not None and dir_elem.text:
                    try:
                        direction = round(float(dir_elem.text), 0)
                    except ValueError:
                        pass
            
            if speed == 0.0:
                cs_elem = feat.find('.//cell_speed')
                if cs_elem is not None and cs_elem.text:
                    try:
                        speed = round(float(cs_elem.text), 1)
                    except ValueError:
                        pass

            intensity = feat.find('.//intensity')
            max_dbz = 0.0
            severity = 0
            hail_flag = 0
            heavy_rain_flag = 0
            gust_speed = 0.0
            
            if intensity is not None:
                dbz_elem = intensity.find('max_value')
                if dbz_elem is not None and dbz_elem.text:
                    try:
                        max_dbz = round(float(dbz_elem.text), 1)
                    except ValueError:
                        pass
                
                sev_elem = intensity.find('severity')
                if sev_elem is not None and sev_elem.text:
                    try:
                        severity = int(sev_elem.text)
                    except ValueError:
                        pass
                
                hf_elem = intensity.find('hail_flag')
                if hf_elem is not None and hf_elem.text:
                    try:
                        hail_flag = int(hf_elem.text)
                    except ValueError:
                        pass
                        
                hrf_elem = intensity.find('heavy_rain_flag')
                if hrf_elem is not None and hrf_elem.text:
                    try:
                        heavy_rain_flag = int(hrf_elem.text)
                    except ValueError:
                        pass
                        
                gust_elem = intensity.find('maximum_estimated_wind_gust')
                if gust_elem is not None and gust_elem.text:
                    try:
                        gust_speed = round(float(gust_elem.text), 1)
                    except ValueError:
                        pass

            lightning_rate = 0
            lt_elem = feat.find('.//lightning/lightning_rate')
            if lt_elem is not None and lt_elem.text:
                try:
                    lightning_rate = int(round(float(lt_elem.text)))
                except ValueError:
                    pass

            is_mesocyclone = False
            meso_elem = feat.find('.//mesocyclone/mesocyclone_severity_index')
            if meso_elem is not None and meso_elem.text:
                try:
                    is_mesocyclone = int(meso_elem.text) > 0
                except ValueError:
                    pass

            polygon_coords = []
            poly = feat.find('.//polygons_projected/geodetic_coordinates/polygon')
            if poly is not None:
                lats_elem = poly.find('latitudes')
                lons_elem = poly.find('longitudes')
                if lats_elem is not None and lons_elem is not None and lats_elem.text and lons_elem.text:
                    try:
                        lats_list = [round(float(x), 5) for x in lats_elem.text.strip().split()]
                        lons_list = [round(float(x), 5) for x in lons_elem.text.strip().split()]
                        polygon_coords = [[lt, ln] for lt, ln in zip(lats_list, lons_list)]
                    except Exception:
                        pass

            forecast_track = []
            for fc_elem in feat.findall('.//forecast/centroid_forecasts/centroid_forecast'):
                fc_lat = fc_elem.find('.//geodetic_coordinate/latitude') or fc_elem.find('latitude')
                fc_lon = fc_elem.find('.//geodetic_coordinate/longitude') or fc_elem.find('longitude')
                
                fc_time_str = fc_elem.attrib.get('forecast_time', '')
                lead_time_min = 0
                if fc_time_str and ref_time:
                    try:
                        t_ref = datetime.fromisoformat(ref_time.replace('Z', '+00:00'))
                        t_fc = datetime.fromisoformat(fc_time_str.replace('Z', '+00:00'))
                        lead_time_min = int(round((t_fc - t_ref).total_seconds() / 60.0))
                    except Exception:
                        lead_time_min = (len(forecast_track) + 1) * 5
                else:
                    lead_time_min = (len(forecast_track) + 1) * 5

                if fc_lat is not None and fc_lon is not None and fc_lat.text and fc_lon.text:
                    try:
                        forecast_track.append({
                            "lead_time_min": lead_time_min,
                            "lat": round(float(fc_lat.text), 5),
                            "lon": round(float(fc_lon.text), 5)
                        })
                    except ValueError:
                        pass

            if forecast_track:
                first_pt = forecast_track[0]
                d_lat = first_pt["lat"] - lat
                d_lon = (first_pt["lon"] - lon) * np.cos(np.radians(lat))
                angle_deg = np.degrees(np.arctan2(d_lon, d_lat))
                if direction == 0:
                    direction = round((angle_deg + 360) % 360, 0)
                if speed == 0 and first_pt["lead_time_min"] > 0:
                    dist_km = np.sqrt((d_lat * 111.32)**2 + (d_lon * 111.32)**2)
                    speed = round(dist_km / (first_pt["lead_time_min"] / 60.0), 1)

            lat_ground, lon_ground, d_lat, d_lon = snap_cell_to_surface_radar(lat, lon, surface_grid)
            if abs(d_lat) > 0.0001 or abs(d_lon) > 0.0001:
                lat = round(lat_ground, 5)
                lon = round(lon_ground, 5)
                if polygon_coords:
                    polygon_coords = [[round(p[0] + d_lat, 5), round(p[1] + d_lon, 5)] for p in polygon_coords]
                if forecast_track:
                    forecast_track = [{**pt, "lat": round(pt["lat"] + d_lat, 5), "lon": round(pt["lon"] + d_lon, 5)} for pt in forecast_track]

            level = "moderate"
            level_name = "Mäßiges Gewitter"
            color = "#f59e0b"
            
            if is_mesocyclone or max_dbz >= 65:
                level = "extreme"
                level_name = "Extremes Unwetter / Superzelle"
                color = "#d946ef"
            elif hail_flag or max_dbz >= 55 or gust_speed >= 75:
                level = "severe"
                level_name = "Schweres Gewitter / Hagel"
                color = "#ef4444"
            elif max_dbz >= 45 or heavy_rain_flag:
                level = "strong"
                level_name = "Kräftiges Gewitter"
                color = "#f97316"

            parsed_cells.append({
                "id": feat_id,
                "reference_time": ref_time,
                "lat": lat,
                "lon": lon,
                "height_m": height_m,
                "level": level,
                "level_name": level_name,
                "color": color,
                "max_dbz": max_dbz,
                "severity": severity,
                "is_hail": bool(hail_flag),
                "is_heavy_rain": bool(heavy_rain_flag),
                "is_mesocyclone": is_mesocyclone,
                "max_gust_kmh": gust_speed,
                "speed_kmh": speed,
                "direction_deg": direction,
                "lightning_rate": lightning_rate,
                "polygon": polygon_coords,
                "forecast_track": forecast_track
            })

        cells_data["total_cells"] = len(parsed_cells)
        cells_data["cells"] = parsed_cells
        print(f"✅ {len(parsed_cells)} aktive Gewitterzellen erfolgreich aus KONRAD3D geparst.")

    except Exception as e:
        print(f"⚠️ Hinweis bei KONRAD3D Parsing: {e}")

    cells_path = os.path.join(output_dir, "cells.json")
    with open(cells_path, 'w', encoding='utf-8') as f:
        json.dump(cells_data, f, indent=2, ensure_ascii=False)

    return cells_data


# ==============================================================================
# HAUPT-GENERATOR & UPLOAD
# ==============================================================================

def generate_vector_radar_dataset():
    start_time = time.time()
    print("🚀 Starte DWD RADOLAN HD Vektor-Stil Radar Pipeline (Analytic Subpixel Engine)...")

    output_dir = "./dist/radar"
    os.makedirs(output_dir, exist_ok=True)

    dwd_files = get_available_dwd_rv_files()
    if not dwd_files:
        print("❌ Keine DWD RV Dateien gefunden.")
        return

    now = datetime.now(timezone.utc)
    eight_hours_ago = now - timedelta(hours=8)
    selected_history = [f for f in dwd_files if f[1] >= eight_hours_ago and f[1] <= now]
    print(f"📦 Lade {len(selected_history)} historische 5-Minuten-Messungen herunter...")

    raw_history_items = []

    def process_history_item(item_tuple, idx):
        filename, valid_dt = item_tuple
        data_dict = download_and_extract_tar_bz2(filename)
        if not data_dict:
            return None

        main_key = [k for k in data_dict.keys() if k.endswith('_000') or len(k.split('_')) == 2]
        if not main_key:
            main_key = list(data_dict.keys())
        main_key = sorted(main_key)[0]

        file_bytes = data_dict[main_key]
        header, grid = parse_radolan_binary(file_bytes)
        if grid is None:
            return None

        return {
            "valid_dt": valid_dt,
            "grid": grid,
            "is_nowcast": False
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_history_item, item, i): i for i, item in enumerate(selected_history)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                raw_history_items.append(res)

    raw_history_items.sort(key=lambda x: x['valid_dt'])
    print(f"✅ {len(raw_history_items)} historische 5-Minuten-Grids geladen.")

    raw_nowcast_items = []
    latest_file = dwd_files[-1][0]
    latest_dt = dwd_files[-1][1]
    print(f"🔮 Lade DWD 5-Minuten Nowcast (+2h) aus {latest_file}...")

    nowcast_data = download_and_extract_tar_bz2(latest_file)
    if nowcast_data:
        nowcast_keys = [k for k in sorted(nowcast_data.keys()) if not k.endswith('_000')]
        selected_nowcast_keys = []
        for k in nowcast_keys:
            m = re.search(r'_(\d{3})$', k)
            if m:
                minutes = int(m.group(1))
                selected_nowcast_keys.append((k, minutes))
        
        selected_nowcast_keys.sort(key=lambda x: x[1])

        for k, minutes in selected_nowcast_keys:
            file_bytes = nowcast_data[k]
            header, grid = parse_radolan_binary(file_bytes)
            if grid is not None:
                valid_dt = latest_dt + timedelta(minutes=minutes)
                raw_nowcast_items.append({
                    "valid_dt": valid_dt,
                    "grid": grid,
                    "is_nowcast": True
                })

    all_items = raw_history_items + raw_nowcast_items
    all_items.sort(key=lambda x: x['valid_dt'])

    raw_grids = [item['grid'] for item in all_items]
    cleaned_grids = apply_temporal_consistency_filter(raw_grids)

    print(f"🎨 Rendere {len(all_items)} Frames im analytischen Vektor-Stil (Glatte Isolinien + Strokes)...")

    frames_metadata = []

    def render_and_save(idx):
        item = all_items[idx]
        grid = cleaned_grids[idx]
        
        # Bicubic-Reprojektion + organische Isolinien-Rundung
        field = reproject_and_smooth_radar(grid)
        file_name = f"radar_{idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)
        
        # Analytisches Vektor-Rendering mit 1-Pixel Subpixel-Schärfe
        render_vector_matrix_to_webp(field, file_path)

        return {
            "step": idx,
            "valid_time": item['valid_dt'].strftime("%Y-%m-%dT%H:%M:00Z"),
            "file": file_name,
            "is_nowcast": item['is_nowcast']
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(render_and_save, i): i for i in range(len(all_items))}
        for future in as_completed(futures):
            res = future.result()
            if res:
                frames_metadata.append(res)

    frames_metadata.sort(key=lambda x: x['step'])
    print(f"✨ Gesamt-Datensatz: {len(frames_metadata)} Vektor-Frames fertig gerendert!")

    # KONRAD3D Zelltracking
    latest_surface_grid = raw_history_items[-1]['grid'] if raw_history_items else (cleaned_grids[-1] if cleaned_grids else None)
    cells_result = fetch_and_parse_konrad3d(output_dir, surface_grid=latest_surface_grid)

    metadata = {
        "model": "DWD RADOLAN HD Vector-Engine",
        "parameter": "precipitation_radar",
        "title": "DWD RADOLAN HD Vektor-Niederschlagsradar (-8h bis +2h, 5-Minuten-Takt)",
        "unit": "mm/h",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": RADAR_BOUNDS,
        "cells_file": "cells.json",
        "cells_count": cells_result.get("total_cells", 0),
        "frames": frames_metadata
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"🎉 Vektor-Radar-Generierung in {duration}s abgeschlossen!")

    upload_directory_to_ftp(output_dir, "radar")


def upload_directory_to_ftp(local_dir, remote_folder="radar"):
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password:
        print(f"ℹ️ Keine FTP-Zugangsdaten. Überspringe Upload für {remote_folder}.")
        return

    print(f"\n📡 Verbinde mit FTP-Server für 'data/{remote_folder}/'...")

    try:
        ftp = ftplib.FTP_TLS()
        ftp.connect(server, 21, timeout=30)
        ftp.login(user, password)
        ftp.prot_p()
    except Exception:
        try:
            ftp = ftplib.FTP()
            ftp.connect(server, 21, timeout=30)
            ftp.login(user, password)
        except Exception as e:
            print(f"❌ FTP-Verbindung fehlgeschlagen: {e}")
            return

    ftp.cwd('/')
    for folder in ["data", remote_folder]:
        try:
            ftp.cwd(folder)
        except ftplib.error_perm:
            try:
                ftp.mkd(folder)
                ftp.cwd(folder)
            except Exception as e:
                pass

    files = [f for f in os.listdir(local_dir) if f.endswith('.webp') or f.endswith('.json')]
    print(f"📤 Lade {len(files)} Radar-Dateien nach data/{remote_folder}/ hoch...")

    uploaded_count = 0
    for filename in sorted(files):
        local_path = os.path.join(local_dir, filename)
        with open(local_path, 'rb') as f_in:
            ftp.storbinary(f"STOR {filename}", f_in)
            uploaded_count += 1

    ftp.quit()
    print(f"✅ {uploaded_count} Radar-Dateien erfolgreich nach data/{remote_folder}/ hochgeladen!")


if __name__ == "__main__":
    generate_vector_radar_dataset()
