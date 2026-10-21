#!/usr/bin/env python3
"""
DWD ICON-EU Synoptik- & Modellkarten Generator (localwx PRO) - WXCHARTS EDITION
================================================================================
Professioneller, vollflächiger 16:10 Renderer für DWD ICON-EU Modellkarten im
ikonischen WXCHARTS-Stil:

Features:
- Vollflächiges 16:10 Widescreen-Format (1600x1000) ohne schwarze Ränder
- Optimierte Lambert-Conformal-Projektion für Europa & den gesamten Nordatlantik
- Originalgetreue WXCHARTS Farbskala (-40°C bis +40°C) mit prägnanter 0°C-Schneegrenze
- Weiße MSLP-Isobaren (Bodendruck alle 4 hPa) mit schwarzem Text-Halo
- Automatische Erkennung & Markierung von Hoch- ("H") und Tiefdruckzentren ("T")
- Edle WXCHARTS-Headerbox & Farbleiste
- Deadlock-freies Multiprocessing & robuster FTPS-Upload
"""

import os
import sys
import json
import time
import bz2
import zipfile
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

# Scipy für Zentren-Erkennung (Tiefs / Hochs)
try:
    import scipy.ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

# Cartopy Setup
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Eccodes für DWD GRIB2 Dekodierung
try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False


# ==============================================================================
# 1. OFFLINE-DOWNLOAD DER KARTEN-GEOMETRIEN (KEIN NETWORK-HANG / DEADLOCK)
# ==============================================================================

def setup_cartopy_offline():
    """
    Lädt Küsten und Grenzen einmalig vorab herunter und entpackt sie im Cartopy-Cache.
    Verhindert Lock-Konflikte und Endlos-Hänger in Subprozessen.
    """
    if not CARTOPY_AVAILABLE:
        return

    data_dir = cartopy.config.get('data_dir', os.path.expanduser('~/.local/share/cartopy'))
    target_dir = os.path.join(data_dir, 'shapefiles', 'natural_earth', 'physical')
    target_cultural = os.path.join(data_dir, 'shapefiles', 'natural_earth', 'cultural')
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_cultural, exist_ok=True)

    downloads = [
        ("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip", target_dir, "ne_50m_coastline.shp"),
        ("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_boundary_lines_land.zip", target_cultural, "ne_50m_admin_0_boundary_lines_land.shp"),
        ("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_1_states_provinces_lines.zip", target_cultural, "ne_50m_admin_1_states_provinces_lines.shp"),
        ("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_lakes.zip", target_dir, "ne_50m_lakes.shp")
    ]

    print("🗺️ Initialisiere Geometrie-Features...")
    for url, out_folder, check_file in downloads:
        if os.path.exists(os.path.join(out_folder, check_file)):
            continue
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            zip_path = os.path.join(out_folder, "temp.zip")
            with urllib.request.urlopen(req, timeout=15) as resp, open(zip_path, 'wb') as f:
                f.write(resp.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(out_folder)
            os.remove(zip_path)
        except Exception as e:
            print(f"⚠️ Hinweis bei {check_file}: {e} (Fallback aktiv)")


# ==============================================================================
# 2. WXCHARTS FARBSKALEN & PALETTEN
# ==============================================================================

# Original WXCHARTS 40-Stufen Temperatur-Farbskala (-40°C bis +40°C in 2°C-Schritten)
# Mit dem berühmten, glasklaren Kontrast-Übergang an der 0°C-Gefriergrenze
WXCHARTS_TEMP_COLORS = [
    '#320436',  # -40 bis -38
    '#46084c',  # -38 bis -36
    '#5c0d63',  # -36 bis -34
    '#72127b',  # -34 bis -32
    '#8a1894',  # -32 bis -30
    '#a21fad',  # -30 bis -28
    '#b826be',  # -28 bis -26
    '#ca2ecc',  # -26 bis -24
    '#da35d8',  # -24 bis -22
    '#ea47e3',  # -22 bis -20
    '#9b5ee6',  # -20 bis -18
    '#685ce6',  # -18 bis -16
    '#335ee0',  # -16 bis -14
    '#207ee8',  # -14 bis -12
    '#139af2',  # -12 bis -10
    '#1eb4f6',  # -10 bis -8
    '#3dccf8',  # -8 bis -6
    '#74e2fb',  # -6 bis -4
    '#aef2fc',  # -4 bis -2
    '#ddf9fd',  # -2 bis 0 (Helles Eis-Cyan)
    # --- 0°C SCHARFE GRENZE ZU SAFTIGEM FRÜHLINGSGRÜN ---
    '#82e734',  # 0 bis 2
    '#5bcf21',  # 2 bis 4
    '#36be15',  # 4 bis 6
    '#1ca50d',  # 6 bis 8
    '#0d8d0c',  # 8 bis 10
    '#259008',  # 10 bis 12
    '#52a207',  # 12 bis 14
    '#8ebc08',  # 14 bis 16
    '#d2d708',  # 16 bis 18
    '#f5c507',  # 18 bis 20
    '#f59f07',  # 20 bis 22
    '#f57a07',  # 22 bis 24
    '#f05008',  # 24 bis 26
    '#e42a0a',  # 26 bis 28
    '#d10e0e',  # 28 bis 30
    '#b20314',  # 30 bis 32
    '#8e011a',  # 32 bis 34
    '#6b0122',  # 34 bis 36
    '#4e0026',  # 36 bis 38
    '#360020',  # 38 bis 40
]
WXCHARTS_TEMP_LEVELS = np.arange(-40, 42, 2)
WXCHARTS_TEMP_CMAP = mcolors.ListedColormap(WXCHARTS_TEMP_COLORS, name='wxcharts_temp')
WXCHARTS_TEMP_CMAP.set_under(WXCHARTS_TEMP_COLORS[0])
WXCHARTS_TEMP_CMAP.set_over(WXCHARTS_TEMP_COLORS[-1])


# WXCHARTS 500 hPa Geopotentialhöhe (gpdm) Farbskala (24 Stufen)
WXCHARTS_Z500_COLORS = [
    '#4a044e', '#6b116d', '#86198f', '#a21caf', '#c026d3',  # 496 - 516 (Polartiefs)
    '#1e3a8a', '#1d4ed8', '#2563eb', '#0284c7', '#0ea5e9',  # 516 - 536 (Tröge)
    '#059669', '#10b981', '#34d399', '#6ee7b7',              # 536 - 552 (Übergang)
    '#84cc16', '#a3e635', '#eab308', '#f59e0b', '#f97316',  # 552 - 572 (Keile / Rücken)
    '#ef4444', '#dc2626', '#b91c1c', '#991b1b', '#701a75'   # 572 - 592 (Subtropenhochs)
]
WXCHARTS_Z500_LEVELS = np.arange(496, 596, 4)
WXCHARTS_Z500_CMAP = mcolors.ListedColormap(WXCHARTS_Z500_COLORS, name='wxcharts_z500')
WXCHARTS_Z500_CMAP.set_under(WXCHARTS_Z500_COLORS[0])
WXCHARTS_Z500_CMAP.set_over(WXCHARTS_Z500_COLORS[-1])


# WXCHARTS 300 hPa Jetstream Windgeschwindigkeit (km/h) (11 Stufen)
WXCHARTS_JET_COLORS = [
    '#0f172a',  # 40 - 70 (Hintergrund / schwacher Wind)
    '#083344',  # 70 - 100
    '#0e7490',  # 100 - 130 (Cyan)
    '#0d9488',  # 130 - 160 (Türkis)
    '#16a34a',  # 160 - 190 (Grün)
    '#ca8a04',  # 190 - 220 (Gelb)
    '#ea580c',  # 220 - 250 (Orange)
    '#dc2626',  # 250 - 280 (Rot)
    '#c026d3',  # 280 - 310 (Magenta)
    '#f472b6',  # 310 - 340 (Rosa)
    '#ffffff',  # 340 - 370 (Extremer Kernjet)
]
WXCHARTS_JET_LEVELS = np.arange(40, 400, 30)
WXCHARTS_JET_CMAP = mcolors.ListedColormap(WXCHARTS_JET_COLORS, name='wxcharts_jet')
WXCHARTS_JET_CMAP.set_under(WXCHARTS_JET_COLORS[0])
WXCHARTS_JET_CMAP.set_over(WXCHARTS_JET_COLORS[-1])


# ==============================================================================
# 3. DWD GRIB2 DOWNLOAD & PARSING
# ==============================================================================

def get_latest_icon_eu_run():
    now = datetime.now(timezone.utc)
    current_hour = now.hour
    if current_hour >= 18:
        ref_hour = 12
        ref_date = now.replace(hour=ref_hour, minute=0, second=0, microsecond=0)
    elif current_hour >= 12:
        ref_hour = 6
        ref_date = now.replace(hour=ref_hour, minute=0, second=0, microsecond=0)
    elif current_hour >= 6:
        ref_hour = 0
        ref_date = now.replace(hour=ref_hour, minute=0, second=0, microsecond=0)
    else:
        ref_hour = 18
        ref_date = (now - timedelta(days=1)).replace(hour=ref_hour, minute=0, second=0, microsecond=0)
    return ref_date


def fetch_dwd_file(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (localwx PRO WXCharts)'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                compressed = resp.read()
                return bz2.decompress(compressed)
        except Exception:
            time.sleep(0.5)
    return None


def parse_grib_array(grib_bytes):
    """
    Parst GRIB2-Nachrichten und normalisiert Koordinaten sowie Scanrichtung.
    """
    if not ECCODES_AVAILABLE or grib_bytes is None:
        return None, None, None
    msg_id = None
    try:
        msg_id = eccodes.codes_new_from_message(grib_bytes)
        ni = eccodes.codes_get(msg_id, 'Ni')
        nj = eccodes.codes_get(msg_id, 'Nj')

        try:
            lat_first = eccodes.codes_get_double(msg_id, 'latitudeOfFirstGridPointInDegrees')
            lat_last = eccodes.codes_get_double(msg_id, 'latitudeOfLastGridPointInDegrees')
            lon_first = eccodes.codes_get_double(msg_id, 'longitudeOfFirstGridPointInDegrees')
            lon_last = eccodes.codes_get_double(msg_id, 'longitudeOfLastGridPointInDegrees')
        except Exception:
            lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint') / 1e6
            lat_last = eccodes.codes_get(msg_id, 'latitudeOfLastGridPoint') / 1e6
            lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint') / 1e6
            lon_last = eccodes.codes_get(msg_id, 'longitudeOfLastGridPoint') / 1e6

        # Längengrade von 0..360 auf -180..+180 normalisieren
        if lon_first > 180.0:
            lon_first -= 360.0
        if lon_last > 180.0:
            lon_last -= 360.0

        j_scans_pos = eccodes.codes_get(msg_id, 'jScansPositively')
        values = eccodes.codes_get_values(msg_id)
        arr = values.reshape((nj, ni))

        lats = np.linspace(min(lat_first, lat_last), max(lat_first, lat_last), nj)
        lons = np.linspace(min(lon_first, lon_last), max(lon_first, lon_last), ni)

        # Wenn von Nord nach Süd gescannt wurde, Array vertikal umdrehen (damit Index 0 Süden ist)
        if j_scans_pos == 0:
            arr = np.flipud(arr)

        return arr, lats, lons
    except Exception:
        return None, None, None
    finally:
        if msg_id is not None:
            try:
                eccodes.codes_release(msg_id)
            except Exception:
                pass


# ==============================================================================
# 4. ZENTREN-DETEKTOR (HOCHS & TIEFS IM WXCHARTS-STIL)
# ==============================================================================

def find_pressure_extrema(pmsl_grid, lats, lons, min_dist=24):
    """
    Findet lokale Tief- ("T") und Hochdruckzentren ("H") in der Bodendruckmatrix.
    """
    if not SCIPY_AVAILABLE or pmsl_grid is None:
        return [], []

    try:
        # Sanfte Glättung gegen Kleinst-Rauschen
        smoothed = ndi.gaussian_filter(pmsl_grid, sigma=2.2)

        local_min = (smoothed == ndi.minimum_filter(smoothed, size=min_dist, mode='nearest'))
        local_max = (smoothed == ndi.maximum_filter(smoothed, size=min_dist, mode='nearest'))

        lows = []
        highs = []

        nj, ni = pmsl_grid.shape
        margin = 12

        min_indices = np.argwhere(local_min)
        for r, c in min_indices:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                val = smoothed[r, c]
                # Echtes signifikantes Tiefdruckzentrum
                if val <= 1012.0:
                    lows.append((float(lons[c]), float(lats[r]), int(round(val))))

        max_indices = np.argwhere(local_max)
        for r, c in max_indices:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                val = smoothed[r, c]
                # Echtes signifikantes Hochdruckzentrum
                if val >= 1018.0:
                    highs.append((float(lons[c]), float(lats[r]), int(round(val))))

        return lows, highs
    except Exception:
        return [], []


# ==============================================================================
# 5. KARTEN-RENDERING EINES KOMPLETTEN ZEITSCHRITTS (TASK)
# ==============================================================================

def render_step_task(task_args):
    """
    Rendert alle 8 Karten (4 Parameter x 2 Domains) eines Zeitschritts autark im Worker-Prozess.
    Erzeugt 100% randlose, vollflächige 16:10 WXCharts-Karten (1600x1000).
    """
    lead_h, date_str, run_str, run_date_iso, base_url, output_base = task_args
    run_date = datetime.fromisoformat(run_date_iso)
    valid_date = run_date + timedelta(hours=lead_h)
    lead_str = f"{lead_h:03d}"

    url_map = {
        't850': f"{base_url}/t/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_850_T.grib2.bz2",
        'fi850': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_850_FI.grib2.bz2",
        'fi500': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_500_FI.grib2.bz2",
        'fi300': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_FI.grib2.bz2",
        'pmsl': f"{base_url}/pmsl/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_PMSL.grib2.bz2",
        'u300': f"{base_url}/u/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_U.grib2.bz2",
        'v300': f"{base_url}/v/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_V.grib2.bz2",
        't2m': f"{base_url}/t_2m/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_T_2M.grib2.bz2"
    }

    raw_bytes = {}
    with ThreadPoolExecutor(max_workers=8) as dl_exec:
        fut_map = {dl_exec.submit(fetch_dwd_file, url): key for key, url in url_map.items()}
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            raw_bytes[key] = fut.result()

    t850_arr, lats, lons = parse_grib_array(raw_bytes.get('t850'))
    fi850_arr, _, _ = parse_grib_array(raw_bytes.get('fi850'))
    fi500_arr, _, _ = parse_grib_array(raw_bytes.get('fi500'))
    fi300_arr, _, _ = parse_grib_array(raw_bytes.get('fi300'))
    pmsl_arr, _, _ = parse_grib_array(raw_bytes.get('pmsl'))
    u300_arr, _, _ = parse_grib_array(raw_bytes.get('u300'))
    v300_arr, _, _ = parse_grib_array(raw_bytes.get('v300'))
    t2m_arr, _, _ = parse_grib_array(raw_bytes.get('t2m'))

    if lats is None or lons is None:
        return lead_h, []

    pmsl_hpa = (pmsl_arr / 100.0) if pmsl_arr is not None else None

    data = {
        't850': (t850_arr - 273.15) if t850_arr is not None else None,
        'fi850': (fi850_arr / 98.0665) if fi850_arr is not None else None,
        'fi500': (fi500_arr / 98.0665) if fi500_arr is not None else None,
        'fi300': (fi300_arr / 98.0665) if fi300_arr is not None else None,
        'pmsl': pmsl_hpa,
        'jet300': (np.sqrt(u300_arr**2 + v300_arr**2) * 3.6) if (u300_arr is not None and v300_arr is not None) else None,
        't2m': (t2m_arr - 273.15) if t2m_arr is not None else None
    }

    # Berechne Tief- & Hochdruckzentren für MSLP Isobaren
    lows, highs = find_pressure_extrema(pmsl_hpa, lats, lons)

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # Projektionen mit zentrierter Ausrichtung
    proj_europe = ccrs.LambertConformal(
        central_longitude=5.0,
        central_latitude=52.0,
        standard_parallels=(35.0, 65.0)
    ) if CARTOPY_AVAILABLE else None

    proj_germany = ccrs.LambertConformal(
        central_longitude=10.5,
        central_latitude=51.0,
        standard_parallels=(45.0, 55.0)
    ) if CARTOPY_AVAILABLE else None

    # Exakte 16:10 Bounding-Boxen in Projektionsmetern
    # Verhältnis: (x_max - x_min) / (y_max - y_min) == 1.60000000 (100% randlos!)
    extent_europe = (-2900000, 2700000, -1550000, 1950000)   # 5600 km x 3500 km
    extent_germany = (-720000, 720000, -450000, 450000)      # 1440 km x 900 km

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    generated_frames = []

    for p in params:
        # WXCHARTS Parameter-Konfiguration
        if p == 't850_gp':
            field_val = data['t850']
            contour_val = data['pmsl']  # WXCHARTS Standard: 850 hPa Temp MIT Bodendruck-Isobaren!
            title_str = "850 hPa Temperatur (°C) & Bodendruck (hPa)"
            unit_str = "°C"
            cmap, levels = WXCHARTS_TEMP_CMAP, WXCHARTS_TEMP_LEVELS
            contour_levels = np.arange(960, 1052, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(-40, 44, 4)
            is_pmsl_contour = True

        elif p == 'z500_mslp':
            field_val = data['fi500']
            contour_val = data['pmsl']
            title_str = "500 hPa Geopotentialhöhe (gpdm) & Bodendruck (hPa)"
            unit_str = "gpdm"
            cmap, levels = WXCHARTS_Z500_CMAP, WXCHARTS_Z500_LEVELS
            contour_levels = np.arange(960, 1052, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(496, 596, 8)
            is_pmsl_contour = True

        elif p == 'jet300':
            field_val = data['jet300']
            contour_val = data['fi300']
            title_str = "300 hPa Wind & Jetstream (km/h) & 300 hPa Isohypsen"
            unit_str = "km/h"
            cmap, levels = WXCHARTS_JET_CMAP, WXCHARTS_JET_LEVELS
            contour_levels = np.arange(840, 988, 8)
            contour_fmt = "%d"
            cbar_ticks = np.arange(40, 370, 30)
            is_pmsl_contour = False

        else:  # t2m_wind
            field_val = data['t2m']
            contour_val = data['pmsl']
            title_str = "2m Temperatur (°C) & Bodendruck (hPa)"
            unit_str = "°C"
            cmap, levels = WXCHARTS_TEMP_CMAP, WXCHARTS_TEMP_LEVELS
            contour_levels = np.arange(960, 1052, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(-40, 44, 4)
            is_pmsl_contour = True

        for dom in domains:
            # 1600 x 1000 Pixel (Perfekt 16:10 für moderne Bildschirme & Mobile)
            fig = plt.figure(figsize=(16.0, 10.0), dpi=100, facecolor='#0b0f19')

            if CARTOPY_AVAILABLE:
                cur_proj = proj_europe if dom == 'europe' else proj_germany
                x_min, x_max, y_min, y_max = extent_europe if dom == 'europe' else extent_germany

                # Vollflächiges Koordinatensystem: 0% Margin (Edge-to-Edge)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=cur_proj)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_facecolor('#0b1120')  # Dunkles Ozean-Blau gegen weiße NaNs

                # 1. Unterlegte Kartenbasis (falls Randbereiche außerhalb des ICON-Gitters liegen)
                try:
                    ax.add_feature(cfeature.OCEAN, facecolor='#0b1120', zorder=0)
                    ax.add_feature(cfeature.LAND, facecolor='#111827', zorder=0)
                except Exception:
                    pass

                # 2. Ausgefüllte meteorologische Farbfelder (zorder=2)
                cf = ax.contourf(
                    lon_mesh, lat_mesh, field_val,
                    levels=levels, cmap=cmap,
                    extend='both', transform=ccrs.PlateCarree(),
                    zorder=2
                )

                # 3. Küstenlinien, Grenzen und Bundesländer im WXCharts-Stil (zorder=4)
                try:
                    # Seen (Bodensee, Müritz, Genfersee etc.)
                    ax.add_feature(
                        cfeature.LAKES.with_scale('50m'),
                        edgecolor='#0f172a', facecolor='none', linewidth=0.8, zorder=4
                    )
                    # Bundesländer / Provinzen (dezent gestrichelt für sofortige Orientierung in Deutschland)
                    states_provinces = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lines',
                        scale='50m',
                        facecolor='none'
                    )
                    ax.add_feature(
                        states_provinces,
                        edgecolor='#0f172a',
                        linewidth=0.8,
                        linestyle='--',
                        alpha=0.65,
                        zorder=4
                    )
                    # Küstenlinien (Doppel-Kontur für maximale Sichtbarkeit auf jedem Farbuntergrund)
                    ax.add_feature(
                        cfeature.COASTLINE.with_scale('50m'),
                        edgecolor='#ffffff', linewidth=2.4, alpha=0.40, zorder=4
                    )
                    ax.add_feature(
                        cfeature.COASTLINE.with_scale('50m'),
                        edgecolor='#000000', linewidth=1.4, zorder=4
                    )
                    # Nationale Außengrenzen (Ländergrenzen) - DEUTLICH, durchgezogen mit Halo
                    ax.add_feature(
                        cfeature.BORDERS.with_scale('50m'),
                        edgecolor='#ffffff', linewidth=3.2, alpha=0.55, linestyle='-', zorder=4
                    )
                    ax.add_feature(
                        cfeature.BORDERS.with_scale('50m'),
                        edgecolor='#000000', linewidth=1.7, linestyle='-', zorder=4
                    )
                except Exception:
                    ax.coastlines(resolution='110m', color='#000000', linewidth=1.5, zorder=4)

                # 4. Weiße Isobaren / Isohypsen mit schwarzem Text-Halo (zorder=5)
                if contour_val is not None:
                    cs = ax.contour(
                        lon_mesh, lat_mesh, contour_val,
                        levels=contour_levels, colors='#ffffff',
                        linewidths=1.2, alpha=0.92, zorder=5,
                        transform=ccrs.PlateCarree()
                    )
                    clabels = ax.clabel(
                        cs, inline=True, fmt=contour_fmt,
                        fontsize=9, colors='#ffffff', inline_spacing=12
                    )
                    for lbl in clabels:
                        lbl.set_fontweight('bold')
                        lbl.set_path_effects([pe.withStroke(linewidth=2.8, foreground='#000000')])

                # 5. Tief- ("T") & Hochdruckzentren ("H") markieren (nur bei Bodendruck)
                if is_pmsl_contour:
                    halo_stroke = [pe.withStroke(linewidth=3.2, foreground='#000000')]
                    # Tiefdruckzentren (kräftiges Rot)
                    for lon_c, lat_c, p_val in lows:
                        pt_proj = cur_proj.transform_point(lon_c, lat_c, ccrs.PlateCarree())
                        if x_min <= pt_proj[0] <= x_max and y_min <= pt_proj[1] <= y_max:
                            ax.text(
                                lon_c, lat_c, f"T\n{p_val}",
                                transform=ccrs.PlateCarree(),
                                color='#ef4444', fontsize=11, fontweight='black',
                                ha='center', va='center', zorder=6,
                                path_effects=halo_stroke
                            )
                    # Hochdruckzentren (leuchtendes Cyan/Blau)
                    for lon_c, lat_c, p_val in highs:
                        pt_proj = cur_proj.transform_point(lon_c, lat_c, ccrs.PlateCarree())
                        if x_min <= pt_proj[0] <= x_max and y_min <= pt_proj[1] <= y_max:
                            ax.text(
                                lon_c, lat_c, f"H\n{p_val}",
                                transform=ccrs.PlateCarree(),
                                color='#38bdf8', fontsize=11, fontweight='black',
                                ha='center', va='center', zorder=6,
                                path_effects=halo_stroke
                            )

                # Dezente Längen- & Breitengradlinien
                ax.gridlines(draw_labels=False, linewidth=0.4, color='#ffffff', alpha=0.15, linestyle='--')

            else:
                # Fallback ohne Cartopy
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor='#0b0f19')
                cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=levels, cmap=cmap, extend='both')
                if contour_val is not None:
                    cs = ax.contour(lon_mesh, lat_mesh, contour_val, levels=contour_levels, colors='#ffffff', linewidths=1.1)
                    ax.clabel(cs, inline=True, fmt=contour_fmt, fontsize=9, colors='#ffffff')

            # ==================================================================
            # 6. WXCHARTS HEADER OVERLAY (OBEN LINKS)
            # ==================================================================
            init_str = run_date.strftime("%a, %d. %b %H:00 UTC")
            valid_str = valid_date.strftime("%a, %d. %b %H:00 UTC")

            header_text = (
                f"localwx.de  •  DWD ICON-EU (0.0625°)  •  {title_str}\n"
                f"Modell-Lauf: {init_str}   |   Gültig: {valid_str} (+{lead_h:02d}h)"
            )
            ax.text(
                0.015, 0.985, header_text,
                transform=ax.transAxes,
                color='#ffffff', fontsize=10.5, fontweight='bold',
                linespacing=1.35, va='top', ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.45',
                    facecolor='#0b0f19',
                    alpha=0.88,
                    edgecolor='#334155',
                    linewidth=0.8
                ),
                zorder=10
            )

            # Oben Rechts: WXCharts-Badge
            domain_label = "EUROPA & NORDATLANTIK" if dom == 'europe' else "MITTELEUROPA"
            ax.text(
                0.985, 0.985, f"localwx PRO  •  {domain_label}",
                transform=ax.transAxes,
                color='#38bdf8', fontsize=9.5, fontweight='bold',
                va='top', ha='right',
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    facecolor='#0b0f19',
                    alpha=0.88,
                    edgecolor='#0284c7',
                    linewidth=0.8
                ),
                zorder=10
            )

            # ==================================================================
            # 7. WXCHARTS BOTTOM COLORBAR OVERLAY (UNTEN)
            # ==================================================================
            # Hintergrundleiste für die Legende
            cbar_bg = fig.add_axes([0.02, 0.015, 0.96, 0.045], facecolor='#0b0f19')
            cbar_bg.patch.set_alpha(0.88)
            cbar_bg.patch.set_edgecolor('#334155')
            cbar_bg.patch.set_linewidth(0.8)
            cbar_bg.set_xticks([])
            cbar_bg.set_yticks([])

            # Einheit sauber links im Kasten platziert (keine Überlappung mit Skalen-Ticks)
            cbar_bg.text(
                0.012, 0.5, f"({unit_str})",
                color='#38bdf8', fontsize=8.5, fontweight='bold',
                va='center', ha='left'
            )

            # Eigentliche Farbskala mit ausreichend Abstand
            cbar_ax = fig.add_axes([0.065, 0.024, 0.885, 0.022])
            cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=cbar_ticks)
            cbar.ax.tick_params(labelsize=8.5, colors='#ffffff', length=3, width=0.8)

            filename = f"frame_{lead_h:03d}.webp"
            filepath = os.path.join(output_base, p, dom, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, format='webp', dpi=100, facecolor='#0b0f19', edgecolor='none')
            plt.close(fig)

            generated_frames.append({
                "param": p,
                "domain": dom,
                "lead_h": lead_h,
                "file": f"{p}/{dom}/{filename}",
                "valid_time": valid_date.isoformat(),
                "valid_label": valid_date.strftime("%a %d.%m. %H:%M")
            })

    return lead_h, generated_frames


# ==============================================================================
# 6. PIPELINE GENERATOR
# ==============================================================================

def generate_synoptic_dataset():
    start_time = time.time()
    print("🚀 Starte DWD ICON Synoptik- & Modellkarten Generator (localwx PRO WXCharts Edition)...")

    setup_cartopy_offline()

    output_base = "./dist/synoptic"
    os.makedirs(output_base, exist_ok=True)

    run_date = get_latest_icon_eu_run()
    date_str = run_date.strftime("%Y%m%d")
    run_str = f"{run_date.hour:02d}"
    print(f"📡 Verwende Modell-Lauf: {run_date.strftime('%Y-%m-%d %H:00 UTC')}")

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    steps = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 72, 84, 96, 120]
    print(f"📦 Verarbeite {len(steps)} Zeitschritte (0h bis +120h) mit {len(steps)*8} Karten...")

    manifest = {
        "model": "DWD ICON-EU",
        "init_time": run_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aspect_ratio": "16:10",
        "step_interval_h": 3,
        "max_lead_h": 120,
        "parameters": params,
        "domains": domains,
        "steps": steps,
        "frames": {p: {dom: [] for dom in domains} for p in params}
    }

    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_str}"

    task_args_list = [
        (lead_h, date_str, run_str, run_date.isoformat(), base_url, output_base)
        for lead_h in steps
    ]

    max_procs = min(os.cpu_count() or 4, 4)
    print(f"⚡ Starte Multiprocess-Rendering mit {max_procs} Workern...")

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = {executor.submit(render_step_task, args): args[0] for args in task_args_list}
        completed = 0
        for future in as_completed(futures):
            lead_h = futures[future]
            try:
                _, frames = future.result()
                for f in frames:
                    manifest["frames"][f["param"]][f["domain"]].append(f)
                completed += 1
                print(f"   ↳ [{completed}/{len(steps)}] Zeitschritt +{lead_h:02d}h fertiggestellt.")
            except Exception as e:
                print(f"⚠️ Fehler bei Zeitschritt +{lead_h}h: {e}")

    for p in params:
        for dom in domains:
            manifest["frames"][p][dom].sort(key=lambda x: x['lead_h'])

    manifest_path = os.path.join(output_base, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✅ Alle 168 synoptischen Modellkarten in {duration}s erfolgreich generiert!")

    upload_synoptic_to_ftp(output_base)


def upload_synoptic_to_ftp(local_dir):
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password:
        print("ℹ️ Keine FTP-Zugangsdaten. Überspringe Upload.")
        return

    print("\n📡 Verbinde mit FTP-Server für 'data/synoptic/'...")
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
    for folder in ["data", "synoptic"]:
        try:
            ftp.cwd(folder)
        except ftplib.error_perm:
            try:
                ftp.mkd(folder)
                ftp.cwd(folder)
            except Exception:
                pass

    uploaded_count = 0
    for root, dirs, files in os.walk(local_dir):
        rel_dir = os.path.relpath(root, local_dir)
        if rel_dir != ".":
            for part in rel_dir.split(os.sep):
                try:
                    ftp.mkd(part)
                except Exception:
                    pass
                try:
                    ftp.cwd(part)
                except Exception:
                    pass

        for file in files:
            if file.endswith('.webp') or file.endswith('.json'):
                full_path = os.path.join(root, file)
                with open(full_path, 'rb') as f_in:
                    ftp.storbinary(f"STOR {file}", f_in)
                    uploaded_count += 1

        if rel_dir != ".":
            depth = len(rel_dir.split(os.sep))
            for _ in range(depth):
                ftp.cwd('..')

    ftp.quit()
    print(f"✅ {uploaded_count} synoptische Modellkarten erfolgreich nach data/synoptic/ hochgeladen!")


if __name__ == "__main__":
    generate_synoptic_dataset()
