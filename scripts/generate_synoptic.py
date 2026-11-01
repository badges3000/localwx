#!/usr/bin/env python3
"""
DWD ICON-EU Synoptik- & Modellkarten Generator (localwx PRO) - WXCHARTS EDITION
================================================================================
Professioneller, vollflächiger 16:10 Renderer für DWD ICON-EU Modellkarten im
ikonischen WXCHARTS-Stil:

Features & Optimierungen:
- Vollflächiges 16:10 Widescreen-Format (1600x1000) ohne schwarze Ränder
- Perfekt abgestimmte Lambert-Conformal-Projektionen für Europa & Mitteleuropa,
  die zu 100% innerhalb der DWD ICON-EU Modellgrenzen liegen (keine diagonalen
  schwarzen Schnittkanten mehr im Atlantik!)
- Originalgetreue WXCHARTS Farbskala (-40°C bis +40°C) mit sichtbarem Arktis-Cyan
  an der 0°C-Gefriergrenze (kein ausgewaschenes papierweißes "Loch" in Mitteleuropa)
- Seidenweiche, geglättete MSLP-Isobaren mit schwarzem Text-Halo
- Intelligenter Druckzentren-Detektor (H & T):
  * Synoptische Filterung gegen Kleinst-Rauschen & Alpen-Dipole
  * Plateau-Deduplizierung (verhindert doppelte 'HH' / 'TT' Markierungen)
  * Prominenz-Filter (echte Druckzentren statt flacher Druckfelder)
  * Geografische Non-Maximum-Suppression (NMS) für aufgeräumte, professionelle Karten
- Edles deutsches Header-Overlay oben links; Freihaltung der oberen rechten Ecke
  für das dynamische Web-Badge der Website
- Ausgereifte Lauf-Erkennung mit HEAD-Check und automatischem stabilen Fallback
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

# Scipy für Zentren-Erkennung (Tiefs / Hochs) und Konturglättung
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
# 1. DEUTSCHE DATUMSFORMATIERUNG
# ==============================================================================

GERMAN_WEEKDAYS = {
    0: 'Mo', 1: 'Di', 2: 'Mi', 3: 'Do', 4: 'Fr', 5: 'Sa', 6: 'So'
}
GERMAN_MONTHS = {
    1: 'Jan', 2: 'Feb', 3: 'Mär', 4: 'Apr', 5: 'Mai', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dez'
}

def format_german_date(dt):
    wd = GERMAN_WEEKDAYS[dt.weekday()]
    mo = GERMAN_MONTHS[dt.month]
    return f"{wd}, {dt.day:02d}. {mo} {dt.hour:02d}:00 UTC"


# ==============================================================================
# 2. OFFLINE-DOWNLOAD DER KARTEN-GEOMETRIEN (KEIN NETWORK-HANG / DEADLOCK)
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
# 3. WXCHARTS FARBSKALEN & PALETTEN
# ==============================================================================

# WXCHARTS 40-Stufen Temperatur-Farbskala (-40°C bis +40°C in 2°C-Schritten)
# Mit optimiertem, edlem Arktis-Cyan (-2°C bis 0°C), das saturiert genug ist,
# um niemals wie ein leeres weißes Datenloch zu wirken, und dennoch den
# berühmten Kontrast-Übergang an der 0°C-Gefriergrenze liefert.
WXCHARTS_TEMP_COLORS = [
    # -40 bis -20: Polare & Arktische Kaltluft (Magenta, Violett, Dunkelblau)
    '#feb2be',  # -40 bis -38
    '#fb90a2',  # -38 bis -36
    '#f7718c',  # -36 bis -34
    '#f0557c',  # -34 bis -32
    '#e43e71',  # -32 bis -30
    '#d42b6a',  # -30 bis -28
    '#bf1d67',  # -28 bis -26
    '#a81768',  # -26 bis -24
    '#8d186c',  # -24 bis -22
    '#702073',  # -22 bis -20
    '#522b7d',  # -20 bis -18
    '#373a8c',  # -18 bis -16
    '#254d9e',  # -16 bis -14
    '#1b62b0',  # -14 bis -12
    '#1578c2',  # -12 bis -10
    '#158ed1',  # -10 bis -8
    '#20a3de',  # -8 bis -6
    '#33b8e8',  # -6 bis -4
    '#50cbef',  # -4 bis -2
    '#8be1f6',  # -2 bis 0 (Klares, edles Eis-Cyan - kein ausgewaschenes Papierweiß!)
    # --- 0°C SCHARFE GRENZE ZU FRISCHEM FRÜHLINGSGRÜN ---
    '#38c227',  # 0 bis 2
    '#28af1b',  # 2 bis 4
    '#1d9c12',  # 4 bis 6
    '#15880c',  # 6 bis 8
    '#17780f',  # 8 bis 10
    '#368b12',  # 10 bis 12
    '#6aa312',  # 12 bis 14
    '#a2bc10',  # 14 bis 16
    '#d5d20f',  # 16 bis 18
    '#f8c80d',  # 18 bis 20
    '#f69f0b',  # 20 bis 22
    '#f3740b',  # 22 bis 24
    '#ed480c',  # 24 bis 26
    '#e01f0c',  # 26 bis 28
    '#c6080b',  # 28 bis 30
    '#a80112',  # 30 bis 32
    '#85001a',  # 32 bis 34
    '#630022',  # 34 bis 36
    '#470025',  # 36 bis 38
    '#2e001e',  # 38 bis 40
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
# 4. DWD GRIB2 DOWNLOAD & PARSING
# ==============================================================================

def get_latest_icon_eu_run():
    """
    Ermittelt den neuesten verfügbaren ICON-EU Modelllauf (00, 06, 12, 18 UTC).
    DWD benötigt ca. 3h 40min nach Lauf-Initialisierung zur Veröffentlichung.
    Prüft vorab per HEAD-Request die Erreichbarkeit und fällt bei Verzögerung
    automatisch sauber auf den vorherigen Lauf zurück.
    """
    now = datetime.now(timezone.utc)
    ref_time = now - timedelta(hours=3, minutes=40)
    ref_hour = (ref_time.hour // 6) * 6
    run_date = ref_time.replace(hour=ref_hour, minute=0, second=0, microsecond=0)

    date_str = run_date.strftime("%Y%m%d")
    run_str = f"{run_date.hour:02d}"
    check_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_str}/pmsl/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_000_PMSL.grib2.bz2"

    try:
        req = urllib.request.Request(check_url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0 (localwx PRO)'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                return run_date
    except Exception:
        pass

    # Falls der neueste Lauf noch im DWD-Upload steckt, stabilen Vorlauf (-6h) verwenden
    fallback_date = run_date - timedelta(hours=6)
    print(f"ℹ️ DWD-Lauf {run_date.strftime('%Y-%m-%d %H:00 UTC')} noch nicht vollständig. Verwende stabilen Vorlauf {fallback_date.strftime('%Y-%m-%d %H:00 UTC')}.")
    return fallback_date


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
# 5. ZENTREN-DETEKTOR (HOCHS & TIEFS IM WXCHARTS-STIL)
# ==============================================================================

def find_pressure_extrema(pmsl_grid, lats, lons, domain='europe'):
    """
    Findet meteorologisch signifikante Hoch- ("H") und Tiefdruckzentren ("T").
    Verhindert Häufungen, Mehrfachmarkierungen bei Plateaus (z.B. 'HH') und
    beschränkt die Zentren auf die dominanten synoptischen Druckgebilde.
    """
    if not SCIPY_AVAILABLE or pmsl_grid is None:
        return [], []

    try:
        # 1. Stärkere synoptische Glättung gegen orographisches & numerisches Rauschen
        # Auf dem 0.0625° Gitter entsprechen sigma=8.0 ca. 55 km
        smoothed = ndi.gaussian_filter(pmsl_grid, sigma=8.0)

        # 2. Neighborhood filter für lokale Extrema
        min_dist = 64 if domain == 'europe' else 44
        local_min = (smoothed == ndi.minimum_filter(smoothed, size=min_dist, mode='reflect'))
        local_max = (smoothed == ndi.maximum_filter(smoothed, size=min_dist, mode='reflect'))

        # 3. Plateaus und benachbarte identische Werte deduplizieren (verhindert 'HH' und 'TT')
        labeled_min, num_min = ndi.label(local_min)
        labeled_max, num_max = ndi.label(local_max)

        min_candidates = []
        for i in range(1, num_min + 1):
            coords = np.argwhere(labeled_min == i)
            r, c = coords.mean(axis=0)
            r, c = int(round(r)), int(round(c))
            val = float(smoothed[r, c])
            min_candidates.append((r, c, val))

        max_candidates = []
        for i in range(1, num_max + 1):
            coords = np.argwhere(labeled_max == i)
            r, c = coords.mean(axis=0)
            r, c = int(round(r)), int(round(c))
            val = float(smoothed[r, c])
            max_candidates.append((r, c, val))

        # 4. Prominenz-Filter (Zentrum muss sich signifikant von seiner Umgebung abheben)
        nj, ni = pmsl_grid.shape
        margin = 25

        valid_lows = []
        for r, c, val in min_candidates:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                ring = smoothed[max(0, r-28):min(nj, r+29), max(0, c-28):min(ni, c+29)]
                prominence = float(np.mean(ring) - val)
                # Tiefdruck: mind. 0.8 hPa tiefer als Umgebung & Druck <= 1016 hPa
                if prominence >= 0.8 and val <= 1016.0:
                    valid_lows.append((float(lons[c]), float(lats[r]), int(round(val)), val))

        valid_highs = []
        for r, c, val in max_candidates:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                ring = smoothed[max(0, r-28):min(nj, r+29), max(0, c-28):min(ni, c+29)]
                prominence = float(val - np.mean(ring))
                # Hochdruck: mind. 0.8 hPa höher als Umgebung & Druck >= 1018 hPa
                if prominence >= 0.8 and val >= 1018.0:
                    valid_highs.append((float(lons[c]), float(lats[r]), int(round(val)), val))

        # 5. Non-Maximum Suppression (NMS) mit Mindestabstand in Kilometern
        min_sep_km = 450.0 if domain == 'europe' else 260.0

        def distance_km(lon1, lat1, lon2, lat2):
            dlat = (lat2 - lat1) * 111.0
            dlon = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) * 0.5))
            return np.sqrt(dlat * dlat + dlon * dlon)

        # Tiefs nach tiefstem Druck zuerst sortieren
        valid_lows.sort(key=lambda x: x[3])
        filtered_lows = []
        for item in valid_lows:
            lon_c, lat_c, p_val, _ = item
            if all(distance_km(lon_c, lat_c, fl[0], fl[1]) > min_sep_km for fl in filtered_lows):
                filtered_lows.append((lon_c, lat_c, p_val))

        # Hochs nach höchstem Druck zuerst sortieren
        valid_highs.sort(key=lambda x: -x[3])
        filtered_highs = []
        for item in valid_highs:
            lon_c, lat_c, p_val, _ = item
            if all(distance_km(lon_c, lat_c, fh[0], fh[1]) > min_sep_km for fh in filtered_highs):
                filtered_highs.append((lon_c, lat_c, p_val))

        max_count = 6 if domain == 'europe' else 3
        return filtered_lows[:max_count], filtered_highs[:max_count]
    except Exception:
        return [], []


# ==============================================================================
# 6. KARTEN-RENDERING EINES KOMPLETTEN ZEITSCHRITTS (TASK)
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

    # Druckzentren für beide Domains mit abgestimmtem Detaillierungsgrad ermitteln
    lows_europe, highs_europe = find_pressure_extrema(pmsl_hpa, lats, lons, domain='europe')
    lows_germany, highs_germany = find_pressure_extrema(pmsl_hpa, lats, lons, domain='germany')

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)

    # Sanft vorgefiltertes PMSL für seidenweiche, ästhetische Isobaren
    pmsl_smooth = ndi.gaussian_filter(pmsl_hpa, sigma=1.6) if (SCIPY_AVAILABLE and pmsl_hpa is not None) else pmsl_hpa

    # Optimierte LambertConformal Projektionen, die 100% innerhalb der ICON-EU Grenzen liegen (-23.5°W bis 45.0°E)
    proj_europe = ccrs.LambertConformal(
        central_longitude=10.0,
        central_latitude=51.0,
        standard_parallels=(35.0, 65.0)
    ) if CARTOPY_AVAILABLE else None

    proj_germany = ccrs.LambertConformal(
        central_longitude=10.5,
        central_latitude=51.2,
        standard_parallels=(45.0, 55.0)
    ) if CARTOPY_AVAILABLE else None

    # Exakte 16:10 Bounding-Boxen in Projektionsmetern (1600x1000 ohne schwarze Ränder)
    # Europe: 3800 km x 2375 km (16:10) - deckt von Irland bis Ukraine & Skandinavien bis Mittelmeer ab
    extent_europe = (-1900000, 1900000, -1350000, 1025000)
    # Germany: 1520 km x 950 km (16:10) - perfekt zentriertes Mitteleuropa
    extent_germany = (-760000, 760000, -475000, 475000)

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    generated_frames = []

    for p in params:
        # WXCHARTS Parameter-Konfiguration
        if p == 't850_gp':
            field_val = data['t850']
            contour_val = pmsl_smooth
            title_str = "850 hPa Temperatur (°C) & Bodendruck (hPa)"
            unit_str = "°C"
            cmap, levels = WXCHARTS_TEMP_CMAP, WXCHARTS_TEMP_LEVELS
            contour_levels = np.arange(960, 1056, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(-40, 44, 4)
            is_pmsl_contour = True

        elif p == 'z500_mslp':
            field_val = data['fi500']
            contour_val = pmsl_smooth
            title_str = "500 hPa Geopotentialhöhe (gpdm) & Bodendruck (hPa)"
            unit_str = "gpdm"
            cmap, levels = WXCHARTS_Z500_CMAP, WXCHARTS_Z500_LEVELS
            contour_levels = np.arange(960, 1056, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(496, 596, 8)
            is_pmsl_contour = True

        elif p == 'jet300':
            field_val = data['jet300']
            contour_val = data['fi300']
            title_str = "300 hPa Jetstream & Höhenwind (km/h) & 300 hPa Isohypsen"
            unit_str = "km/h"
            cmap, levels = WXCHARTS_JET_CMAP, WXCHARTS_JET_LEVELS
            contour_levels = np.arange(840, 992, 8)
            contour_fmt = "%d"
            cbar_ticks = np.arange(40, 370, 30)
            is_pmsl_contour = False

        else:  # t2m_wind
            field_val = data['t2m']
            contour_val = pmsl_smooth
            title_str = "2m Temperatur (°C) & Bodendruck (hPa)"
            unit_str = "°C"
            cmap, levels = WXCHARTS_TEMP_CMAP, WXCHARTS_TEMP_LEVELS
            contour_levels = np.arange(960, 1056, 4)
            contour_fmt = "%d"
            cbar_ticks = np.arange(-40, 44, 4)
            is_pmsl_contour = True

        for dom in domains:
            # 1600 x 1000 Pixel (Vollflächig 16:10 für moderne Bildschirme & Mobile)
            fig = plt.figure(figsize=(16.0, 10.0), dpi=100, facecolor='#0b0f19')

            if CARTOPY_AVAILABLE:
                cur_proj = proj_europe if dom == 'europe' else proj_germany
                x_min, x_max, y_min, y_max = extent_europe if dom == 'europe' else extent_germany
                lows, highs = (lows_europe, highs_europe) if dom == 'europe' else (lows_germany, highs_germany)

                # Vollflächiges Koordinatensystem: 0% Margin (Edge-to-Edge)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=cur_proj)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_facecolor('#0b1120')

                # 1. Unterlegte Kartenbasis
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
                    # Bundesländer / Provinzen
                    states_provinces = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lines',
                        scale='50m',
                        facecolor='none'
                    )
                    state_alpha = 0.65 if dom == 'germany' else 0.35
                    ax.add_feature(
                        states_provinces,
                        edgecolor='#0f172a',
                        linewidth=0.8,
                        linestyle='--',
                        alpha=state_alpha,
                        zorder=4
                    )
                    # Küstenlinien (Doppel-Kontur für maximale Sichtbarkeit)
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
                        fontsize=9, colors='#ffffff', inline_spacing=18
                    )
                    for lbl in clabels:
                        lbl.set_fontweight('bold')
                        lbl.set_path_effects([pe.withStroke(linewidth=2.8, foreground='#000000')])

                # 5. Tief- ("T") & Hochdruckzentren ("H") markieren (nur bei Bodendruck)
                if is_pmsl_contour:
                    halo_stroke = [pe.withStroke(linewidth=3.5, foreground='#000000')]
                    x_pad = (x_max - x_min) * 0.04
                    y_pad = (y_max - y_min) * 0.06

                    # Tiefdruckzentren (kräftiges Rot)
                    for lon_c, lat_c, p_val in lows:
                        pt_proj = cur_proj.transform_point(lon_c, lat_c, ccrs.PlateCarree())
                        if (x_min + x_pad) <= pt_proj[0] <= (x_max - x_pad) and (y_min + y_pad) <= pt_proj[1] <= (y_max - y_pad):
                            ax.text(
                                lon_c, lat_c, f"T\n{p_val}",
                                transform=ccrs.PlateCarree(),
                                color='#ef4444', fontsize=11.5, fontweight='black',
                                ha='center', va='center', zorder=6,
                                path_effects=halo_stroke
                            )
                    # Hochdruckzentren (leuchtendes Cyan/Blau)
                    for lon_c, lat_c, p_val in highs:
                        pt_proj = cur_proj.transform_point(lon_c, lat_c, ccrs.PlateCarree())
                        if (x_min + x_pad) <= pt_proj[0] <= (x_max - x_pad) and (y_min + y_pad) <= pt_proj[1] <= (y_max - y_pad):
                            ax.text(
                                lon_c, lat_c, f"H\n{p_val}",
                                transform=ccrs.PlateCarree(),
                                color='#38bdf8', fontsize=11.5, fontweight='black',
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
            # 7. WXCHARTS HEADER OVERLAY (OBEN LINKS)
            # ==================================================================
            init_str = format_german_date(run_date)
            valid_str = format_german_date(valid_date)
            domain_label = "Europa" if dom == 'europe' else "Mitteleuropa"

            header_text = (
                f"localwx.de  •  DWD ICON-EU (0.0625°)  •  {domain_label}\n"
                f"{title_str}\n"
                f"Modell-Lauf: {init_str}   |   Gültig: {valid_str} (+{lead_h:02d}h)"
            )
            ax.text(
                0.015, 0.985, header_text,
                transform=ax.transAxes,
                color='#ffffff', fontsize=10, fontweight='bold',
                linespacing=1.35, va='top', ha='left',
                bbox=dict(
                    boxstyle='round,pad=0.45',
                    facecolor='#0b0f19',
                    alpha=0.90,
                    edgecolor='#334155',
                    linewidth=0.8
                ),
                zorder=10
            )
            # Hinweis: Die obere rechte Ecke wird bewusst frei gelassen, da die
            # Website dort das interaktive Vorhersage-Badge darstellt.

            # ==================================================================
            # 8. WXCHARTS BOTTOM COLORBAR OVERLAY (UNTEN)
            # ==================================================================
            # Hintergrundleiste für die Legende (Glassmorphism)
            cbar_bg = fig.add_axes([0.02, 0.015, 0.96, 0.046], facecolor='#0b0f19')
            cbar_bg.patch.set_alpha(0.90)
            cbar_bg.patch.set_edgecolor('#334155')
            cbar_bg.patch.set_linewidth(0.8)
            cbar_bg.set_xticks([])
            cbar_bg.set_yticks([])

            # Einheit sauber links im Kasten platziert
            cbar_bg.text(
                0.012, 0.5, f"({unit_str})",
                color='#38bdf8', fontsize=8.5, fontweight='bold',
                va='center', ha='left'
            )

            # Farbskala mit optimalem Kontrast und Tick-Abständen
            cbar_ax = fig.add_axes([0.062, 0.024, 0.900, 0.022])
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
# 7. PIPELINE GENERATOR
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
    print(f"📦 Verarbeite bis zu {len(steps)} Zeitschritte (0h bis +120h) mit {len(steps)*8} Karten...")

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

    # Tatsächlich generierte Zeitschritte im Manifest hinterlegen
    available_steps = sorted(list(set(
        f["lead_h"] for p in params for dom in domains for f in manifest["frames"][p][dom]
    )))
    manifest["steps"] = available_steps if available_steps else steps
    if available_steps:
        manifest["max_lead_h"] = max(available_steps)

    manifest_path = os.path.join(output_base, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✅ Synoptische Modellkarten in {duration}s erfolgreich generiert!")

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
