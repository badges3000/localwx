#!/usr/bin/env python3
"""
Synoptik- & Modellkarten Generator (localwx PRO) - WXCHARTS EDITION
=====================================================================
Professioneller, vollflächiger 16:10 Renderer für globale & regionale
Wettermodelle im ikonischen WXCHARTS-Stil:

Hybrid-Modellarchitektur:
- Nordatlantik & Europa ('europe'): Globales ECMWF IFS Open Data (0.25°)
  mit vollständiger Abdeckung von der US-Ostküste / Grönland über den
  gesamten Nordatlantik bis Osteuropa (keine Modellrand-Schnittkanten mehr!)
- Mitteleuropa ('germany'): Hochauflösendes DWD ICON-EU (0.0625°, ~6.5 km)
  für maximale Detailschärfe in Deutschland, Alpenraum & Nachbarländern

Features & Optimierungen:
- Vollflächiges 16:10 Widescreen-Format (1600x1000) ohne schwarze Ränder
- Originalgetreue WXCHARTS Farbskala (-40°C bis +40°C) mit sichtbarem Arktis-Cyan
  an der 0°C-Gefriergrenze (kein ausgewaschenes papierweißes "Loch")
- Seidenweiche, auflösungsabhängig geglättete MSLP-Isobaren mit Text-Halo
- Auflösungsunabhängiger Druckzentren-Detektor (H & T) mit Plateau-Deduplizierung,
  Prominenz-Filter und geografischer Non-Maximum-Suppression (NMS)
- Edles deutsches Header-Overlay oben links; Freihaltung der oberen rechten Ecke
  für das dynamische Web-Badge der Website
- Deadlock-freies Multiprocessing & robuster FTPS-Upload
"""

import os
import json
import time
import bz2
import zipfile
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import ftplib

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

# Eccodes für GRIB2 Dekodierung (DWD & ECMWF)
try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False

# ECMWF Open Data Client für globale IFS-Daten (0.25°)
try:
    from ecmwf.opendata import Client as EcmwfClient
    ECMWF_AVAILABLE = True
except ImportError:
    ECMWF_AVAILABLE = False


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
# 4. GRIB2 DOWNLOAD & PARSING (DWD ICON-EU & ECMWF IFS)
# ==============================================================================

def _decode_grib_message(msg_id, crop_bounds=None):
    """
    Dekodiert eine einzelne GRIB2-Nachricht, normalisiert Längen- (-180..+180)
    und Breitengrade (Süd->Nord) und schneidet optional auf einen Zielbereich zu.
    """
    ni = eccodes.codes_get(msg_id, 'Ni')
    nj = eccodes.codes_get(msg_id, 'Nj')

    try:
        lat_first = eccodes.codes_get_double(msg_id, 'latitudeOfFirstGridPointInDegrees')
        lat_last = eccodes.codes_get_double(msg_id, 'latitudeOfLastGridPointInDegrees')
        lon_first = eccodes.codes_get_double(msg_id, 'longitudeOfFirstGridPointInDegrees')
        di = eccodes.codes_get_double(msg_id, 'iDirectionIncrementInDegrees')
        i_scans_pos = eccodes.codes_get(msg_id, 'iScansPositively')
    except Exception:
        lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint') / 1e6
        lat_last = eccodes.codes_get(msg_id, 'latitudeOfLastGridPoint') / 1e6
        lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint') / 1e6
        di = 360.0 / ni
        i_scans_pos = 1

    values = eccodes.codes_get_values(msg_id)
    arr = values.reshape((nj, ni))

    # Breitengrade exakt von Erstpunkt zu Letztpunkt aufbauen und ggf. Süd->Nord flippen
    lats = np.linspace(lat_first, lat_last, nj)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        arr = np.flipud(arr)

    # Längengrade exakt über Schrittweite aufbauen, auf [-180, 180) umbrechen & sortieren
    step_lon = di if i_scans_pos == 1 else -di
    lons = lon_first + np.arange(ni) * step_lon
    lons = ((lons + 180.0) % 360.0) - 180.0
    if not np.all(np.diff(lons) > 0):
        order = np.argsort(lons)
        lons = lons[order]
        arr = arr[:, order]

    # Für globale Gitter auf unseren Arbeitsbereich zuschneiden (verhindert ±180°-Artefakte)
    if crop_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = crop_bounds
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        lats = lats[lat_mask]
        lons = lons[lon_mask]
        arr = arr[np.ix_(lat_mask, lon_mask)]

    return arr, lats, lons


def get_latest_icon_eu_run():
    """
    Ermittelt den neuesten verfügbaren ICON-EU Modelllauf (00, 06, 12, 18 UTC).
    DWD benötigt ca. 3h 40min nach Lauf-Initialisierung zur Veröffentlichung.
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

    fallback_date = run_date - timedelta(hours=6)
    print(f"ℹ️ DWD-Lauf {run_date.strftime('%Y-%m-%d %H:00 UTC')} noch nicht vollständig. Verwende stabilen Vorlauf {fallback_date.strftime('%Y-%m-%d %H:00 UTC')}.")
    return fallback_date


def fetch_dwd_file(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (localwx PRO WXCharts)'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                compressed = resp.read()
                return bz2.decompress(compressed)
        except Exception:
            time.sleep(0.5)
    return None


def parse_grib_array(grib_bytes):
    """Parst eine einzelne DWD-GRIB2-Nachricht aus dem Arbeitsspeicher."""
    if not ECCODES_AVAILABLE or grib_bytes is None:
        return None, None, None
    msg_id = None
    try:
        msg_id = eccodes.codes_new_from_message(grib_bytes)
        return _decode_grib_message(msg_id, crop_bounds=None)
    except Exception:
        return None, None, None
    finally:
        if msg_id is not None:
            try:
                eccodes.codes_release(msg_id)
            except Exception:
                pass


def get_latest_ecmwf_run():
    """
    Ermittelt den neuesten vollständigen globalen ECMWF IFS Open-Data-Lauf (0.25°).
    Prüft step=120 im Hauptstream ('oper', 00 & 12 UTC), damit alle Zeitschritte
    bis +120h garantiert vollständig vorliegen. Probiert erst ECMWF-Mirror, dann AWS.
    """
    if not ECMWF_AVAILABLE:
        print("⚠️ ecmwf-opendata nicht installiert.")
        return None, None

    for source in ("ecmwf", "aws"):
        try:
            client = EcmwfClient(source=source, model="ifs", resol="0p25")
            latest_dt = client.latest(
                stream="oper",
                type="fc",
                step=120,
                levtype="sfc",
                param="msl"
            )
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            return latest_dt, source
        except Exception as e:
            print(f"ℹ️ ECMWF Quelle '{source}' Check fehlgeschlagen: {e}")

    return None, None


def _retrieve_ecmwf_chunk(source, date_str, time_hour, step, request_kwargs, target_path):
    for attempt in range(3):
        try:
            client = EcmwfClient(source=source, model="ifs", resol="0p25")
            client.retrieve(
                date=date_str,
                time=time_hour,
                stream="oper",
                type="fc",
                step=step,
                target=target_path,
                **request_kwargs
            )
            return target_path
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None


def fetch_ecmwf_step_data(lead_h, ecmwf_run_date, source="ecmwf"):
    """
    Lädt genau die 7 benötigten ECMWF IFS 0.25° Felder für einen Zeitschritt
    über Index/Byte-Range-Abruf in ein temporäres Verzeichnis und dekodiert sie.
    """
    if not (ECMWF_AVAILABLE and ECCODES_AVAILABLE) or ecmwf_run_date is None:
        return None, None, None

    date_str = ecmwf_run_date.strftime("%Y-%m-%d")
    time_hour = int(ecmwf_run_date.hour)
    # Großzügiger Zuschnitt für den gesamten Nordatlantik & Europa (inkl. Lambert-Ecken)
    crop_bounds = (15.0, 86.0, -95.0, 70.0)

    raw_fields = {}
    lats, lons = None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = [
            ({"levtype": "sfc", "param": ["msl", "2t"]}, os.path.join(tmpdir, "sfc.grib2")),
            ({"levtype": "pl", "levelist": 850, "param": "t"}, os.path.join(tmpdir, "pl850.grib2")),
            ({"levtype": "pl", "levelist": 500, "param": "gh"}, os.path.join(tmpdir, "pl500.grib2")),
            ({"levtype": "pl", "levelist": 300, "param": ["gh", "u", "v"]}, os.path.join(tmpdir, "pl300.grib2")),
        ]

        grib_paths = []
        with ThreadPoolExecutor(max_workers=4) as dl_exec:
            futures = [
                dl_exec.submit(_retrieve_ecmwf_chunk, source, date_str, time_hour, lead_h, req, path)
                for req, path in chunks
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res and os.path.exists(res):
                    grib_paths.append(res)

        for path in grib_paths:
            with open(path, 'rb') as f:
                while True:
                    msg_id = eccodes.codes_grib_new_from_file(f)
                    if msg_id is None:
                        break
                    try:
                        short_name = eccodes.codes_get(msg_id, 'shortName')
                        try:
                            level = int(eccodes.codes_get(msg_id, 'level'))
                        except Exception:
                            level = 0
                        arr, msg_lats, msg_lons = _decode_grib_message(msg_id, crop_bounds=crop_bounds)

                        if lats is None:
                            lats, lons = msg_lats, msg_lons

                        if short_name == 'msl':
                            raw_fields['msl'] = arr
                        elif short_name in ('2t', 't2m'):
                            raw_fields['2t'] = arr
                        elif short_name == 't' and level == 850:
                            raw_fields['t850'] = arr
                        elif short_name in ('gh', 'z') and level == 500:
                            divisor = 10.0 if short_name == 'gh' else 98.0665
                            raw_fields['fi500'] = arr / divisor
                        elif short_name in ('gh', 'z') and level == 300:
                            divisor = 10.0 if short_name == 'gh' else 98.0665
                            raw_fields['fi300'] = arr / divisor
                        elif short_name == 'u' and level == 300:
                            raw_fields['u300'] = arr
                        elif short_name == 'v' and level == 300:
                            raw_fields['v300'] = arr
                    finally:
                        eccodes.codes_release(msg_id)

    if lats is None or lons is None:
        return None, None, None

    pmsl_hpa = (raw_fields['msl'] / 100.0) if 'msl' in raw_fields else None
    u300 = raw_fields.get('u300')
    v300 = raw_fields.get('v300')

    data = {
        't850': (raw_fields['t850'] - 273.15) if 't850' in raw_fields else None,
        'fi500': raw_fields.get('fi500'),
        'fi300': raw_fields.get('fi300'),
        'pmsl': pmsl_hpa,
        'jet300': (np.sqrt(u300**2 + v300**2) * 3.6) if (u300 is not None and v300 is not None) else None,
        't2m': (raw_fields['2t'] - 273.15) if '2t' in raw_fields else None,
    }
    return data, lats, lons


def fetch_dwd_step_data(lead_h, date_str, run_str, base_url):
    """Lädt und dekodiert alle benötigten DWD ICON-EU 0.0625° Felder für einen Zeitschritt."""
    lead_str = f"{lead_h:03d}"
    url_map = {
        't850': f"{base_url}/t/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_850_T.grib2.bz2",
        'fi500': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_500_FI.grib2.bz2",
        'fi300': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_FI.grib2.bz2",
        'pmsl': f"{base_url}/pmsl/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_PMSL.grib2.bz2",
        'u300': f"{base_url}/u/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_U.grib2.bz2",
        'v300': f"{base_url}/v/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_V.grib2.bz2",
        't2m': f"{base_url}/t_2m/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_T_2M.grib2.bz2"
    }

    raw_bytes = {}
    with ThreadPoolExecutor(max_workers=7) as dl_exec:
        fut_map = {dl_exec.submit(fetch_dwd_file, url): key for key, url in url_map.items()}
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            raw_bytes[key] = fut.result()

    t850_arr, lats, lons = parse_grib_array(raw_bytes.get('t850'))
    fi500_arr, _, _ = parse_grib_array(raw_bytes.get('fi500'))
    fi300_arr, _, _ = parse_grib_array(raw_bytes.get('fi300'))
    pmsl_arr, _, _ = parse_grib_array(raw_bytes.get('pmsl'))
    u300_arr, _, _ = parse_grib_array(raw_bytes.get('u300'))
    v300_arr, _, _ = parse_grib_array(raw_bytes.get('v300'))
    t2m_arr, _, _ = parse_grib_array(raw_bytes.get('t2m'))

    if lats is None or lons is None:
        return None, None, None

    pmsl_hpa = (pmsl_arr / 100.0) if pmsl_arr is not None else None
    data = {
        't850': (t850_arr - 273.15) if t850_arr is not None else None,
        'fi500': (fi500_arr / 98.0665) if fi500_arr is not None else None,
        'fi300': (fi300_arr / 98.0665) if fi300_arr is not None else None,
        'pmsl': pmsl_hpa,
        'jet300': (np.sqrt(u300_arr**2 + v300_arr**2) * 3.6) if (u300_arr is not None and v300_arr is not None) else None,
        't2m': (t2m_arr - 273.15) if t2m_arr is not None else None
    }
    return data, lats, lons


# ==============================================================================
# 5. ZENTREN-DETEKTOR (HOCHS & TIEFS IM WXCHARTS-STIL, AUFLÖSUNGSUNABHÄNGIG)
# ==============================================================================

def find_pressure_extrema(pmsl_grid, lats, lons, domain='europe'):
    """
    Findet meteorologisch signifikante Hoch- ("H") und Tiefdruckzentren ("T").
    Skaliert Glättung & Suchradien dynamisch anhand der Gitterauflösung (dlon),
    damit sowohl ECMWF 0.25° als auch ICON-EU 0.0625° saubere Zentren liefern.
    """
    if not SCIPY_AVAILABLE or pmsl_grid is None or len(lons) < 2:
        return [], []

    try:
        dlon = max(0.05, abs(float(lons[1] - lons[0])))

        # 1. Synoptische Glättung (~0.5° Filterradius: sigma=8.0 bei 0.0625°, sigma=2.0 bei 0.25°)
        sigma_cells = max(1.0, 0.5 / dlon)
        smoothed = ndi.gaussian_filter(pmsl_grid, sigma=sigma_cells)

        # 2. Neighborhood filter für lokale Extrema (~4.0° für Europa/Atlantik, ~2.75° für Mitteleuropa)
        min_dist_deg = 4.0 if domain == 'europe' else 2.75
        min_dist = max(5, int(round(min_dist_deg / dlon)))
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
            min_candidates.append((r, c, float(smoothed[r, c])))

        max_candidates = []
        for i in range(1, num_max + 1):
            coords = np.argwhere(labeled_max == i)
            r, c = coords.mean(axis=0)
            r, c = int(round(r)), int(round(c))
            max_candidates.append((r, c, float(smoothed[r, c])))

        # 4. Prominenz-Filter (~1.75° Umgebungsring)
        nj, ni = pmsl_grid.shape
        margin = max(2, int(round(1.5625 / dlon)))
        ring_rad = max(3, int(round(1.75 / dlon)))

        valid_lows = []
        for r, c, val in min_candidates:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                ring = smoothed[max(0, r - ring_rad):min(nj, r + ring_rad + 1),
                                max(0, c - ring_rad):min(ni, c + ring_rad + 1)]
                prominence = float(np.mean(ring) - val)
                if prominence >= 0.8 and val <= 1016.0:
                    valid_lows.append((float(lons[c]), float(lats[r]), int(round(val)), val))

        valid_highs = []
        for r, c, val in max_candidates:
            if margin <= r < nj - margin and margin <= c < ni - margin:
                ring = smoothed[max(0, r - ring_rad):min(nj, r + ring_rad + 1),
                                max(0, c - ring_rad):min(ni, c + ring_rad + 1)]
                prominence = float(val - np.mean(ring))
                if prominence >= 0.8 and val >= 1018.0:
                    valid_highs.append((float(lons[c]), float(lats[r]), int(round(val)), val))

        # 5. Non-Maximum Suppression (NMS) mit Mindestabstand in Kilometern
        min_sep_km = 520.0 if domain == 'europe' else 260.0

        def distance_km(lon1, lat1, lon2, lat2):
            dlat = (lat2 - lat1) * 111.0
            dlon_km = (lon2 - lon1) * 111.0 * np.cos(np.radians((lat1 + lat2) * 0.5))
            return np.sqrt(dlat * dlat + dlon_km * dlon_km)

        valid_lows.sort(key=lambda x: x[3])
        filtered_lows = []
        for lon_c, lat_c, p_val, _ in valid_lows:
            if all(distance_km(lon_c, lat_c, fl[0], fl[1]) > min_sep_km for fl in filtered_lows):
                filtered_lows.append((lon_c, lat_c, p_val))

        valid_highs.sort(key=lambda x: -x[3])
        filtered_highs = []
        for lon_c, lat_c, p_val, _ in valid_highs:
            if all(distance_km(lon_c, lat_c, fh[0], fh[1]) > min_sep_km for fh in filtered_highs):
                filtered_highs.append((lon_c, lat_c, p_val))

        max_count = 8 if domain == 'europe' else 3
        return filtered_lows[:max_count], filtered_highs[:max_count]
    except Exception:
        return [], []


# ==============================================================================
# 6. KARTEN-RENDERING FÜR EINE DOMAIN & EINEN ZEITSCHRITT
# ==============================================================================

DOMAIN_CONFIG = {
    'europe': {
        'label': 'Nordatlantik & Europa',
        'model_label': 'ECMWF IFS (0.25°)',
        'proj_kwargs': {'central_longitude': -8.0, 'central_latitude': 51.0, 'standard_parallels': (35.0, 65.0)},
        # Exakte 16:10 Box (6400 km x 4000 km): Von US-Ostküste/Neufundland/Grönland über den gesamten Nordatlantik bis Osteuropa
        'extent': (-3200000, 3200000, -1900000, 2100000),
        'state_alpha': 0.25,
    },
    'germany': {
        'label': 'Mitteleuropa',
        'model_label': 'DWD ICON-EU (0.0625°)',
        'proj_kwargs': {'central_longitude': 10.5, 'central_latitude': 51.2, 'standard_parallels': (45.0, 55.0)},
        # Exakte 16:10 Box (1520 km x 950 km): Perfekt zentriertes Mitteleuropa inkl. Alpenraum & Benelux
        'extent': (-760000, 760000, -475000, 475000),
        'state_alpha': 0.65,
    }
}

PARAMS_ORDER = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']


def render_domain_charts(dom, run_date, lead_h, data, lats, lons, output_base):
    """
    Rendert alle 4 synoptischen Parameter für genau eine Domain ('europe' oder 'germany')
    und einen Zeitschritt (1600x1000 WebP im WXCharts-Stil).
    """
    if data is None or lats is None or lons is None:
        return []

    cfg = DOMAIN_CONFIG[dom]
    valid_date = run_date + timedelta(hours=lead_h)
    dlon = max(0.05, abs(float(lons[1] - lons[0])))

    pmsl_hpa = data.get('pmsl')
    lows, highs = find_pressure_extrema(pmsl_hpa, lats, lons, domain=dom)

    # Auflösungsabhängig sanft vorgefiltertes PMSL für seidenweiche Isobaren
    sigma_iso = max(0.8, 0.10 / dlon)
    pmsl_smooth = ndi.gaussian_filter(pmsl_hpa, sigma=sigma_iso) if (SCIPY_AVAILABLE and pmsl_hpa is not None) else pmsl_hpa

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    cur_proj = ccrs.LambertConformal(**cfg['proj_kwargs']) if CARTOPY_AVAILABLE else None
    x_min, x_max, y_min, y_max = cfg['extent']

    param_specs = {
        't850_gp': {
            'field': data.get('t850'),
            'contour': pmsl_smooth,
            'title': "850 hPa Temperatur (°C) & Bodendruck (hPa)",
            'unit': "°C",
            'cmap': WXCHARTS_TEMP_CMAP,
            'levels': WXCHARTS_TEMP_LEVELS,
            'contour_levels': np.arange(960, 1056, 4),
            'contour_fmt': "%d",
            'cbar_ticks': np.arange(-40, 44, 4),
            'is_pmsl': True,
        },
        'z500_mslp': {
            'field': data.get('fi500'),
            'contour': pmsl_smooth,
            'title': "500 hPa Geopotentialhöhe (gpdm) & Bodendruck (hPa)",
            'unit': "gpdm",
            'cmap': WXCHARTS_Z500_CMAP,
            'levels': WXCHARTS_Z500_LEVELS,
            'contour_levels': np.arange(960, 1056, 4),
            'contour_fmt': "%d",
            'cbar_ticks': np.arange(496, 596, 8),
            'is_pmsl': True,
        },
        'jet300': {
            'field': data.get('jet300'),
            'contour': data.get('fi300'),
            'title': "300 hPa Jetstream & Höhenwind (km/h) & 300 hPa Isohypsen",
            'unit': "km/h",
            'cmap': WXCHARTS_JET_CMAP,
            'levels': WXCHARTS_JET_LEVELS,
            'contour_levels': np.arange(840, 992, 8),
            'contour_fmt': "%d",
            'cbar_ticks': np.arange(40, 370, 30),
            'is_pmsl': False,
        },
        't2m_wind': {
            'field': data.get('t2m'),
            'contour': pmsl_smooth,
            'title': "2m Temperatur (°C) & Bodendruck (hPa)",
            'unit': "°C",
            'cmap': WXCHARTS_TEMP_CMAP,
            'levels': WXCHARTS_TEMP_LEVELS,
            'contour_levels': np.arange(960, 1056, 4),
            'contour_fmt': "%d",
            'cbar_ticks': np.arange(-40, 44, 4),
            'is_pmsl': True,
        },
    }

    init_str = format_german_date(run_date)
    valid_str = format_german_date(valid_date)
    generated_frames = []

    for p in PARAMS_ORDER:
        spec = param_specs[p]
        field_val = spec['field']
        if field_val is None:
            continue
        contour_val = spec['contour']

        fig = plt.figure(figsize=(16.0, 10.0), dpi=100, facecolor='#0b0f19')

        if CARTOPY_AVAILABLE:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection=cur_proj)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_facecolor('#0b1120')

            try:
                ax.add_feature(cfeature.OCEAN, facecolor='#0b1120', zorder=0)
                ax.add_feature(cfeature.LAND, facecolor='#111827', zorder=0)
            except Exception:
                pass

            cf = ax.contourf(
                lon_mesh, lat_mesh, field_val,
                levels=spec['levels'], cmap=spec['cmap'],
                extend='both', transform=ccrs.PlateCarree(),
                zorder=2
            )

            try:
                ax.add_feature(
                    cfeature.LAKES.with_scale('50m'),
                    edgecolor='#0f172a', facecolor='none', linewidth=0.8, zorder=4
                )
                states_provinces = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lines',
                    scale='50m',
                    facecolor='none'
                )
                ax.add_feature(
                    states_provinces,
                    edgecolor='#0f172a',
                    linewidth=0.7,
                    linestyle='--',
                    alpha=cfg['state_alpha'],
                    zorder=4
                )
                # Küstenlinien (Doppel-Kontur für maximale Sichtbarkeit)
                ax.add_feature(
                    cfeature.COASTLINE.with_scale('50m'),
                    edgecolor='#ffffff', linewidth=2.2, alpha=0.38, zorder=4
                )
                ax.add_feature(
                    cfeature.COASTLINE.with_scale('50m'),
                    edgecolor='#000000', linewidth=1.3, zorder=4
                )
                # Nationale Außengrenzen (Ländergrenzen)
                ax.add_feature(
                    cfeature.BORDERS.with_scale('50m'),
                    edgecolor='#ffffff', linewidth=3.0, alpha=0.50, linestyle='-', zorder=4
                )
                ax.add_feature(
                    cfeature.BORDERS.with_scale('50m'),
                    edgecolor='#000000', linewidth=1.5, linestyle='-', zorder=4
                )
            except Exception:
                ax.coastlines(resolution='110m', color='#000000', linewidth=1.4, zorder=4)

            if contour_val is not None:
                cs = ax.contour(
                    lon_mesh, lat_mesh, contour_val,
                    levels=spec['contour_levels'], colors='#ffffff',
                    linewidths=1.2, alpha=0.92, zorder=5,
                    transform=ccrs.PlateCarree()
                )
                clabels = ax.clabel(
                    cs, inline=True, fmt=spec['contour_fmt'],
                    fontsize=9, colors='#ffffff', inline_spacing=18
                )
                for lbl in clabels:
                    lbl.set_fontweight('bold')
                    lbl.set_path_effects([pe.withStroke(linewidth=2.8, foreground='#000000')])

            if spec['is_pmsl']:
                halo_stroke = [pe.withStroke(linewidth=3.5, foreground='#000000')]
                x_pad = (x_max - x_min) * 0.04
                y_pad = (y_max - y_min) * 0.06

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

            ax.gridlines(draw_labels=False, linewidth=0.4, color='#ffffff', alpha=0.15, linestyle='--')

        else:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor='#0b0f19')
            cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=spec['levels'], cmap=spec['cmap'], extend='both')
            if contour_val is not None:
                cs = ax.contour(lon_mesh, lat_mesh, contour_val, levels=spec['contour_levels'], colors='#ffffff', linewidths=1.1)
                ax.clabel(cs, inline=True, fmt=spec['contour_fmt'], fontsize=9, colors='#ffffff')

        # WXCHARTS HEADER OVERLAY (OBEN LINKS; obere rechte Ecke bleibt frei für Web-Badge)
        header_text = (
            f"localwx.de  •  {cfg['model_label']}  •  {cfg['label']}\n"
            f"{spec['title']}\n"
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

        # WXCHARTS BOTTOM COLORBAR OVERLAY (UNTEN)
        cbar_bg = fig.add_axes([0.02, 0.015, 0.96, 0.046], facecolor='#0b0f19')
        cbar_bg.patch.set_alpha(0.90)
        cbar_bg.patch.set_edgecolor('#334155')
        cbar_bg.patch.set_linewidth(0.8)
        cbar_bg.set_xticks([])
        cbar_bg.set_yticks([])

        cbar_bg.text(
            0.012, 0.5, f"({spec['unit']})",
            color='#38bdf8', fontsize=8.5, fontweight='bold',
            va='center', ha='left'
        )

        cbar_ax = fig.add_axes([0.062, 0.024, 0.900, 0.022])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', ticks=spec['cbar_ticks'])
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

    return generated_frames


def render_step_task(task_args):
    """
    Worker-Task für einen einzelnen Zeitschritt:
    1. Lädt ECMWF IFS (0.25°) für 'europe' (Nordatlantik & Europa)
    2. Lädt DWD ICON-EU (0.0625°) für 'germany' (Mitteleuropa)
    Beide Domains laufen unabhängig, damit ein temporärer Fehler bei einem
    Anbieter die andere Domain nicht blockiert.
    """
    lead_h, dwd_date_str, dwd_run_str, dwd_run_iso, dwd_base_url, ecmwf_run_iso, ecmwf_source, output_base = task_args
    dwd_run_date = datetime.fromisoformat(dwd_run_iso)
    ecmwf_run_date = datetime.fromisoformat(ecmwf_run_iso) if ecmwf_run_iso else None

    generated_frames = []

    # 1. Nordatlantik & Europa (ECMWF IFS 0.25°)
    if ecmwf_run_date is not None:
        try:
            ecmwf_data, ec_lats, ec_lons = fetch_ecmwf_step_data(lead_h, ecmwf_run_date, source=ecmwf_source)
            generated_frames.extend(
                render_domain_charts('europe', ecmwf_run_date, lead_h, ecmwf_data, ec_lats, ec_lons, output_base)
            )
        except Exception as e:
            print(f"⚠️ ECMWF +{lead_h:02d}h Fehler: {e}")

    # 2. Mitteleuropa (DWD ICON-EU 0.0625°)
    try:
        dwd_data, dwd_lats, dwd_lons = fetch_dwd_step_data(lead_h, dwd_date_str, dwd_run_str, dwd_base_url)
        generated_frames.extend(
            render_domain_charts('germany', dwd_run_date, lead_h, dwd_data, dwd_lats, dwd_lons, output_base)
        )
    except Exception as e:
        print(f"⚠️ DWD ICON-EU +{lead_h:02d}h Fehler: {e}")

    return lead_h, generated_frames


# ==============================================================================
# 7. PIPELINE GENERATOR
# ==============================================================================

def generate_synoptic_dataset():
    start_time = time.time()
    print("🚀 Starte Hybrid Synoptik- & Modellkarten Generator (ECMWF IFS + DWD ICON-EU)...")

    setup_cartopy_offline()

    output_base = "./dist/synoptic"
    os.makedirs(output_base, exist_ok=True)

    dwd_run_date = get_latest_icon_eu_run()
    dwd_date_str = dwd_run_date.strftime("%Y%m%d")
    dwd_run_str = f"{dwd_run_date.hour:02d}"
    print(f"📡 DWD ICON-EU (Mitteleuropa):    {dwd_run_date.strftime('%Y-%m-%d %H:00 UTC')}")

    ecmwf_run_date, ecmwf_source = get_latest_ecmwf_run()
    if ecmwf_run_date is not None:
        print(f"🌍 ECMWF IFS (Nordatlantik/EU):   {ecmwf_run_date.strftime('%Y-%m-%d %H:00 UTC')} (Quelle: {ecmwf_source})")
    else:
        print("⚠️ Kein ECMWF IFS Lauf ermittelt — es werden nur DWD ICON-EU Karten gerendert.")

    params = PARAMS_ORDER
    domains = ['europe', 'germany']
    steps = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 72, 84, 96, 120]
    print(f"📦 Verarbeite bis zu {len(steps)} Zeitschritte (0h bis +120h)...")

    manifest = {
        "model": "ECMWF IFS (0.25°) / DWD ICON-EU (0.0625°)",
        "init_time": (ecmwf_run_date or dwd_run_date).isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aspect_ratio": "16:10",
        "step_interval_h": 3,
        "max_lead_h": 120,
        "parameters": params,
        "domains": domains,
        "domain_meta": {
            "europe": {
                "label": DOMAIN_CONFIG['europe']['label'],
                "model": "ECMWF IFS",
                "resolution": "0.25°",
                "init_time": ecmwf_run_date.isoformat() if ecmwf_run_date else None
            },
            "germany": {
                "label": DOMAIN_CONFIG['germany']['label'],
                "model": "DWD ICON-EU",
                "resolution": "0.0625°",
                "init_time": dwd_run_date.isoformat()
            }
        },
        "steps": steps,
        "frames": {p: {dom: [] for dom in domains} for p in params}
    }

    dwd_base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{dwd_run_str}"
    ecmwf_run_iso = ecmwf_run_date.isoformat() if ecmwf_run_date else None

    task_args_list = [
        (lead_h, dwd_date_str, dwd_run_str, dwd_run_date.isoformat(), dwd_base_url, ecmwf_run_iso, ecmwf_source, output_base)
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
                print(f"   ↳ [{completed}/{len(steps)}] Zeitschritt +{lead_h:02d}h ({len(frames)} Karten) fertiggestellt.")
            except Exception as e:
                print(f"⚠️ Fehler bei Zeitschritt +{lead_h}h: {e}")

    for p in params:
        for dom in domains:
            manifest["frames"][p][dom].sort(key=lambda x: x['lead_h'])

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
