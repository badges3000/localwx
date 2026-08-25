#!/usr/bin/env python3
"""
DWD ICON-D2 Multi-Parameter Modellkarten Generator & Uploader (localwx PRO)
===========================================================================
Ultra-schneller paralleler Downloader & Vektorisierter Renderer für DWD ICON-D2
GRIB2-Vorhersagen mit automatischem FTPS-Upload zu netcup.

Unterstützte Parameter:
1. sigwx: 🌩️ Wetter-Phänomene (14 meteorologische Kategorien)
2. wind:  🌬️ 48h Spitzen-Windböen (km/h)
3. cape:  ⚡ Unwetter- & Gewitterpotenzial (J/kg)
4. rain:  🌧️ Akkumulierte 48h Regensummen (mm / l/m²)
5. snow:  ❄️ Neuschneemenge (cm)
"""

import os
import sys
import json
import time
import bz2
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False


# ==============================================================================
# PARAMETER- & FARBSKALEN-DEFINITIONEN
# ==============================================================================

# 1. Wetter-Phänomene Farbskala (14 Klassen)
SIGWX_COLOR_MAP = {
    'transparent': (0, 0, 0, 0),
    'fog': (234, 179, 8, 200),               # Bodennebel (#eab308)
    'fog_frost': (180, 83, 9, 215),           # Eisnebel / Reif (#b45309)
    'rain_light': (74, 222, 128, 210),        # Leichter Regen (#4ade80)
    'rain_medium': (22, 163, 74, 225),        # Mäßiger Regen (#16a34a)
    'rain_heavy': (6, 95, 70, 240),           # Starkregen (#065f46)
    'freezing_rain_light': (244, 63, 94, 225),# Glatteisregen (#f43f5e)
    'freezing_rain_heavy': (159, 18, 57, 240),# Schweres Glatteis (#9f1239)
    'sleet_light': (251, 146, 60, 215),       # Schneeregen (#fb923c)
    'sleet_heavy': (194, 65, 12, 230),        # Nassschnee (#c2410c)
    'snow_light': (56, 189, 248, 210),        # Leichter Schnee (#38bdf8)
    'snow_medium': (37, 99, 235, 225),        # Schneefall (#2563eb)
    'snow_heavy': (30, 27, 75, 245),          # Starker Schnee (#1e1b4b)
    'thunder_medium': (192, 132, 252, 230),   # Gewitter (#c084fc)
    'thunder_heavy': (126, 34, 206, 245),     # Unwetter / Hagel (#7e22ce)
}

PARAM_CONFIGS = {
    'sigwx': {
        'dwd_var': 'ww',
        'folder': 'sigwx',
        'title': 'Signifikantes Wetter',
        'unit': '14 Kategorien',
        'scale_type': 'sigwx'
    },
    'wind': {
        'dwd_var': 'vmax_10m',
        'folder': 'wind',
        'title': 'Spitzen-Windböen',
        'unit': 'km/h',
        'scale_type': 'wind'
    },
    'cape': {
        'dwd_var': 'cape_ml',
        'folder': 'cape',
        'title': 'Unwetter- & Gewitterpotenzial',
        'unit': 'J/kg',
        'scale_type': 'cape'
    },
    'precip_rate': {
        'dwd_var': 'tot_prec',
        'folder': 'precip_rate',
        'title': '48h Niederschlagsvorhersage (stündlich)',
        'unit': 'mm/h',
        'scale_type': 'precip_rate'
    },
    'rain': {
        'dwd_var': 'tot_prec',
        'folder': 'rain',
        'title': 'Akkumulierte Regensummen',
        'unit': 'mm',
        'scale_type': 'rain'
    },
    'snow': {
        'dwd_var': 'freshsnow_acc',
        'folder': 'snow',
        'title': 'Neuschneemenge',
        'unit': 'cm',
        'scale_type': 'snow'
    }
}

DWD_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"


def get_latest_model_run():
    """
    Ermittelt den neuesten verfügbaren ICON-D2 Modell-Lauf auf dem DWD Open Data Server.
    """
    now = datetime.now(timezone.utc)
    check_time = now - timedelta(hours=1, minutes=45)
    run_hour = (check_time.hour // 3) * 3
    run_date = check_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    
    date_str = run_date.strftime("%Y%m%d")
    hour_str = f"{run_hour:02d}"
    return date_str, hour_str, run_date


def download_dwd_file(date_str, hour_str, step, var="ww", temp_dir="./tmp_grib"):
    """
    Lädt eine einzelne DWD ICON-D2 GRIB2.bz2 Datei herunter und entpackt sie.
    """
    os.makedirs(temp_dir, exist_ok=True)
    step_str = f"{step:03d}"
    file_name = f"icon-d2_germany_regular-lat-lon_single-level_{date_str}{hour_str}_{step_str}_2d_{var}.grib2"
    bz2_file_name = f"{file_name}.bz2"
    url = f"{DWD_BASE_URL}/{hour_str}/{var}/{bz2_file_name}"
    
    bz2_path = os.path.join(temp_dir, bz2_file_name)
    grib_path = os.path.join(temp_dir, file_name)

    if os.path.exists(grib_path) and os.path.getsize(grib_path) > 1000:
        return grib_path

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'localwx-MultiModel-Generator/2.0'})
        with urllib.request.urlopen(req, timeout=25) as response, open(bz2_path, 'wb') as out_file:
            out_file.write(response.read())

        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(grib_path, 'wb') as f_out:
            f_out.write(f_in.read())

        if os.path.exists(bz2_path):
            os.remove(bz2_path)

        return grib_path
    except Exception:
        return None


# ==============================================================================
# VEKTORISIERTE FARB-TRANSFORMATIONEN (NUMPY ACCELERATED)
# ==============================================================================

def clean_grib_grid(grid_2d, max_valid=10000.0, min_valid=0.0):
    """
    Bereinigt das GRIB2-Gitter von DWD Missing-Values (z.B. 9999, 1e20), NaNs und Infs.
    Punkte außerhalb des Modell-Gebiets werden transparent (0).
    """
    arr = np.nan_to_num(grid_2d, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where((arr > max_valid) | (arr < min_valid), 0.0, arr)
    return arr


def colorize_sigwx(grid_2d):
    """Färbt die diskreten DWD Wetterzustandscodes (0-99) ein."""
    clean_val = clean_grib_grid(grid_2d, max_valid=99.0, min_valid=0.0)
    h, w = clean_val.shape

    # Erzeuge Lookup-Array für Codes 0 bis 100
    lut = np.zeros((101, 4), dtype=np.uint8)
    
    for val in range(101):
        if val in [40, 41, 42, 43, 44, 45, 46, 47, 48]:
            lut[val] = SIGWX_COLOR_MAP['fog']
        elif val == 49:
            lut[val] = SIGWX_COLOR_MAP['fog_frost']
        elif val in [50, 51, 52, 53, 58, 60, 61, 80]:
            lut[val] = SIGWX_COLOR_MAP['rain_light']
        elif val in [54, 55, 62, 63, 81]:
            lut[val] = SIGWX_COLOR_MAP['rain_medium']
        elif val in [64, 65, 82]:
            lut[val] = SIGWX_COLOR_MAP['rain_heavy']
        elif val in [56, 66]:
            lut[val] = SIGWX_COLOR_MAP['freezing_rain_light']
        elif val in [57, 67]:
            lut[val] = SIGWX_COLOR_MAP['freezing_rain_heavy']
        elif val in [68, 83]:
            lut[val] = SIGWX_COLOR_MAP['sleet_light']
        elif val in [69, 84]:
            lut[val] = SIGWX_COLOR_MAP['sleet_heavy']
        elif val in [70, 71, 77, 85]:
            lut[val] = SIGWX_COLOR_MAP['snow_light']
        elif val in [72, 73]:
            lut[val] = SIGWX_COLOR_MAP['snow_medium']
        elif val in [74, 75, 76, 86]:
            lut[val] = SIGWX_COLOR_MAP['snow_heavy']
        elif val in [87, 88, 89, 90, 95]:
            lut[val] = SIGWX_COLOR_MAP['thunder_medium']
        elif val in [96, 97, 98, 99]:
            lut[val] = SIGWX_COLOR_MAP['thunder_heavy']

    clipped = np.clip(clean_val.astype(int), 0, 100)
    return lut[clipped]


def colorize_wind(grid_2d):
    """
    Spitzenwindböen: Umrechnung von m/s in km/h (* 3.6).
    Schwellenwerte von 30 km/h bis > 120 km/h.
    """
    # Filtere unphysikalische Werte und DWD Missing-Values (9999, 1e20) heraus
    clean_ms = clean_grib_grid(grid_2d, max_valid=120.0, min_valid=0.0) # max 120 m/s = 432 km/h
    kmh = clean_ms * 3.6
    h, w = kmh.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # < 30 km/h = Transparent
    # 30-45 km/h = Sanftes Türkisgrün
    mask = (kmh >= 30) & (kmh < 45)
    rgba[mask] = [52, 211, 153, 160]

    # 45-60 km/h = Gelb (starker Wind / Bft 6-7)
    mask = (kmh >= 45) & (kmh < 60)
    rgba[mask] = [250, 204, 21, 200]

    # 60-75 km/h = Orange (stürmische Böen / Bft 8)
    mask = (kmh >= 60) & (kmh < 75)
    rgba[mask] = [251, 146, 60, 220]

    # 75-90 km/h = Kräftiges Rot (Sturmböen / Bft 9-10)
    mask = (kmh >= 75) & (kmh < 90)
    rgba[mask] = [239, 68, 68, 235]

    # 90-105 km/h = Dunkelrot (schwere Sturmböen / Bft 10-11)
    mask = (kmh >= 90) & (kmh < 105)
    rgba[mask] = [185, 28, 28, 245]

    # 105-120 km/h = Magenta / Violett (orkanartige Böen / Bft 11-12)
    mask = (kmh >= 105) & (kmh < 120)
    rgba[mask] = [192, 38, 211, 250]

    # > 120 km/h = Pink/Weiß (Orkanböen / Bft 12+) mit Obergrenze
    mask = (kmh >= 120) & (kmh <= 450)
    rgba[mask] = [244, 114, 182, 255]

    return rgba


def colorize_cape(grid_2d):
    """
    Unwetter- & Gewitterenergie (CAPE in J/kg).
    """
    cape = clean_grib_grid(grid_2d, max_valid=8000.0, min_valid=0.0)
    h, w = cape.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 100 - 350 J/kg = Zartes Gelbgrün (geringe Labilität)
    mask = (cape >= 100) & (cape < 350)
    rgba[mask] = [163, 230, 53, 160]

    # 350 - 800 J/kg = Gelb (mäßige Gewittergefahr)
    mask = (cape >= 350) & (cape < 800)
    rgba[mask] = [250, 204, 21, 200]

    # 800 - 1500 J/kg = Warmes Orange (erhöhte Unwettergefahr / Starkregen)
    mask = (cape >= 800) & (cape < 1500)
    rgba[mask] = [249, 115, 22, 225]

    # 1500 - 2500 J/kg = Kräftiges Rot (hohe Unwettergefahr / Hagel)
    mask = (cape >= 1500) & (cape < 2500)
    rgba[mask] = [220, 38, 38, 245]

    # > 2500 J/kg = Magenta / Violett (Extremes Schwergewitter-Potenzial) mit Obergrenze
    mask = (cape >= 2500) & (cape <= 8000)
    rgba[mask] = [147, 51, 234, 255]

    return rgba


def colorize_precip_rate(grid_2d):
    """
    Stündliche Niederschlagsrate in mm/h mit Google Turbo-Farbskala:
    0.1 - 0.5 mm/h: Saphirblau / Indigo
    0.5 - 2.5 mm/h: Cyan / Türkis
    2.5 - 5.0 mm/h: Frisches Lime-Grün
    5.0 - 10.0 mm/h: Vivid Gold / Gelb
    10.0 - 25.0 mm/h: Leuchtendes Orange-Rot
    > 25.0 mm/h: Magenta / Tiefviolett
    """
    precip = clean_grib_grid(grid_2d, max_valid=300.0, min_valid=0.0)
    h, w = precip.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 0.1 - 0.5 mm/h = Saphirblau / Indigo
    mask = (precip >= 0.1) & (precip < 0.5)
    rgba[mask] = [59, 130, 246, 170]

    # 0.5 - 2.5 mm/h = Cyan / Türkis
    mask = (precip >= 0.5) & (precip < 2.5)
    rgba[mask] = [6, 182, 212, 210]

    # 2.5 - 5.0 mm/h = Frisches Lime-Grün
    mask = (precip >= 2.5) & (precip < 5.0)
    rgba[mask] = [34, 197, 94, 235]

    # 5.0 - 10.0 mm/h = Vivid Gold / Gelb
    mask = (precip >= 5.0) & (precip < 10.0)
    rgba[mask] = [234, 179, 8, 250]

    # 10.0 - 25.0 mm/h = Leuchtendes Orange-Rot
    mask = (precip >= 10.0) & (precip < 25.0)
    rgba[mask] = [249, 115, 22, 255]

    # > 25.0 mm/h = Magenta / Tiefviolett
    mask = (precip >= 25.0) & (precip <= 300.0)
    rgba[mask] = [217, 70, 239, 255]

    return rgba


def colorize_rain(grid_2d):
    """
    Akkumulierte 48h Niederschlagssumme (mm / l/m²).
    """
    rain = clean_grib_grid(grid_2d, max_valid=1000.0, min_valid=0.0)
    h, w = rain.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 0.5 - 2 mm = Hauchzartes Hellblau
    mask = (rain >= 0.5) & (rain < 2.0)
    rgba[mask] = [186, 230, 253, 140]

    # 2 - 10 mm = Türkisgrün
    mask = (rain >= 2.0) & (rain < 10.0)
    rgba[mask] = [52, 211, 153, 180]

    # 10 - 20 mm = Saftiges Grün
    mask = (rain >= 10.0) & (rain < 20.0)
    rgba[mask] = [22, 163, 74, 210]

    # 20 - 35 mm = Bernstein / Gelb
    mask = (rain >= 20.0) & (rain < 35.0)
    rgba[mask] = [234, 179, 8, 230]

    # 35 - 50 mm = Orange
    mask = (rain >= 35.0) & (rain < 50.0)
    rgba[mask] = [234, 88, 12, 240]

    # 50 - 75 mm = Kräftiges Rot (Dauerregen / Überflutung)
    mask = (rain >= 50.0) & (rain < 75.0)
    rgba[mask] = [220, 38, 38, 245]

    # > 75 mm = Violett / Magenta (Extremer Starkregen) mit Obergrenze
    mask = (rain >= 75.0) & (rain <= 1000.0)
    rgba[mask] = [126, 34, 206, 255]

    return rgba


def colorize_snow(grid_2d):
    """
    Akkumulierter Neuschnee in cm.
    """
    snow = clean_grib_grid(grid_2d, max_valid=500.0, min_valid=0.0) # in kg/m² (~cm bei Dichte 100)
    h, w = snow.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 0.5 - 2 cm = Zartes Cyan
    mask = (snow >= 0.5) & (snow < 2.0)
    rgba[mask] = [125, 211, 252, 170]

    # 2 - 5 cm = Hellblau
    mask = (snow >= 2.0) & (snow < 5.0)
    rgba[mask] = [56, 189, 248, 200]

    # 5 - 10 cm = Mittelblau
    mask = (snow >= 5.0) & (snow < 10.0)
    rgba[mask] = [37, 99, 235, 225]

    # 10 - 20 cm = Kräftiges Royalblau
    mask = (snow >= 10.0) & (snow < 20.0)
    rgba[mask] = [29, 78, 216, 240]

    # 20 - 35 cm = Dunkles Indigo
    mask = (snow >= 20.0) & (snow < 35.0)
    rgba[mask] = [30, 27, 75, 250]

    # > 35 cm = Schneeviolett / Pink mit Obergrenze
    mask = (snow >= 35.0) & (snow <= 500.0)
    rgba[mask] = [236, 72, 153, 255]

    return rgba


def render_grib_to_png(grib_path, output_png_path, scale_type='sigwx', target_size=(1024, 1024)):
    """
    Liest die GRIB2-Datei mit eccodes, wendet die Vektor-Colormap an und speichert das PNG.
    """
    if not os.path.exists(grib_path):
        return None

    try:
        with open(grib_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                return None

            Ni = eccodes.codes_get(gid, 'Ni')
            Nj = eccodes.codes_get(gid, 'Nj')

            try:
                lat_first = eccodes.codes_get_double(gid, 'latitudeOfFirstGridPointInDegrees')
                lon_first = eccodes.codes_get_double(gid, 'longitudeOfFirstGridPointInDegrees')
                lat_last = eccodes.codes_get_double(gid, 'latitudeOfLastGridPointInDegrees')
                lon_last = eccodes.codes_get_double(gid, 'longitudeOfLastGridPointInDegrees')
            except Exception:
                lat_first = eccodes.codes_get(gid, 'latitudeOfFirstGridPoint') / 1e6
                lon_first = eccodes.codes_get(gid, 'longitudeOfFirstGridPoint') / 1e6
                lat_last = eccodes.codes_get(gid, 'latitudeOfLastGridPoint') / 1e6
                lon_last = eccodes.codes_get(gid, 'longitudeOfLastGridPoint') / 1e6

            # Längengrade normalisieren (z.B. 356.06° -> -3.94°)
            if lon_first > 180: lon_first -= 360
            if lon_last > 180: lon_last -= 360

            # Exakte Gültigkeits-Uhrzeit aus dem GRIB-Header lesen
            exact_valid_iso = None
            try:
                valid_date = str(eccodes.codes_get(gid, 'validityDate'))
                valid_time_int = eccodes.codes_get(gid, 'validityTime')
                valid_time_str = f"{valid_time_int:04d}"
                valid_dt = datetime.strptime(f"{valid_date}{valid_time_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                exact_valid_iso = valid_dt.isoformat()
            except Exception:
                pass

            j_scans_pos = eccodes.codes_get(gid, 'jScansPositively')
            values = eccodes.codes_get_values(gid)
            eccodes.codes_release(gid)

            grid_2d = values.reshape((Nj, Ni))

            # Vektorisierte Farb-Transformation je nach Skalentyp
            if scale_type == 'sigwx':
                rgba_array = colorize_sigwx(grid_2d)
            elif scale_type == 'precip_rate':
                rgba_array = colorize_precip_rate(grid_2d)
            elif scale_type == 'wind':
                rgba_array = colorize_wind(grid_2d)
            elif scale_type == 'cape':
                rgba_array = colorize_cape(grid_2d)
            elif scale_type == 'rain':
                rgba_array = colorize_rain(grid_2d)
            elif scale_type == 'snow':
                rgba_array = colorize_snow(grid_2d)
            else:
                rgba_array = colorize_sigwx(grid_2d)

            # Falls Scan von Süd nach Nord läuft, vertikal spiegeln
            if j_scans_pos == 1:
                rgba_array = np.flipud(rgba_array)

            img = Image.fromarray(rgba_array, mode='RGBA')
            # 1400x1400 Auflösung für gestochen scharfe Details beim Hereinzoomen
            img_resized = img.resize((1400, 1400), Image.NEAREST)
            # 100% verlustfreies WebP (Zero Artifacts) mit Alpha-Transparenz
            img_resized.save(output_png_path, 'WEBP', lossless=True, method=6)

            min_lat = round(min(lat_first, lat_last), 2)
            max_lat = round(max(lat_first, lat_last), 2)
            min_lon = round(min(lon_first, lon_last), 2)
            max_lon = round(max(lon_first, lon_last), 2)

            # Plausibilitätsprüfung für DWD ICON-D2 Regular Grid
            if not (40 <= min_lat <= 46 and 55 <= max_lat <= 60 and -6 <= min_lon <= 0 and 17 <= max_lon <= 24):
                min_lat, max_lat = 43.18, 58.08
                min_lon, max_lon = -3.94, 20.34

            return [[min_lat, min_lon], [max_lat, max_lon]], exact_valid_iso

    except Exception as e:
        print(f"Fehler beim Rendern von {grib_path}: {e}")
        return [[43.18, -3.94], [58.08, 20.34]], None


def render_hourly_precip_diff(grib_curr_path, grib_prev_path, output_webp_path):
    """
    Berechnet die echte stündliche Niederschlagsdifferenz (tot_prec[t] - tot_prec[t-1]) in mm/h.
    """
    if not os.path.exists(grib_curr_path) or not os.path.exists(grib_prev_path):
        return None
    try:
        with open(grib_curr_path, 'rb') as fc, open(grib_prev_path, 'rb') as fp:
            gid_c = eccodes.codes_grib_new_from_file(fc)
            gid_p = eccodes.codes_grib_new_from_file(fp)
            if not gid_c or not gid_p:
                return None

            Ni = eccodes.codes_get(gid_c, 'Ni')
            Nj = eccodes.codes_get(gid_c, 'Nj')
            lat_first = eccodes.codes_get(gid_c, 'latitudeOfFirstGridPoint') / 1e6
            lon_first = eccodes.codes_get(gid_c, 'longitudeOfFirstGridPoint') / 1e6
            lat_last = eccodes.codes_get(gid_c, 'latitudeOfLastGridPoint') / 1e6
            lon_last = eccodes.codes_get(gid_c, 'longitudeOfLastGridPoint') / 1e6
            j_scans_pos = eccodes.codes_get(gid_c, 'jScansPositively')

            if lon_first > 180: lon_first -= 360
            if lon_last > 180: lon_last -= 360

            exact_valid_iso = None
            try:
                valid_date = str(eccodes.codes_get(gid_c, 'validityDate'))
                valid_time_int = eccodes.codes_get(gid_c, 'validityTime')
                valid_time_str = f"{valid_time_int:04d}"
                valid_dt = datetime.strptime(f"{valid_date}{valid_time_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                exact_valid_iso = valid_dt.isoformat()
            except Exception:
                pass

            vals_curr = eccodes.codes_get_values(gid_c).reshape((Nj, Ni))
            vals_prev = eccodes.codes_get_values(gid_p).reshape((Nj, Ni))
            eccodes.codes_release(gid_c)
            eccodes.codes_release(gid_p)

            # Echte stündliche Niederschlagsrate:
            hourly_rain = np.maximum(0.0, vals_curr - vals_prev)
            rgba_array = colorize_precip_rate(hourly_rain)

            if j_scans_pos == 1:
                rgba_array = np.flipud(rgba_array)

            img = Image.fromarray(rgba_array, mode='RGBA')
            img_resized = img.resize((1400, 1400), Image.NEAREST)
            img_resized.save(output_webp_path, 'WEBP', lossless=True, method=6)

            min_lat = round(min(lat_first, lat_last), 2)
            max_lat = round(max(lat_first, lat_last), 2)
            min_lon = round(min(lon_first, lon_last), 2)
            max_lon = round(max(lon_first, lon_last), 2)

            if not (40 <= min_lat <= 46 and 55 <= max_lat <= 60 and -6 <= min_lon <= 0 and 17 <= max_lon <= 24):
                min_lat, max_lat = 43.18, 58.08
                min_lon, max_lon = -3.94, 20.34

            return [[min_lat, min_lon], [max_lat, max_lon]], exact_valid_iso
    except Exception as e:
        print(f"Fehler bei Niederschlagsdifferenz: {e}")
        return None


def process_single_step(step, date_str, hour_str, run_date, param_key, output_dir, temp_dir):
    """
    Verarbeitet einen einzelnen Vorhersageschritt für einen bestimmten Parameter (WebP).
    """
    config = PARAM_CONFIGS[param_key]
    dwd_var = config['dwd_var']
    scale_type = config['scale_type']

    step_time = run_date + timedelta(hours=step)
    webp_name = f"frame_{step:02d}.webp"
    if param_key == 'sigwx':
        webp_name = f"sigwx_{step:02d}.webp"

    output_path = os.path.join(output_dir, webp_name)

    # Spezialfall: Stündliche Regenrate (Differenz zur Vorstunde)
    if param_key == 'precip_rate':
        if step == 0:
            # Stunde 0: Transparent
            empty = Image.new('RGBA', (1400, 1400), (0, 0, 0, 0))
            empty.save(output_path, 'WEBP', lossless=True)
            return step, webp_name, [[43.18, -3.94], [58.08, 20.34]], step_time.isoformat()
        else:
            grib_curr = download_dwd_file(date_str, hour_str, step, var='tot_prec', temp_dir=temp_dir)
            grib_prev = download_dwd_file(date_str, hour_str, step - 1, var='tot_prec', temp_dir=temp_dir)
            if not grib_curr or not grib_prev:
                return step, None, None, None

            res = render_hourly_precip_diff(grib_curr, grib_prev, output_path)
            for f in [grib_curr, grib_prev]:
                if f and os.path.exists(f):
                    try: os.remove(f)
                    except Exception: pass

            if res:
                bounds, exact_iso = res
                return step, webp_name, bounds, exact_iso or step_time.isoformat()
            return step, None, None, None

    grib_file = download_dwd_file(date_str, hour_str, step, var=dwd_var, temp_dir=temp_dir)
    if not grib_file:
        return step, None, None, None

    res = render_grib_to_png(grib_file, output_path, scale_type=scale_type)
    if os.path.exists(grib_file):
        try:
            os.remove(grib_file)
        except Exception:
            pass

    if res:
        bounds, exact_iso = res
        valid_iso = exact_iso or step_time.isoformat()
        return step, webp_name, bounds, valid_iso
    return step, None, None, None


def upload_directory_to_ftp(local_dir, remote_folder):
    """
    Lädt alle generierten Dateien (.webp, .png, .json) per FTPS in das Verzeichnis data/{remote_folder}/
    sowie in das Netcup Domain-Wurzelverzeichnis localwx/data/{remote_folder}/ hoch.
    """
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password:
        print(f"ℹ️ Keine FTP-Zugangsdaten. Überspringe Upload für {remote_folder}.")
        return

    print(f"\n📡 Verbinde mit FTP-Server für '{remote_folder}'...")

    ftp = None
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

    # Prüfe verfügbare Verzeichnisse im FTP-Root
    try:
        ftp.cwd('/')
        root_items = ftp.nlst()
    except Exception:
        root_items = []

    # Bestimme Zielpfade: Wenn 'localwx' im FTP-Root existiert (Netcup Domainordner),
    # lade prioritär nach localwx/data/{remote_folder} und parallel nach data/{remote_folder}
    target_paths = []
    if 'localwx' in root_items or any('localwx' in x for x in root_items):
        target_paths.append(['localwx', 'data', remote_folder])
    target_paths.append(['data', remote_folder])

    files = [f for f in os.listdir(local_dir) if f.endswith('.webp') or f.endswith('.png') or f.endswith('.json')]
    print(f"📤 Lade {len(files)} Dateien für '{remote_folder}' hoch...")

    for path_segments in target_paths:
        try:
            ftp.cwd('/')
            for folder in path_segments:
                try:
                    ftp.cwd(folder)
                except ftplib.error_perm:
                    try:
                        ftp.mkd(folder)
                        ftp.cwd(folder)
                    except Exception as e:
                        print(f"Hinweis beim Ordnererstellen ({folder}): {e}")
            
            target_str = "/".join(path_segments)
            uploaded_count = 0
            for filename in sorted(files):
                local_path = os.path.join(local_dir, filename)
                with open(local_path, 'rb') as f_in:
                    ftp.storbinary(f"STOR {filename}", f_in)
                    uploaded_count += 1

            print(f"✅ {uploaded_count} Dateien erfolgreich nach /{target_str}/ hochgeladen!")
        except Exception as e:
            print(f"⚠️ Fehler beim Upload nach /{'/'.join(path_segments)}: {e}")

    try:
        ftp.quit()
    except Exception:
        pass


def process_parameter(param_key, date_str, hour_str, run_date, max_steps=48):
    """
    Verarbeitet alle 48 Zeitschritte eines Parameters parallel.
    """
    config = PARAM_CONFIGS[param_key]
    folder = config['folder']
    title = config['title']

    print(f"\n=======================================================")
    print(f"🔄 Starte Generierung für: {title} ({param_key})")
    print(f"=======================================================")

    output_dir = f"./dist/{folder}"
    temp_dir = f"./tmp_{param_key}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    metadata = {
        "model": "DWD ICON-D2",
        "parameter": param_key,
        "title": title,
        "unit": config['unit'],
        "model_run": f"{date_str}{hour_str}z",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": [[43.18, -3.94], [58.08, 20.34]],
        "frames": []
    }

    detected_bounds = None
    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_single_step, step, date_str, hour_str, run_date, param_key, output_dir, temp_dir): step
            for step in range(max_steps + 1)
        }

        completed = 0
        for future in as_completed(futures):
            step = futures[future]
            try:
                step_idx, png_name, bounds, valid_iso = future.result()
                if png_name and valid_iso:
                    results[step_idx] = {
                        "step": step_idx,
                        "valid_time": valid_iso,
                        "file": png_name
                    }
                    if bounds and detected_bounds is None:
                        detected_bounds = bounds
                        metadata["bounds"] = detected_bounds
                completed += 1
                if completed % 12 == 0 or completed == (max_steps + 1):
                    print(f"   ↳ [{param_key}] {completed}/{max_steps+1} Stunden fertig...")
            except Exception as e:
                print(f"⚠️ Fehler bei [{param_key}] Schritt {step}: {e}")

    # Frames sortieren & meta.json schreiben
    metadata["frames"] = [results[s] for s in sorted(results.keys())]

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✨ Parameter '{param_key}' abgeschlossen ({len(metadata['frames'])} Frames).")

    # Upload
    upload_directory_to_ftp(output_dir, folder)


def main():
    parser = argparse.ArgumentParser(description="DWD ICON-D2 Multi-Model Map Generator")
    parser.add_argument(
        '--param',
        choices=['all', 'precip_rate', 'sigwx', 'wind', 'cape', 'rain', 'snow'],
        default='all',
        help="Welcher Parameter generiert werden soll (Standard: all)"
    )
    args = parser.parse_args()

    start_time = time.time()
    print("🚀 Starte DWD ICON-D2 Multi-Parameter Generator (localwx PRO)...")

    date_str, hour_str, run_date = get_latest_model_run()
    print(f"📅 DWD Modell-Lauf: {date_str} {hour_str}:00 UTC")

    if args.param == 'all':
        active_params = ['precip_rate', 'sigwx', 'wind', 'cape', 'rain', 'snow']
    else:
        active_params = [args.param]

    for p in active_params:
        process_parameter(p, date_str, hour_str, run_date)

    duration = round(time.time() - start_time, 1)
    print(f"\n🎉 GESAMT-ERFOLG: Alle {len(active_params)} Modell-Karten in {duration} Sekunden verarbeitet!")


if __name__ == "__main__":
    main()
