#!/usr/bin/env python3
"""
DWD ICON-D2 Signifikantes Wetter (SigWx) Generator
==================================================
Lädt die echten DWD ICON-D2 GRIB2-Vorhersagedateien von opendata.dwd.de herunter,
klassifiziert jedes Gitterfeld nach den 14 Kachelmann/DWD-Wettererscheinungen
(Regen, Schnee, Gewitter, Nebel, etc.) und speichert transparente PNG-Overlays
sowie eine metadata.json für die Web-App.
"""

import os
import sys
import json
import time
import bz2
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image

# Versuche eccodes/cfgrib zu importieren
try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


# DWD Farbskala (Kachelmann / DWD Signifikantes Wetter Matrix)
# RGBA Farbcodes
COLOR_MAP = {
    'transparent': (0, 0, 0, 0),
    'fog': (234, 179, 8, 200),               # Nebel (Gelb)
    'fog_frost': (202, 138, 4, 215),          # Nebel Reifbildung (Dunkelgelb)
    'rain_light': (85, 240, 85, 210),         # Regen leicht (Hellgrün)
    'rain_medium': (22, 163, 74, 225),        # Regen mäßig (Grün)
    'rain_heavy': (21, 128, 61, 235),         # Regen stark (Dunkelgrün)
    'freezing_rain_light': (239, 68, 68, 225),# gefr. Regen leicht (Rot)
    'freezing_rain_heavy': (153, 27, 27, 240),# gefr. Regen stark (Dunkelrot)
    'sleet_light': (245, 158, 11, 215),       # Schneeregen leicht (Hellorange)
    'sleet_heavy': (234, 88, 12, 230),        # Schneeregen mäßig/stark (Orange)
    'snow_light': (125, 211, 252, 210),       # Schneefall leicht (Hellblau)
    'snow_medium': (2, 132, 199, 225),        # Schneefall mäßig (Mittelblau)
    'snow_heavy': (30, 58, 138, 240),         # Schneefall stark (Dunkelblau)
    'thunder_medium': (217, 70, 239, 230),    # Gewitter leicht/mäßig (Pink/Lila)
    'thunder_heavy': (162, 28, 175, 245),     # Gewitter stark (Dunkellila)
}

DWD_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"

def get_latest_model_run():
    """
    Ermittelt den neuesten verfügbaren ICON-D2 Modell-Lauf auf dem DWD Open Data Server.
    Hauptläufe: 00, 03, 06, 09, 12, 15, 18, 21 UTC (bis 48h Vorhersage).
    """
    now = datetime.now(timezone.utc)
    # ICON-D2 benötigt ca. 1h45m bis zur Bereitstellung
    check_time = now - timedelta(hours=1, minutes=45)
    
    # Runde auf den letzten 3h-Hauptlauf ab
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

    # Wenn die entpackte Datei schon existiert, überspringen
    if os.path.exists(grib_path) and os.path.getsize(grib_path) > 1000:
        return grib_path

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'localwx-SigWx-Generator/2.0'})
        with urllib.request.urlopen(req, timeout=20) as response, open(bz2_path, 'wb') as out_file:
            out_file.write(response.read())

        # Entpacken
        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(grib_path, 'wb') as f_out:
            f_out.write(f_in.read())

        # Bz2 Datei löschen um Speicher zu sparen
        if os.path.exists(bz2_path):
            os.remove(bz2_path)

        return grib_path
    except Exception as e:
        # Fallback falls Hauptlauf noch nicht vollständig hochgeladen ist
        return None


def classify_ww_code(ww_val):
    """
    Wandelt den WMO / DWD Wetterzustandscode (0-99) in einen Farb-Schlüssel um.
    """
    val = int(round(ww_val)) if not np.isnan(ww_val) else 0

    if val == 0 or val == 1 or val == 2 or val == 3:
        return 'transparent'
    elif val == 45 or val == 48:
        return 'fog'
    elif val == 49:
        return 'fog_frost'
    elif val in [50, 51, 52, 53, 58, 60, 61, 80]:
        return 'rain_light'
    elif val in [54, 55, 62, 63, 81]:
        return 'rain_medium'
    elif val in [64, 65, 82]:
        return 'rain_heavy'
    elif val in [56, 66]:
        return 'freezing_rain_light'
    elif val in [57, 67]:
        return 'freezing_rain_heavy'
    elif val in [68, 83]:
        return 'sleet_light'
    elif val in [69, 84]:
        return 'sleet_heavy'
    elif val in [70, 71, 77, 85]:
        return 'snow_light'
    elif val in [72, 73]:
        return 'snow_medium'
    elif val in [74, 75, 76, 86]:
        return 'snow_heavy'
    elif val in [87, 88, 89, 90, 95]:
        return 'thunder_medium'
    elif val in [96, 97, 98, 99]:
        return 'thunder_heavy'
    else:
        return 'transparent'


def render_grib_to_png(grib_path, output_png_path, target_size=(640, 640)):
    """
    Liest die GRIB2-Datei mit eccodes/cfgrib und erzeugt ein optimiertes transparentes PNG.
    """
    if not os.path.exists(grib_path):
        return False

    try:
        # 1. Methode: Direkt über eccodes
        with open(grib_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                return False

            Ni = eccodes.codes_get(gid, 'Ni')
            Nj = eccodes.codes_get(gid, 'Nj')
            values = eccodes.codes_get_values(gid)
            eccodes.codes_release(gid)

            # Umformen in 2D Matrix (Nj, Ni)
            grid_2d = values.reshape((Nj, Ni))
            
            # Koordinaten-Grenzen prüfen
            # DWD ICON-D2 Regular Grid ist typischerweise von Nord nach Süd orientiert
            height, width = grid_2d.shape

            # Erstelle RGBA Bild
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            pixels = img.load()

            for y in range(height):
                for x in range(width):
                    val = grid_2d[y, x]
                    color_key = classify_ww_code(val)
                    if color_key != 'transparent':
                        pixels[x, y] = COLOR_MAP[color_key]

            # Auf Zielgröße skalieren (Nearest Neighbor für scharfe Wetterkanten)
            img_resized = img.resize(target_size, Image.NEAREST)
            img_resized.save(output_png_path, 'PNG', optimize=True)
            return True

    except Exception as e:
        print(f"Fehler beim Rendern von {grib_path}: {e}")
        return False


def main():
    print("🚀 Starte DWD ICON-D2 Signifikantes Wetter Generator...")
    
    date_str, hour_str, run_date = get_latest_model_run()
    print(f"📅 Aktueller DWD Modell-Lauf: {date_str} {hour_str}:00 UTC")

    output_dir = "./dist/sigwx"
    temp_dir = "./tmp_grib"
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "model": "DWD ICON-D2",
        "model_run": f"{date_str}{hour_str}z",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": [[43.5, 2.0], [57.0, 18.0]],  # Geografische Bounding-Box für Leaflet
        "frames": []
    }

    success_count = 0
    max_steps = 48  # 48 Stunden Vorhersage

    for step in range(max_steps + 1):
        step_time = run_date + timedelta(hours=step)
        png_name = f"sigwx_{step:02d}.png"
        png_path = os.path.join(output_dir, png_name)

        print(f"⏳ Lade und rendere Stunde +{step:02d}h ({step_time.strftime('%d.%m. %H:%M UTC')})...")
        grib_file = download_dwd_file(date_str, hour_str, step, var="ww", temp_dir=temp_dir)

        if grib_file:
            if render_grib_to_png(grib_file, png_path):
                metadata["frames"].append({
                    "step": step,
                    "valid_time": step_time.isoformat(),
                    "file": png_name
                })
                success_count += 1
                # GRIB-Datei nach Verarbeitung löschen um Speicher zu schonen
                if os.path.exists(grib_file):
                    os.remove(grib_file)
            else:
                print(f"⚠️ Rendern für Schritt {step} fehlgeschlagen.")
        else:
            print(f"⚠️ GRIB-Datei für Schritt {step} nicht verfügbar.")

    # Schreibe metadata.json
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Fertig! {success_count} von {max_steps+1} Frames erfolgreich gerendert.")
    print(f"📁 Ausgabe: {output_dir}/ (meta.json und sigwx_*.png)")


if __name__ == "__main__":
    main()
