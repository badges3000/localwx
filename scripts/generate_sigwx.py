#!/usr/bin/env python3
"""
DWD ICON-D2 Wetter-Phänomene Generator & Uploader (localwx PRO)
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
import ftplib
import ssl

try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False

# localwx PRO Farbskala (Modernes Farbsystem)
COLOR_MAP = {
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

DWD_BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"

def get_latest_model_run():
    now = datetime.now(timezone.utc)
    check_time = now - timedelta(hours=1, minutes=45)
    run_hour = (check_time.hour // 3) * 3
    run_date = check_time.replace(hour=run_hour, minute=0, second=0, microsecond=0)
    return run_date.strftime("%Y%m%d"), f"{run_hour:02d}", run_date

def download_dwd_file(date_str, hour_str, step, var="ww", temp_dir="./tmp_grib"):
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
        req = urllib.request.Request(url, headers={'User-Agent': 'localwx-SigWx-Generator/2.0'})
        with urllib.request.urlopen(req, timeout=20) as response, open(bz2_path, 'wb') as out_file:
            out_file.write(response.read())

        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(grib_path, 'wb') as f_out:
            f_out.write(f_in.read())

        if os.path.exists(bz2_path):
            os.remove(bz2_path)
        return grib_path
    except Exception:
        return None

def classify_ww_code(ww_val):
    val = int(round(ww_val)) if not np.isnan(ww_val) else 0
    if val in [0, 1, 2, 3]: return 'transparent'
    elif val in [40, 41, 42, 43, 44, 45, 46, 47, 48]: return 'fog'
    elif val == 49: return 'fog_frost'
    elif val in [50, 51, 52, 53, 58, 60, 61, 80]: return 'rain_light'
    elif val in [54, 55, 62, 63, 81]: return 'rain_medium'
    elif val in [64, 65, 82]: return 'rain_heavy'
    elif val in [56, 66]: return 'freezing_rain_light'
    elif val in [57, 67]: return 'freezing_rain_heavy'
    elif val in [68, 83]: return 'sleet_light'
    elif val in [69, 84]: return 'sleet_heavy'
    elif val in [70, 71, 77, 85]: return 'snow_light'
    elif val in [72, 73]: return 'snow_medium'
    elif val in [74, 75, 76, 86]: return 'snow_heavy'
    elif val in [87, 88, 89, 90, 95]: return 'thunder_medium'
    elif val in [96, 97, 98, 99]: return 'thunder_heavy'
    return 'transparent'

def render_grib_to_png(grib_path, output_png_path, target_size=(1024, 1024)):
    if not os.path.exists(grib_path): return None
    try:
        with open(grib_path, 'rb') as f:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None: return None
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

            j_scans_pos = eccodes.codes_get(gid, 'jScansPositively')
            values = eccodes.codes_get_values(gid)
            eccodes.codes_release(gid)

            grid_2d = values.reshape((Nj, Ni))
            height, width = grid_2d.shape

            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            pixels = img.load()

            for y in range(height):
                for x in range(width):
                    val = grid_2d[y, x]
                    color_key = classify_ww_code(val)
                    if color_key != 'transparent':
                        pixels[x, y] = COLOR_MAP[color_key]

            if j_scans_pos == 1:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

            img_resized = img.resize(target_size, Image.NEAREST)
            img_resized.save(output_png_path, 'PNG', optimize=True)

            min_lat = min(lat_first, lat_last)
            max_lat = max(lat_first, lat_last)
            min_lon = min(lon_first, lon_last)
            max_lon = max(lon_first, lon_last)

            if not (35 <= min_lat <= 60 and 40 <= max_lat <= 65 and -5 <= min_lon <= 25 and 0 <= max_lon <= 30):
                min_lat, max_lat = 43.2, 55.85
                min_lon, max_lon = 1.8, 16.2

            return [[min_lat, min_lon], [max_lat, max_lon]]
    except Exception as e:
        print(f"Fehler beim Rendern: {e}")
        return [[43.2, 1.8], [55.85, 16.2]]

def upload_files_to_ftp(output_dir):
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password: return

    print(f"\n📡 Verbinde mit FTP-Server: {server} als {user}...")
    ftp = None
    try:
        ftp = ftplib.FTP_TLS()
        ftp.connect(server, 21, timeout=30)
        ftp.login(user, password)
        ftp.prot_p()
    except Exception:
        ftp = ftplib.FTP()
        ftp.connect(server, 21, timeout=30)
        ftp.login(user, password)

    ftp.cwd('/')
    for folder in ["data", "sigwx"]:
        try:
            ftp.cwd(folder)
        except ftplib.error_perm:
            try:
                ftp.mkd(folder)
                ftp.cwd(folder)
            except Exception: pass

    files = [f for f in os.listdir(output_dir) if f.endswith('.png') or f.endswith('.json')]
    print(f"\n📤 Starte Upload von {len(files)} Dateien...")
    uploaded_count = 0
    for filename in sorted(files):
        local_path = os.path.join(output_dir, filename)
        with open(local_path, 'rb') as f_in:
            ftp.storbinary(f"STOR {filename}", f_in)
            uploaded_count += 1
            if uploaded_count % 10 == 0 or uploaded_count == len(files):
                print(f"   ↳ {uploaded_count}/{len(files)} Dateien hochgeladen ({filename})...")

    ftp.quit()
    print(f"\n🎉 ERFOLG: Alle {uploaded_count} DWD SigWx Wetterkarten liegen jetzt in data/sigwx/!")

def main():
    print("🚀 Starte DWD ICON-D2 Generator & Uploader...")
    date_str, hour_str, run_date = get_latest_model_run()
    output_dir = "./dist/sigwx"
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "model": "DWD ICON-D2",
        "model_run": f"{date_str}{hour_str}z",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": [[43.2, 1.8], [55.85, 16.2]],
        "frames": []
    }

    success_count = 0
    detected_bounds = None

    for step in range(49):
        step_time = run_date + timedelta(hours=step)
        png_name = f"sigwx_{step:02d}.png"
        png_path = os.path.join(output_dir, png_name)
        grib_file = download_dwd_file(date_str, hour_str, step, var="ww")
        if grib_file:
            bounds = render_grib_to_png(grib_file, png_path)
            if bounds:
                if detected_bounds is None:
                    detected_bounds = bounds
                    metadata["bounds"] = detected_bounds
                metadata["frames"].append({"step": step, "valid_time": step_time.isoformat(), "file": png_name})
                success_count += 1
                if os.path.exists(grib_file): os.remove(grib_file)

    with open(os.path.join(output_dir, "meta.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ {success_count} Frames gerendert.")
    upload_files_to_ftp(output_dir)

if __name__ == "__main__":
    main()
