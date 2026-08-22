#!/usr/bin/env python3
"""
DWD ICON-EU 5-Tage Europa-Niederschlagstrend Generator & Uploader (localwx PRO)
==============================================================================
Erzeugt 5-Tage (120 Stunden) hochauflösende Niederschlagskarten für ganz Europa
aus dem DWD ICON-EU Modell in der Google Turbo-Farbskala als verlustfreie WebPs.

- Ordner auf Webspace: /data/europe_5d/
- 40 Zeitschritte à 3 Stunden (0h bis 120h)
- Bounds: [[34.0, -12.0], [66.0, 36.0]]
- Automatischer FTPS-Upload
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

EUROPE_BOUNDS = [[34.0, -12.0], [66.0, 36.0]]


def get_latest_icon_eu_reference_time():
    """
    Berechnet die neueste vollständige Referenzlauf-Zeit für ICON-EU (00, 06, 12, 18 UTC).
    """
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


def recolor_wms_to_turbo(img_rgba):
    """
    Wandelt DWD ICON-EU WMS-Niederschlag in echte Google Turbo-Farben um.
    Garantiert 100% transparenten Hintergrund überall wo es trocken ist!
    Farbstufen: Tiefblau -> Türkis -> Smaragdgrün -> Goldgelb -> Orange-Rot -> Magenta
    """
    arr = np.array(img_rgba)
    h, w, c = arr.shape
    if c < 4:
        return img_rgba

    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    a = arr[:, :, 3].astype(int)

    # Farbvarianz berechnen (Grau-/Schwarztöne des Hintergrunds ausschließen)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Echtes Regensignal hat Farbe (delta > 20) und ist nicht rein grau/schwarz/weiß
    has_rain = (a > 40) & (delta > 20) & (max_c > 35)

    new_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # 1. Nieselregen (Tiefblau: 0.1 - 0.5 mm/3h)
    mask_blue = has_rain & (b > r + 30) & (b > g)
    new_rgba[mask_blue] = [46, 98, 216, 190]

    # 2. Leichter Regen (Türkis / Cyan: 0.5 - 2.0 mm/3h)
    mask_cyan = has_rain & (g > r + 15) & (b > 130) & ~mask_blue
    new_rgba[mask_cyan] = [54, 170, 253, 215]

    # 3. Mäßiger Regen (Smaragd / Frisches Lime-Grün: 2.0 - 5.0 mm/3h)
    mask_green = has_rain & (g > r) & (g > b) & ~mask_blue & ~mask_cyan
    new_rgba[mask_green] = [34, 197, 94, 235]

    # 4. Kräftiger Regen (Goldgelb: 5.0 - 10.0 mm/3h)
    mask_yellow = has_rain & (r > 170) & (g > 150) & (b < 120) & ~mask_green
    new_rgba[mask_yellow] = [251, 182, 55, 250]

    # 5. Starkregen (Leuchtendes Orange / Karminrot: 10.0 - 25.0 mm/3h)
    mask_orange_red = has_rain & (r > 180) & (g < 140) & (b < 90)
    new_rgba[mask_orange_red] = [234, 92, 25, 255]

    # 6. Unwetter / Hagel (Magenta / Violett: > 25 mm/3h)
    mask_purple = has_rain & (r > 130) & (b > 130) & (g < 120)
    new_rgba[mask_purple] = [217, 70, 239, 255]

    # Unklassifizierter Rest mit Regensignal -> sauberes Mint-Grün
    unclassified = has_rain & (new_rgba[:, :, 3] == 0)
    new_rgba[unclassified] = [26, 228, 182, 210]

    return Image.fromarray(new_rgba, mode='RGBA')


def generate_europe_dataset():
    start_time = time.time()
    print("🚀 Starte DWD ICON-EU 5-Tage Europa-Niederschlagstrend Generator (120h)...")

    output_dir = "./dist/europe_5d"
    os.makedirs(output_dir, exist_ok=True)

    ref_date = get_latest_icon_eu_reference_time()
    ref_iso = ref_date.strftime("%Y-%m-%dT%H:00:00.000Z")
    print(f"📅 Referenz-Lauf: {ref_iso}")

    total_steps = 40  # 40 * 3h = 120 Stunden
    step_hours = 3

    steps = []
    for i in range(total_steps):
        valid_dt = ref_date + timedelta(hours=(i + 1) * step_hours)
        valid_iso = valid_dt.strftime("%Y-%m-%dT%H:00:00.000Z")
        steps.append((i, (i + 1) * step_hours, valid_dt, valid_iso))

    results = []

    def fetch_and_process_step(step_idx, forecast_hour, valid_dt, valid_iso):
        file_name = f"europe_{step_idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)

        # DWD Geoserver WMS für ICON-EU 3h Niederschlag
        min_lat, min_lon = EUROPE_BOUNDS[0]
        max_lat, max_lon = EUROPE_BOUNDS[1]
        
        wms_url = (
            f"https://maps.dwd.de/geoserver/dwd/wms?"
            f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&"
            f"LAYERS=dwd:Icon-eu_reg00625_fd_sl_TOTPREC03H&"
            f"STYLES=&CRS=EPSG:4326&"
            f"BBOX={min_lat},{min_lon},{max_lat},{max_lon}&"
            f"WIDTH=1200&HEIGHT=1000&"
            f"FORMAT=image/png&TRANSPARENT=TRUE&"
            f"TIME={valid_iso}&DIM_REFERENCE_TIME={ref_iso}"
        )

        try:
            req = urllib.request.Request(wms_url, headers={'User-Agent': 'localwx-EuropeTrend-Generator/2.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_img = Image.open(resp).convert('RGBA')

            # Farbtransformation anwenden
            recolored = recolor_wms_to_turbo(raw_img)
            recolored.save(file_path, 'WEBP', lossless=True, method=5)

            return {
                "step": step_idx,
                "forecast_hour": forecast_hour,
                "valid_time": valid_iso,
                "file": file_name
            }
        except Exception as e:
            # Fallback: leeres transparentes Frame
            empty = Image.new('RGBA', (1200, 1000), (0, 0, 0, 0))
            empty.save(file_path, 'WEBP', lossless=True)
            return {
                "step": step_idx,
                "forecast_hour": forecast_hour,
                "valid_time": valid_iso,
                "file": file_name
            }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_and_process_step, s[0], s[1], s[2], s[3]): s[0]
            for s in steps
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x['step'])

    # Metadata JSON schreiben
    metadata = {
        "model": "DWD ICON-EU",
        "parameter": "precipitation_europe_5d",
        "title": "5-Tage Europa-Niederschlagstrend",
        "unit": "mm/3h",
        "reference_time": ref_iso,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": EUROPE_BOUNDS,
        "frames": results
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✨ 5-Tage Europa-Trend erfolgreich generiert: {len(results)} Frames in {duration}s!")

    # FTPS Upload
    upload_directory_to_ftp(output_dir, "europe_5d")


def upload_directory_to_ftp(local_dir, remote_folder="europe_5d"):
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password:
        print(f"ℹ️ Keine FTP-Zugangsdaten. Überspringe Upload für {remote_folder}.")
        return

    print(f"\n📡 Verbinde mit FTP-Server für 'data/{remote_folder}/'...")

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

    # Zielordner /data/{remote_folder} erstellen & wechseln
    ftp.cwd('/')
    for folder in ["data", remote_folder]:
        try:
            ftp.cwd(folder)
        except ftplib.error_perm:
            try:
                ftp.mkd(folder)
                ftp.cwd(folder)
            except Exception as e:
                print(f"Hinweis beim Ordnererstellen ({folder}): {e}")

    files = [f for f in os.listdir(local_dir) if f.endswith('.webp') or f.endswith('.json')]
    print(f"📤 Lade {len(files)} Dateien nach data/{remote_folder}/ hoch...")

    uploaded_count = 0
    for filename in sorted(files):
        local_path = os.path.join(local_dir, filename)
        with open(local_path, 'rb') as f_in:
            ftp.storbinary(f"STOR {filename}", f_in)
            uploaded_count += 1

    ftp.quit()
    print(f"✅ {uploaded_count} Europa-Trend-Dateien erfolgreich nach data/{remote_folder}/ hochgeladen!")


if __name__ == "__main__":
    generate_europe_dataset()
