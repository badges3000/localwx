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


# DWD GeoServer WMS Palette -> RADOLAN Turbo Palette Mapping
# Mappt jede DWD-Farbstufe exakt auf das Design von generate_radar.py:
# Niesel (Türkis) -> Leicht/Mäßig (Grün) -> Mäßig/Stark (Gelb) -> Stark (Orange/Rot) -> Unwetter (Magenta/Weiß)
DWD_WMS_COLOR_MAP = [
    # (r, g, b) DWD WMS               -> [r, g, b, a] Turbo Radar
    ((192, 192, 192), [35, 190, 240, 175]),   # Trace Niesel (0.01 - 0.1 mm) -> Feiner Türkis
    ((254, 254, 192), [40, 200, 240, 210]),   # 0.1 - 0.2 mm -> Zarter Türkis
    ((254, 254, 91),  [45, 215, 210, 240]),   # 0.2 - 0.5 mm -> Mint-Türkis
    ((207, 242, 0),   [60, 225, 100, 255]),   # 0.5 - 1.0 mm -> Helles Lime-Grün
    ((160, 214, 37),  [45, 218, 70, 255]),    # 1.0 - 2.0 mm -> Frisches Grasgrün
    ((54, 201, 105),  [34, 197, 94, 255]),    # 2.0 - 5.0 mm -> Sattes Smaragdgrün
    ((0, 215, 215),   [245, 205, 20, 255]),   # 5.0 - 10.0 mm -> Leuchtendes Goldgelb
    ((0, 166, 221),   [250, 165, 15, 255]),   # 10.0 - 20.0 mm -> Warmes Sonnengelb/Orange
    ((0, 0, 254),     [245, 100, 20, 255]),   # 20.0 - 35.0 mm -> Kräftiges Orange-Rot
    ((151, 48, 194),  [230, 40, 40, 255]),    # 35.0 - 50.0 mm -> Karminrot
    ((217, 38, 199),  [195, 20, 35, 255]),    # 50.0 - 75.0 mm -> Tiefes Scharlachrot
    ((254, 0, 0),     [215, 50, 220, 255]),   # 75.0 - 100.0 mm -> Leuchtendes Magenta
    ((160, 0, 0),     [250, 200, 255, 255]),  # > 100 mm -> Extremes Weiß-Violett
]


def recolor_wms_to_turbo(img_rgba):
    """
    Wandelt die DWD ICON-EU WMS-Farben exakt in die aus generate_radar.py bekannten
    Turbo-Radar-Farben um (Türkis -> Grün -> Goldgelb -> Orange -> Rot -> Magenta).
    """
    arr = np.array(img_rgba)
    h, w, c = arr.shape
    if c < 4:
        return img_rgba

    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    a = arr[:, :, 3].astype(int)

    new_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Hintergrund / Nicht-Niederschlag (Transparenz oder reines Weiß / Schwarz / Grau)
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    is_bg = (a < 30) | ((max_c > 250) & (min_c > 240)) | ((max_c < 25) & (min_c < 25))

    # Ordne jedem Pixel die am besten passende DWD-WMS-Klasse zu
    for (dwd_r, dwd_g, dwd_b), turbo_rgba in DWD_WMS_COLOR_MAP:
        dist_sq = (r - dwd_r)**2 + (g - dwd_g)**2 + (b - dwd_b)**2
        # Exakter Match oder minimale Antialiasing-Toleranz
        mask = (~is_bg) & (dist_sq < 28**2)
        new_rgba[mask] = turbo_rgba

    # Ränder mit interpolierten Farben: Nächster Nachbar unter den Klassen
    unmatched = (~is_bg) & (new_rgba[:, :, 3] == 0)
    if np.any(unmatched):
        best_dist = np.full((h, w), 1e9, dtype=np.float32)
        best_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        for (dwd_r, dwd_g, dwd_b), turbo_rgba in DWD_WMS_COLOR_MAP:
            d = (r - dwd_r)**2 + (g - dwd_g)**2 + (b - dwd_b)**2
            closer = unmatched & (d < best_dist)
            best_dist[closer] = d[closer]
            best_rgba[closer] = turbo_rgba
            
        new_rgba[unmatched] = best_rgba[unmatched]

    return Image.fromarray(new_rgba, mode='RGBA')


def lat_lon_to_web_mercator(lat, lon):
    """Konvertiert WGS84 Lat/Lon in EPSG:3857 Web-Mercator Koordinaten (in Metern)."""
    r = 6378137.0
    x = lon * (np.pi / 180.0) * r
    y = np.log(np.tan((np.pi / 4.0) + (np.radians(lat) / 2.0))) * r
    return x, y


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

    # Exakte Web-Mercator Bounding Box für ganz Europa
    min_lat, min_lon = EUROPE_BOUNDS[0]
    max_lat, max_lon = EUROPE_BOUNDS[1]
    min_x, min_y = lat_lon_to_web_mercator(min_lat, min_lon)
    max_x, max_y = lat_lon_to_web_mercator(max_lat, max_lon)
    merc_bbox_str = f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}"

    steps = []
    for i in range(total_steps):
        valid_dt = ref_date + timedelta(hours=(i + 1) * step_hours)
        valid_iso = valid_dt.strftime("%Y-%m-%dT%H:00:00.000Z")
        steps.append((i, (i + 1) * step_hours, valid_dt, valid_iso))

    results = []

    def fetch_and_process_step(step_idx, forecast_hour, valid_dt, valid_iso):
        file_name = f"europe_{step_idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)

        # DWD Geoserver WMS für ICON-EU 3h Niederschlag - Nativ in EPSG:3857 (Web Mercator)
        wms_url = (
            f"https://maps.dwd.de/geoserver/dwd/wms?"
            f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&"
            f"LAYERS=dwd:Icon-eu_reg00625_fd_sl_TOTPREC03H&"
            f"STYLES=&CRS=EPSG:3857&"
            f"BBOX={merc_bbox_str}&"
            f"WIDTH=1400&HEIGHT=1100&"
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
