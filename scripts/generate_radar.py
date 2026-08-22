#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (localwx PRO)
==========================================================================
Erzeugt ein ultra-schnelles, 100% verlustfreies WebP-Niederschlagsradar
mit der echten Google Turbo-Farbskala für die letzten 6 bis 8 Stunden (Historie)
und 2 Stunden (Nowcast) direkt aus den rohen DWD RADOLAN-Rasterdaten.

- 0% Hintergrund-Artefakte (überall 100% transparent wo es nicht regnet)
- Exakte Turbo-Farben passend zur Legende
- 1400x1400 Pixel Lossless WebP
- Automatischer FTPS-Upload zu netcup (/data/radar/)
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

# ==============================================================================
# ECHTE GOOGLE TURBO-FARBSKALA (256 RGBA NUMPY LOOKUP TABLE)
# ==============================================================================

def build_turbo_lut():
    """
    Erzeugt eine 256x4 RGBA LookUp-Table passend zur Radar-Legende:
    0:        100% Transparent
    1..25:    Leichter Niesel / Regen (< 0.5 mm) -> Saphirblau / Indigo
    26..65:   Leichter bis mäßiger Regen (0.5 - 2.5 mm) -> Cyan / Mint-Türkis
    66..110:  Mäßiger bis starker Regen (2.5 - 5.0 mm) -> Frisches Lime-Grün
    111..160: Kräftiger Schauer (5.0 - 10 mm) -> Vivid Gold / Gelb
    161..210: Starkregen (10 - 25 mm) -> Leuchtendes Orange / Rot
    211..255: Extremer Starkregen / Hagel (> 25 mm) -> Magenta / Tiefviolett
    """
    lut = np.zeros((256, 4), dtype=np.uint8)
    
    # 0 = Trocken (100% Transparent)
    lut[0] = [0, 0, 0, 0]

    for i in range(1, 256):
        if i < 25:
            # 1 - 24: Saphirblau bis Indigo (Nieselregen)
            t = (i - 1) / 24.0
            r = int(40 + t * 20)
            g = int(80 + t * 70)
            b = int(220 + t * 30)
            a = int(140 + t * 45)
        elif i < 65:
            # 25 - 64: Cyan bis Türkis / Mint (0.5 - 2.5 mm)
            t = (i - 25) / 40.0
            r = int(20 + t * 15)
            g = int(190 + t * 35)
            b = int(235 - t * 25)
            a = int(185 + t * 35)
        elif i < 110:
            # 65 - 109: Frisches Lime-Grün (2.5 - 5.0 mm)
            t = (i - 65) / 45.0
            r = int(50 + t * 140)
            g = int(225 + t * 25)
            b = int(80 - t * 40)
            a = int(220 + t * 25)
        elif i < 160:
            # 110 - 159: Vivid Gold / Gelb (5.0 - 10 mm)
            t = (i - 110) / 50.0
            r = int(245 + t * 10)
            g = int(215 - t * 70)
            b = int(35 - t * 15)
            a = int(245 + t * 10)
        elif i < 210:
            # 160 - 209: Leuchtendes Orange bis Knallrot (10 - 25 mm)
            t = (i - 160) / 50.0
            r = int(245 - t * 15)
            g = int(110 - t * 80)
            b = int(20 + t * 15)
            a = 255
        else:
            # 210 - 255: Magenta bis Tiefviolett (Extremer Starkregen / Hagel)
            t = (i - 210) / 45.0
            r = int(210 - t * 70)
            g = int(30 - t * 15)
            b = int(180 + t * 65)
            a = 255
            
        lut[i] = [r, g, b, a]

    return lut

TURBO_LUT = build_turbo_lut()


# Deutschland Bounding Box für RADOLAN
RADAR_BOUNDS = [[46.8, 5.5], [55.6, 15.8]]


def generate_radar_dataset():
    start_time = time.time()
    print("🚀 Starte DWD RADOLAN Turbo-Niederschlagsradar Generator (-8h bis +2h)...")

    output_dir = "./dist/radar"
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    # 8 Stunden Vergangenheit bis 2 Stunden Nowcast
    history_hours = 8
    nowcast_hours = 2
    past_time = now - timedelta(hours=history_hours)
    future_time = now + timedelta(hours=nowcast_hours)

    date_str = past_time.strftime("%Y-%m-%dT%H:00:00Z")
    last_date_str = future_time.strftime("%Y-%m-%dT%H:00:00Z")

    params = urllib.parse.urlencode({
        'lat': '51.1657',
        'lon': '10.4515',
        'distance': '850000',
        'date': date_str,
        'last_date': last_date_str,
        'format': 'plain'
    })
    url = f"https://api.brightsky.dev/radar?{params}"

    data = None
    for attempt in range(3):
        try:
            print(f"📡 Lade DWD RADOLAN-Daten (Versuch {attempt+1}/3): {date_str} bis {last_date_str}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'localwx-TurboRadar-Generator/2.0'})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data and data.get('radar'):
                print(f"✅ Erfolgreich {len(data['radar'])} DWD RADOLAN-Frames empfangen!")
                break
        except urllib.error.HTTPError as he:
            print(f"⚠️ HTTP-Fehler ({he.code} {he.reason}) bei Anfrage. Versuche Fallback...")
            params_fb = urllib.parse.urlencode({
                'lat': '51.1657',
                'lon': '10.4515',
                'distance': '850000',
                'date': date_str,
                'format': 'plain'
            })
            url = f"https://api.brightsky.dev/radar?{params_fb}"
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Verbindungsfehler: {e}")
            time.sleep(2)

    if not data or not data.get('radar'):
        print("❌ Keine Radar-Frames vom DWD-Service erhalten!")
        return

    frames_data = data.get('radar', [])

    print(f"📦 Verarbeite {len(frames_data)} DWD RADOLAN Raster-Frames mit Turbo-Farben...")

    # Bounding Box aus Geometrie ermitteln
    valid_bounds = RADAR_BOUNDS
    if data.get('geometry') and data['geometry'].get('coordinates'):
        try:
            flat_coords = []
            def extract_coords(obj):
                if isinstance(obj, (list, tuple)):
                    if len(obj) == 2 and isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                        flat_coords.append(obj)
                    else:
                        for item in obj:
                            extract_coords(item)
            extract_coords(data['geometry']['coordinates'])
            if len(flat_coords) >= 4:
                lats = [c[1] for c in flat_coords]
                lons = [c[0] for c in flat_coords]
                valid_bounds = [[round(min(lats), 2), round(min(lons), 2)], [round(max(lats), 2), round(max(lons), 2)]]
        except Exception:
            valid_bounds = RADAR_BOUNDS

    results = []

    def process_single_matrix(idx, item):
        matrix = item.get('precipitation_5', [])
        if not matrix or len(matrix) == 0:
            return None

        # Rohe 2D-Matrix in NumPy Array umwandeln (0..255)
        grid = np.array(matrix, dtype=np.uint8)
        
        # Auf exakte Turbo LookUp-Table mappen (0 = 100% transparent!)
        rgba = TURBO_LUT[grid]
        img = Image.fromarray(rgba, mode='RGBA')
        
        # Auf scharfe 1400x1400 Auflösung skalieren (NEAREST für gestochen scharfe Kanten)
        img_resized = img.resize((1400, 1400), Image.NEAREST)

        file_name = f"radar_{idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)
        
        # Als 100% verlustfreies WebP speichern (keine Kompressionsartefakte!)
        img_resized.save(file_path, 'WEBP', lossless=True, method=6)

        ts_str = item.get('timestamp')
        is_nowcast = False
        if ts_str:
            try:
                frame_dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                is_nowcast = frame_dt > now
            except Exception:
                pass

        return {
            "step": idx,
            "valid_time": ts_str,
            "file": file_name,
            "is_nowcast": is_nowcast
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_single_matrix, i, item): i
            for i, item in enumerate(frames_data)
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    results.sort(key=lambda x: x['step'])

    # Metadaten JSON schreiben
    metadata = {
        "model": "DWD RADOLAN HD",
        "parameter": "precipitation_radar",
        "title": "Turbo-Niederschlagsradar",
        "colormap": "google_turbo",
        "history_hours": history_hours,
        "nowcast_hours": nowcast_hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": valid_bounds,
        "frames": results
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✨ Radar-Generierung erfolgreich: {len(results)} Turbo-Frames in {duration}s gerendert!")

    # FTPS Upload
    upload_directory_to_ftp(output_dir, "radar")


def upload_directory_to_ftp(local_dir, remote_folder="radar"):
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
    generate_radar_dataset()
