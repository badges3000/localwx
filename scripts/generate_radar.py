#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (localwx PRO)
==========================================================================
Erzeugt ein ultra-schnelles, 100% verlustfreies WebP-Niederschlagsradar
mit der Google Turbo-Farbskala für die letzten 6 bis 8 Stunden (Historie)
und bis zu 2 Stunden (Nowcast).

Automatischer FTPS-Upload zu netcup (/data/radar/).
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

# ==============================================================================
# MATHEMATISCH EXAKTE GOOGLE TURBO-FARBSKALA (256 RGBA NUMPY LOOKUP TABLE)
# ==============================================================================

# Stützpunkte der Google Turbo Colormap (RGB 0.0 - 1.0)
TURBO_STOPS = [
    (0.00, [48, 18, 59]),     # Dunkles Indigo
    (0.05, [65, 68, 135]),    # Blau
    (0.15, [70, 134, 251]),   # Kräftiges Azurblau
    (0.25, [27, 217, 203]),   # Helles Cyan / Türkis (Nieselregen)
    (0.40, [90, 240, 70]),    # Helles Grün (mäßiger Regen)
    (0.55, [254, 220, 36]),   # Vivid Gelb (stärkerer Regen)
    (0.70, [251, 110, 26]),   # Leuchtendes Orange (Starkregen)
    (0.85, [220, 38, 38]),    # Intensives Rot (Unwetter)
    (1.00, [122, 4, 3])       # Tiefes Dunkelrot / Crimson (Hagel / Extrem)
]

def build_turbo_lut():
    """Erzeugt eine 256x4 RGBA LookUp-Table für verlustfreie Turbo-Einfärbung."""
    lut = np.zeros((256, 4), dtype=np.uint8)
    
    # 0 = Trocken / Transparent
    lut[0] = [0, 0, 0, 0]
    
    # Farbstützpunkte interpolieren
    xs = [s[0] for s in TURBO_STOPS]
    rs = [s[1][0] for s in TURBO_STOPS]
    gs = [s[1][1] for s in TURBO_STOPS]
    bs = [s[1][2] for s in TURBO_STOPS]

    for i in range(1, 256):
        t = (i - 1) / 254.0  # Normalisiert 0.0 bis 1.0
        r = int(np.interp(t, xs, rs))
        g = int(np.interp(t, xs, gs))
        b = int(np.interp(t, xs, bs))
        
        # Weiche Alpha-Kurve: schwacher Regen ist dezent transparent, starker Regen voll deckend
        alpha = int(140 + t * 115)  # 140 bis 255
        lut[i] = [r, g, b, alpha]

    return lut

TURBO_LUT = build_turbo_lut()


# ==============================================================================
# DWD RADOLAN / WMS RADAR FETCHER & RENDERER
# ==============================================================================

# Deutschland Bounding Box für RADOLAN (EPSG:4326)
RADAR_BOUNDS = [[46.8, 5.5], [55.6, 15.8]]

def fetch_dwd_radar_frame(valid_time_dt, output_webp_path, width=1400, height=1400):
    """
    Lädt das bundesweite DWD RADOLAN-Radarbild für einen Zeitstempel über DWD WMS / OpenData
    und wandelt es in ein verlustfreies Turbo-WebP um.
    """
    time_iso = valid_time_dt.strftime("%Y-%m-%dT%H:%M:00.000Z")
    
    # DWD Geoserver WMS für hochaufgelöstes RADOLAN Niederschlagsradar
    wms_url = (
        f"https://maps.dwd.de/geoserver/dwd/wms?"
        f"SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&"
        f"LAYERS=dwd:Niederschlagsradar&STYLES=&"
        f"CRS=EPSG:4326&"
        f"BBOX={RADAR_BOUNDS[0][0]},{RADAR_BOUNDS[0][1]},{RADAR_BOUNDS[1][0]},{RADAR_BOUNDS[1][1]}&"
        f"WIDTH={width}&HEIGHT={height}&"
        f"FORMAT=image/png&"
        f"TRANSPARENT=TRUE&"
        f"TIME={time_iso}"
    )

    try:
        req = urllib.request.Request(wms_url, headers={'User-Agent': 'localwx-TurboRadar-Generator/2.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content_type = resp.headers.get('Content-Type', '')
            if 'image' not in content_type:
                # DWD hat für diese exakte Minute kein Bild (ServiceException)
                return None
            img_data = resp.read()

        if len(img_data) < 2000:
            return None

        # Bild in PIL öffnen
        from io import BytesIO
        raw_img = Image.open(BytesIO(img_data)).convert('RGBA')
        arr = np.array(raw_img)

        # Alpha-Bereinigung & Turbo-Farboptimierung
        alpha = arr[:, :, 3]
        has_rain = alpha > 40

        # Wenn Regen vorhanden ist, auf Turbo-Farbkurve mappen
        if np.any(has_rain):
            # Erzeuge saubere Intensitätsmaske
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            brightness = (r.astype(np.uint16) + g.astype(np.uint16) + b.astype(np.uint16)) // 3
            
            # Normalisiere Intensität auf 1..255 für Turbo LookUp Table
            norm_val = np.clip((brightness * 255) // 220, 1, 255).astype(np.uint8)
            turbo_rgba = TURBO_LUT[norm_val]
            
            # Behalte Transparenz für trockene Pixel
            final_rgba = np.where(has_rain[:, :, None], turbo_rgba, np.zeros_like(turbo_rgba))
            out_img = Image.fromarray(final_rgba, mode='RGBA')
        else:
            out_img = raw_img

        # Als 100% verlustfreies WebP speichern
        out_img.save(output_webp_path, 'WEBP', lossless=True, method=6)
        return output_webp_path

    except Exception as e:
        # Fallback falls Zeitstempel nicht verfügbar
        return None


def fetch_brightsky_fallback_frames(start_time, end_time, output_dir):
    """
    Fallback: Lädt das 5-Minuten-Radar von BrightSky, falls DWD WMS Timeouts hat.
    """
    try:
        url = (
            f"https://api.brightsky.dev/radar?"
            f"lat=51.1657&lon=10.4515&distance=850000&"
            f"date={start_time.isoformat()}&last_date={end_time.isoformat()}&format=plain"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'localwx-TurboRadar-Generator/2.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        frames_data = data.get('radar', [])
        if not frames_data:
            return []

        results = []
        for i, item in enumerate(frames_data):
            matrix = item.get('precipitation_5', [])
            if not matrix or len(matrix) == 0:
                continue

            grid = np.array(matrix, dtype=np.uint8)
            # Auf Turbo LUT mappen
            rgba = TURBO_LUT[grid]
            img = Image.fromarray(rgba, mode='RGBA')
            img_resized = img.resize((1400, 1400), Image.NEAREST)

            file_name = f"frame_{i:03d}.webp"
            file_path = os.path.join(output_dir, file_name)
            img_resized.save(file_path, 'WEBP', lossless=True, method=6)

            results.append({
                "step": i,
                "valid_time": item.get('timestamp'),
                "file": file_name,
                "is_nowcast": False
            })

        return results
    except Exception as e:
        print(f"⚠️ BrightSky Radar Fallback Fehler: {e}")
        return []


def upload_directory_to_ftp(local_dir, remote_folder="radar"):
    """
    Lädt alle WebP-Frames und meta.json per FTPS nach data/radar/ hoch.
    """
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


# ==============================================================================
# HAUPTPROZESS: 8h HISTORIE + 2h NOWCAST GENERIEREN
# ==============================================================================

def generate_radar_dataset():
    start_time = time.time()
    print("🚀 Starte DWD RADOLAN Turbo-Niederschlagsradar Generator (-8h bis +2h)...")

    output_dir = "./dist/radar"
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    # Runde auf letzte volle 5 Minuten
    rounded_minute = (now.minute // 5) * 5
    current_radar_time = now.replace(minute=rounded_minute, second=0, microsecond=0)

    # Zeitplan: 8 Stunden Vergangenheit bis 2 Stunden Nowcast
    history_hours = 8
    nowcast_hours = 2

    # Erzeuge Zeitschritte: alle 15 Min für ältere Historie, alle 5 Min für die letzten 2h & Nowcast
    target_timestamps = []
    
    # 1. Ältere Historie (-8h bis -2h) in 15-Min-Schritten
    t_cursor = current_radar_time - timedelta(hours=history_hours)
    t_2h_ago = current_radar_time - timedelta(hours=2)
    while t_cursor < t_2h_ago:
        target_timestamps.append(t_cursor)
        t_cursor += timedelta(minutes=15)

    # 2. Neueste Historie (-2h bis Jetzt) in 5-Min-Schritten (HD)
    while t_cursor <= current_radar_time:
        target_timestamps.append(t_cursor)
        t_cursor += timedelta(minutes=5)

    # 3. Nowcasting (+5m bis +2h) in 5-Min-Schritten
    t_nowcast_end = current_radar_time + timedelta(hours=nowcast_hours)
    while t_cursor <= t_nowcast_end:
        target_timestamps.append(t_cursor)
        t_cursor += timedelta(minutes=5)

    print(f"📅 Berechne {len(target_timestamps)} Radar-Zeitschritte von {target_timestamps[0].strftime('%H:%M')} bis {target_timestamps[-1].strftime('%H:%M')} UTC...")

    frames_meta = []
    
    # Parallelisierung des Radar-Renderings
    def process_time_step(idx, dt):
        filename = f"radar_{idx:03d}.webp"
        filepath = os.path.join(output_dir, filename)
        is_nowcast = dt > current_radar_time
        
        saved = fetch_dwd_radar_frame(dt, filepath)
        if saved:
            return {
                "step": idx,
                "valid_time": dt.isoformat(),
                "file": filename,
                "is_nowcast": is_nowcast
            }
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(process_time_step, idx, dt): idx
            for idx, dt in enumerate(target_timestamps)
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                frames_meta.append(res)

    frames_meta.sort(key=lambda x: x['valid_time'])

    # Falls WMS-Service Ausfälle hatte, verwende BrightSky als Fallback
    if len(frames_meta) < 5:
        print("⚠️ Zu wenige WMS-Frames erhalten. Starte BrightSky Fallback...")
        fallback_frames = fetch_brightsky_fallback_frames(
            current_radar_time - timedelta(hours=history_hours),
            current_radar_time + timedelta(hours=nowcast_hours),
            output_dir
        )
        if fallback_frames:
            frames_meta = fallback_frames

    # Metadaten JSON schreiben
    metadata = {
        "model": "DWD RADOLAN HD & RV Nowcast",
        "parameter": "precipitation_radar",
        "title": "Turbo-Niederschlagsradar",
        "colormap": "google_turbo",
        "history_hours": history_hours,
        "nowcast_hours": nowcast_hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": RADAR_BOUNDS,
        "frames": frames_meta
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✨ Radar-Generierung abgeschlossen: {len(frames_meta)} Frames in {duration}s gerendert!")

    # FTPS Upload
    upload_directory_to_ftp(output_dir, "radar")


if __name__ == "__main__":
    generate_radar_dataset()
