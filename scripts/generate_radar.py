#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (100% DWD OpenData)
================================================================================
Lädt hochauflösende DWD RADOLAN HD Radardaten im lückenlosen 5-Minuten-Takt von opendata.dwd.de:
- -8h bis 0h Vergangenheit: Alle 5-Minuten-Messungen (DWD RADOLAN RV)
- 0h bis +2h Nowcasting: Alle 5-Minuten-Vorhersageschritte (+5m bis +120m)
- Bereinigung von Inversions-Bodenclutter & Sensorfehlern (Schwellenwert >= 0.48 mm/h)
- Meteorologisch kalibrierte Google Turbo Farbtabelle (WarnWetter-Dynamik)
- 100% verlustfreies WebP (lossless=True, method=6)
- Erzeugt meta.json und lädt per FTPS nach /data/radar/ hoch.
"""

import os
import sys
import json
import time
import bz2
import tarfile
import io
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

# Exakte DWD RADOLAN DE1200 Bounding Box (1100x1200 Polar-Stereographic Grid)
# SW: [45.68°N, 1.46°E] bis NE: [55.86°N, 18.73°E]
RADAR_BOUNDS = [[45.68, 1.46], [55.86, 18.73]]


def build_turbo_lut():
    """
    Erstellt die meteorologisch kalibrierte Google Turbo LookUp-Table (0..255).
    """
    lut = np.zeros((256, 4), dtype=np.uint8)

    for i in range(256):
        if i == 0:
            # 0: Absolut transparent (Kein Niederschlag)
            r, g, b, a = 0, 0, 0, 0
        elif i < 26:
            # 1 - 25: Zarter Türkis / Cyan Außensaum
            t = (i - 1) / 24.0
            r = int(35 + t * 20)
            g = int(180 + t * 45)
            b = int(245 - t * 20)
            a = int(160 + t * 50)
        elif i < 91:
            # 26 - 90: Satte Smaragd- & Lime-Grüntöne (Leichter bis mäßiger Regen)
            t = (i - 26) / 64.0
            r = int(35 + t * 115)
            g = int(215 + t * 25)
            b = int(120 - t * 105)
            a = int(220 + t * 25)
        elif i < 151:
            # 91 - 150: Leuchtendes Goldgelb bis Sonnengelb
            t = (i - 91) / 59.0
            r = int(235 + t * 20)
            g = int(210 - t * 50)
            b = int(20 - t * 10)
            a = 250
        elif i < 201:
            # 151 - 200: Kräftiges Orange (Starkregen)
            t = (i - 151) / 49.0
            r = int(245 - t * 10)
            g = int(140 - t * 75)
            b = int(15 + t * 10)
            a = 255
        elif i < 236:
            # 201 - 235: Intensives Karminrot bis Scharlachrot
            t = (i - 201) / 34.0
            r = int(235 - t * 25)
            g = int(45 - t * 25)
            b = int(25 + t * 65)
            a = 255
        else:
            # 236 - 255: Magenta bis Weiß-Violett (Extremer Starkregen / Hagel)
            t = (i - 236) / 19.0
            r = int(210 + t * 45)
            g = int(30 + t * 220)
            b = int(210 + t * 45)
            a = 255
            
        lut[i] = [r, g, b, a]

    return lut

TURBO_LUT = build_turbo_lut()


def map_radolan_val_to_index(val):
    """
    Mappt DWD RADOLAN RV Rohwerte auf 256 Turbo-Farbstufen.
    Filtert Sensorrauschen und Bodenclutter unterhalb von 0.48 mm/h (< 18 dBZ) heraus.
    """
    idx = np.zeros_like(val, dtype=np.uint8)
    
    # 1. Zarter Regen erst ab val >= 4 (>= 0.48 mm/h)
    m1 = (val >= 4) & (val < 8)
    idx[m1] = (1 + ((val[m1] - 4) / 4.0) * 24).astype(np.uint8)
    
    # 2. Leichter bis mäßiger Landregen (Sattes Grün): val 8..24 (ca. 1.0..3.0 mm/h)
    m2 = (val >= 8) & (val < 25)
    idx[m2] = (26 + ((val[m2] - 8) / 17.0) * 64).astype(np.uint8)
    
    # 3. Kräftiger Schauer (Goldgelb): val 25..55 (3.0..6.6 mm/h)
    m3 = (val >= 25) & (val < 56)
    idx[m3] = (91 + ((val[m3] - 25) / 31.0) * 59).astype(np.uint8)
    
    # 4. Starkregen (Orange): val 56..140 (6.7..16.8 mm/h)
    m4 = (val >= 56) & (val < 141)
    idx[m4] = (151 + ((val[m4] - 56) / 85.0) * 49).astype(np.uint8)
    
    # 5. Extremregen (Karminrot): val 141..280 (16.9..33.6 mm/h)
    m5 = (val >= 141) & (val < 281)
    idx[m5] = (201 + ((val[m5] - 141) / 140.0) * 34).astype(np.uint8)
    
    # 6. Unwetter / Hagel (Magenta/Weiß): val >= 281 (> 33.6 mm/h)
    m6 = val >= 281
    idx[m6] = np.clip(236 + (val[m6] - 281) / 5.0, 236, 255).astype(np.uint8)
    
    return idx


def parse_radolan_binary(data_bytes):
    """
    Parst das binäre DWD RADOLAN-Kompositfile und maskiert NoData- und Clutter-Bits.
    """
    etx_pos = data_bytes.find(b'\x03')
    if etx_pos == -1:
        etx_pos = data_bytes.find(b'\n\x00')
    if etx_pos == -1:
        return None, None

    header = data_bytes[:etx_pos].decode('latin1', errors='ignore')
    raw_data = data_bytes[etx_pos + 1:]

    width = 1100
    height = 1200
    if '1200x1100' in header:
        width, height = 1100, 1200
    elif '900x900' in header:
        width, height = 900, 900

    expected_16bit = width * height * 2
    if len(raw_data) >= expected_16bit:
        arr = np.frombuffer(raw_data[:expected_16bit], dtype=np.uint16).reshape((height, width))
        
        # Maskiere NoData (Bit 13) und Clutter (Bit 12)
        is_invalid = ((arr & 0x2000) > 0) | ((arr & 0x1000) > 0)
        
        # Reine Datenbits extrahieren (Bits 0-11)
        val = arr & 0x0FFF
        val[is_invalid] = 0
        
        grid_indexed = map_radolan_val_to_index(val)
        grid_indexed = np.flipud(grid_indexed)
        return header, grid_indexed

    elif len(raw_data) >= width * height:
        arr = np.frombuffer(raw_data[:width * height], dtype=np.uint8).reshape((height, width))
        arr = np.flipud(arr)
        return header, arr

    return None, None


def get_available_dwd_rv_files():
    """
    Listet alle verfügbaren RADOLAN RV tar.bz2 Dateien auf opendata.dwd.de auf.
    """
    url = "https://opendata.dwd.de/weather/radar/composite/rv/"
    req = urllib.request.Request(url, headers={'User-Agent': 'localwx-RADOLAN-Engine/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        
        pattern = r'href="(DE1200_RV(\d{10})\.tar\.bz2)"'
        matches = re.findall(pattern, html)
        
        files = []
        for filename, dt_str in matches:
            try:
                dt = datetime.strptime(f"20{dt_str}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                files.append((filename, dt))
            except Exception:
                pass
        
        files.sort(key=lambda x: x[1])
        return files
    except Exception as e:
        print(f"⚠️ Fehler beim Abrufen des DWD-Index: {e}")
        return []


def download_and_extract_tar_bz2(filename):
    """
    Lädt eine einzelne DE1200_RV tar.bz2 Datei von DWD OpenData herunter und entpackt sie im Speicher.
    """
    url = f"https://opendata.dwd.de/weather/radar/composite/rv/{filename}"
    req = urllib.request.Request(url, headers={'User-Agent': 'localwx-RADOLAN-Engine/2.0'})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            compressed_bytes = resp.read()
        
        tar_bytes = bz2.decompress(compressed_bytes)
        tar_stream = io.BytesIO(tar_bytes)
        
        extracted_files = {}
        with tarfile.open(fileobj=tar_stream, mode="r:") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        extracted_files[member.name] = f.read()
        
        return extracted_files
    except Exception as e:
        print(f"⚠️ Fehler beim Laden von {filename}: {e}")
        return {}


def render_matrix_to_webp(grid_2d, output_path):
    """
    Wendet die WarnWetter-Turbo LUT an, glättet die Konturen
    und speichert als verlustfreies WebP.
    """
    rgba = TURBO_LUT[grid_2d]
    img = Image.fromarray(rgba, mode='RGBA')
    
    img_resized = img.resize((1400, 1400), Image.BILINEAR)
    
    arr = np.array(img_resized)
    arr[arr[:, :, 3] < 10, 3] = 0
    
    img_clean = Image.fromarray(arr, mode='RGBA')
    img_clean.save(output_path, 'WEBP', lossless=True, method=6)


def generate_radar_dataset():
    start_time = time.time()
    print("🚀 Starte DWD RADOLAN HD 5-Minuten Turbo-Radar Generator (100% DWD OpenData)...")

    output_dir = "./dist/radar"
    os.makedirs(output_dir, exist_ok=True)

    dwd_files = get_available_dwd_rv_files()
    if not dwd_files:
        print("❌ Keine RADOLAN-Dateien auf opendata.dwd.de gefunden!")
        return

    now = datetime.now(timezone.utc)
    print(f"📡 DWD OpenData Server erreichbar ({len(dwd_files)} RADOLAN RV Komposite verfügbar).")

    # Letzte 8 Stunden Historie
    eight_hours_ago = now - timedelta(hours=8)
    selected_history = [f for f in dwd_files if f[1] >= eight_hours_ago and f[1] <= now]

    print(f"📦 Lade {len(selected_history)} lückenlose 5-Minuten-Messungen (-8h bis Jetzt)...")

    frames_metadata = []

    def process_history_item(item, idx):
        filename, valid_dt = item
        data_dict = download_and_extract_tar_bz2(filename)
        if not data_dict:
            return None

        main_key = next((k for k in sorted(data_dict.keys()) if '_000' in k or k.endswith('000')), None)
        if not main_key:
            main_key = sorted(data_dict.keys())[0]

        file_bytes = data_dict[main_key]
        header, grid = parse_radolan_binary(file_bytes)
        if grid is None:
            return None

        file_name = f"radar_{idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)
        render_matrix_to_webp(grid, file_path)

        return {
            "step": idx,
            "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:00Z"),
            "file": file_name,
            "is_nowcast": False
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_history_item, item, i): i for i, item in enumerate(selected_history)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                frames_metadata.append(res)

    frames_metadata.sort(key=lambda x: x['valid_time'])

    for i, f in enumerate(frames_metadata):
        f['step'] = i
        old_name = f['file']
        new_name = f"radar_{i:03d}.webp"
        if old_name != new_name:
            old_p = os.path.join(output_dir, old_name)
            new_p = os.path.join(output_dir, new_name)
            if os.path.exists(old_p):
                os.rename(old_p, new_p)
            f['file'] = new_name

    frame_idx = len(frames_metadata)
    print(f"✅ {frame_idx} historische 5-Minuten-Frames erfolgreich generiert.")

    # Nowcast (+2h)
    latest_file = dwd_files[-1][0]
    latest_dt = dwd_files[-1][1]
    print(f"🔮 Lade DWD 5-Minuten Nowcast (+2h) aus neuester Datei: {latest_file}...")

    nowcast_data = download_and_extract_tar_bz2(latest_file)
    if nowcast_data:
        nowcast_keys = [k for k in sorted(nowcast_data.keys()) if not k.endswith('_000')]
        selected_nowcast_keys = []
        for k in nowcast_keys:
            m = re.search(r'_(\d{3})$', k)
            if m:
                minutes = int(m.group(1))
                selected_nowcast_keys.append((k, minutes))
        
        selected_nowcast_keys.sort(key=lambda x: x[1])

        for k, minutes in selected_nowcast_keys:
            file_bytes = nowcast_data[k]
            header, grid = parse_radolan_binary(file_bytes)
            if grid is not None:
                valid_dt = latest_dt + timedelta(minutes=minutes)
                file_name = f"radar_{frame_idx:03d}.webp"
                file_path = os.path.join(output_dir, file_name)
                render_matrix_to_webp(grid, file_path)

                frames_metadata.append({
                    "step": frame_idx,
                    "valid_time": valid_dt.strftime("%Y-%m-%dT%H:%M:00Z"),
                    "file": file_name,
                    "is_nowcast": True
                })
                frame_idx += 1

    print(f"✨ Gesamt-Datensatz: {len(frames_metadata)} flüssige 5-Minuten-Frames fertig!")

    metadata = {
        "model": "DWD RADOLAN HD",
        "parameter": "precipitation_radar",
        "title": "DWD RADOLAN HD Doppler-Radar (-8h bis +2h, 5-Minuten-Takt)",
        "unit": "mm/h",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bounds": RADAR_BOUNDS,
        "frames": frames_metadata
    }

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"🎉 Radar-Generierung in {duration}s abgeschlossen!")

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
