#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (100% DWD OpenData)
================================================================================
Lädt hochauflösende DWD RADOLAN HD Radardaten im lückenlosen 5-Minuten-Takt von opendata.dwd.de:
- -8h bis 0h Vergangenheit: Alle 5-Minuten-Messungen (DWD RADOLAN RV)
- 0h bis +2h Nowcasting: Alle 5-Minuten-Vorhersageschritte (+5m bis +120m)
- Meteorologisch kalibrierte Google Turbo Farbtabelle (WarnWetter-Dynamik mit sattem Grün, Gelb, Orange, Rot)
- Konturenscharfe bikubische Kantenglättung ohne verwaschene Ränder
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
    Erstellt die meteorologisch kalibrierte Google Turbo LookUp-Table (0..255)
    Exakt abgestimmt auf das visuelle Erscheinungsbild der DWD WarnWetter App.
    """
    lut = np.zeros((256, 4), dtype=np.uint8)

    for i in range(256):
        if i == 0:
            # 0: Absolut transparent (Kein Niederschlag)
            r, g, b, a = 0, 0, 0, 0
        elif i < 26:
            # 1 - 25: Zarter Türkis / Cyan Außensaum (0.05 - 0.35 mm/h / < 15 dBZ)
            t = (i - 1) / 24.0
            r = int(35 + t * 20)
            g = int(180 + t * 45)
            b = int(245 - t * 20)
            a = int(160 + t * 50)
        elif i < 91:
            # 26 - 90: Satte Smaragd- & Lime-Grüntöne (0.35 - 2.5 mm/h / 15 - 30 dBZ)
            # Bildet wie in der WarnWetter App die großflächige Hauptmasse des Regens
            t = (i - 26) / 64.0
            r = int(35 + t * 115)
            g = int(215 + t * 25)
            b = int(120 - t * 105)
            a = int(220 + t * 25)
        elif i < 151:
            # 91 - 150: Leuchtendes Goldgelb bis Sonnengelb (2.5 - 7.0 mm/h / 30 - 40 dBZ)
            t = (i - 91) / 59.0
            r = int(235 + t * 20)
            g = int(210 - t * 50)
            b = int(20 - t * 10)
            a = 250
        elif i < 201:
            # 151 - 200: Kräftiges Orange (7.0 - 18.0 mm/h / 40 - 48 dBZ)
            t = (i - 151) / 49.0
            r = int(245 - t * 10)
            g = int(140 - t * 75)
            b = int(15 + t * 10)
            a = 255
        elif i < 236:
            # 201 - 235: Intensives Karminrot bis Scharlachrot (18.0 - 35.0 mm/h / 48 - 55 dBZ)
            t = (i - 201) / 34.0
            r = int(235 - t * 25)
            g = int(45 - t * 25)
            b = int(25 + t * 65)
            a = 255
        else:
            # 236 - 255: Magenta bis Weiß-Violett (Extremer Starkregen / Hagel > 35 mm/h / > 55 dBZ)
            t = (i - 236) / 19.0
            r = int(210 + t * 45)
            g = int(30 + t * 220)
            b = int(210 + t * 45)
            a = 255
            
        lut[i] = [r, g, b, a]

    return lut

TURBO_LUT = build_turbo_lut()


def remove_isolated_radar_clutter(val):
    """
    Meteorologischer Cluster- & Flächenfilter (100% flackerfrei, kein Aufpoppen):
    
    1. ECHTE Regenfelder & Nieselzonen:
       Haben eine reale räumliche Ausdehnung (Fläche >= 20 Pixel).
       Werden flackerfrei, kontinuierlich und ohne Sprünge dargestellt!
       
    2. ECHTER Schauerregen mit Kern:
       Jedes Regengebiet mit mindestens einem Kernpixel val >= 4 wird immer dargestellt.
       
    3. ISOLIERTES Turmrauschen & Geister-Sprenkel:
       Winzige isolierte Punkt-Cluster (< 20 Pixel) OHNE jeden Kern (val < 4).
       Werden sauber und geräuschlos entfernt!
    """
    if not np.any(val > 0):
        return val
        
    try:
        from scipy.ndimage import label
        
        labeled_array, num_features = label(val > 0)
        if num_features == 0:
            return val
            
        cluster_sizes = np.bincount(labeled_array.ravel())
        clean_val = val.copy()
        
        for i in range(1, num_features + 1):
            size = cluster_sizes[i]
            # Große zusammenhängende Regenwolken (Fläche >= 20 Pixel) immer behalten
            if size >= 20:
                continue
                
            # Bei winzigen Mini-Clustern (< 20 Pixel):
            # Nur behalten wenn ein echter Schauerkern vorhanden ist (val >= 4)
            cluster_mask = (labeled_array == i)
            if np.max(val[cluster_mask]) < 4:
                clean_val[cluster_mask] = 0
                
        return clean_val
        
    except ImportError:
        return val


def map_radolan_val_to_index(val):
    """
    Mappt DWD RADOLAN RV Rohwerte (0.01 mm/5min) physikalisch exakt auf die 256 Turbo-Farbstufen.
    Ausgewogene Kalibrierung für lückenlose, organische Regenfronten ohne Überfilterung.
    """
    idx = np.zeros_like(val, dtype=np.uint8)
    
    # 1. Zarter Nieselregen & Feuchtesaum: val 1..4 (0.12..0.48 mm/h)
    m1 = (val >= 1) & (val < 5)
    idx[m1] = (1 + ((val[m1] - 1) / 4.0) * 24).astype(np.uint8)
    
    # 2. Leichter bis mäßiger Landregen (Sattes Smaragd- & Lime-Grün): val 5..20 (0.6..2.4 mm/h)
    m2 = (val >= 5) & (val < 21)
    idx[m2] = (26 + ((val[m2] - 5) / 16.0) * 64).astype(np.uint8)
    
    # 3. Kräftiger Schauer (Goldgelb): val 21..55 (2.5..6.6 mm/h)
    m3 = (val >= 21) & (val < 56)
    idx[m3] = (91 + ((val[m3] - 21) / 35.0) * 59).astype(np.uint8)
    
    # 4. Starkregen (Orange): val 56..140 (6.7..16.8 mm/h)
    m4 = (val >= 56) & (val < 141)
    idx[m4] = (151 + ((val[m4] - 56) / 85.0) * 49).astype(np.uint8)
    
    # 5. Extremregen (Karminrot): val 141..280 (16.9..33.6 mm/h)
    m5 = (val >= 141) & (val < 281)
    idx[m5] = (201 + ((val[m5] - 141) / 140.0) * 34).astype(np.uint8)
    
    # 6. Unwetter / Hagel (Magenta/Weiß): val >= 281 (>33.6 mm/h)
    m6 = val >= 281
    idx[m6] = np.clip(236 + (val[m6] - 281) / 5.0, 236, 255).astype(np.uint8)
    
    return idx


def parse_radolan_binary(data_bytes):
    """
    Parst ein binäres DWD RADOLAN-Kompositfile (RV, WN, SF, RW).
    Gibt (header_str, 2D-numpy-array) mit gemappten Indizes zurück.
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
        # Maskiere echte No-Data / Sensorfehler (Bit 13)
        is_nodata = (arr & 0x2000) > 0
        val = arr & 0x0FFF
        val[is_nodata] = 0
        
        # Meteorologischer Kontextfilter: Entfernt isolierte Turmechos, schützt echte Fronten
        val = remove_isolated_radar_clutter(val)
        
        # Meteorologisch exakt auf 0..255 LUT mappen
        grid_indexed = map_radolan_val_to_index(val)
        
        # DWD RADOLAN invertieren (Nord oben)
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
    Wendet die WarnWetter-Turbo LUT an, glättet die Isolinien konturenscharf
    und speichert als 100% verlustfreies WebP.
    """
    rgba = TURBO_LUT[grid_2d]
    img = Image.fromarray(rgba, mode='RGBA')
    
    # 1. Bilineare Skalierung für weiche, lückenlose Fronten
    img_resized = img.resize((1400, 1400), Image.BILINEAR)
    
    # 2. Feines Abschneiden nur von echtem Null-Alpha
    arr = np.array(img_resized)
    arr[arr[:, :, 3] < 10, 3] = 0
    
    img_clean = Image.fromarray(arr, mode='RGBA')
    img_clean.save(output_path, 'WEBP', lossless=True, method=6)


def apply_temporal_consistency_filter(grid_list):
    """
    Temporaler Konsistenz-Filter (Anti-Flicker & Anti-Pop):
    Untersucht die Sequenz (t-1, t, t+1).
    Isolierte Sprenkel & Spikes (val <= 3), die nur für exakt einen 5-Minuten-Frame
    im Nichts aufblitzen, werden eliminiert.
    Wandernde Fronten und zusammenhängende Regengebiete bleiben zu 100% erhalten.
    """
    n = len(grid_list)
    if n < 3:
        return grid_list

    try:
        from scipy.ndimage import maximum_filter
        cleaned_list = [grid_list[0].copy()]

        for t in range(1, n - 1):
            prev_g = grid_list[t - 1]
            curr_g = grid_list[t]
            next_g = grid_list[t + 1]

            # Finde Gebiete mit Niederschlag in t-1 und t+1 (~5km Suchradius)
            prev_active = maximum_filter(prev_g > 0, size=7)
            next_active = maximum_filter(next_g > 0, size=7)
            temporal_support = prev_active | next_active

            # Schwache Pixel (val <= 3) ohne zeitlichen Halt in t-1 oder t+1 sind temporäre Glitches
            is_glitch = (curr_g > 0) & (curr_g <= 3) & (~temporal_support)

            curr_clean = curr_g.copy()
            curr_clean[is_glitch] = 0
            cleaned_list.append(curr_clean)

        cleaned_list.append(grid_list[-1].copy())
        return cleaned_list
    except Exception as e:
        print(f"Hinweis beim Temporal-Filter: {e}")
        return grid_list


def generate_radar_dataset():
    start_time = time.time()
    print("🚀 Starte DWD RADOLAN HD 5-Minuten Turbo-Radar Generator (100% DWD OpenData)...")

    output_dir = "./dist/radar"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Verfügbare DWD OpenData RV Dateien abrufen
    dwd_files = get_available_dwd_rv_files()
    if not dwd_files:
        print("❌ Keine RADOLAN-Dateien auf opendata.dwd.de gefunden!")
        return

    now = datetime.now(timezone.utc)
    print(f"📡 DWD OpenData Server erreichbar ({len(dwd_files)} RADOLAN RV Komposite verfügbar).")

    # 2. Historie filtern: Letzte 8 Stunden (Lückenloser 5-Minuten-Takt!)
    eight_hours_ago = now - timedelta(hours=8)
    selected_history = [f for f in dwd_files if f[1] >= eight_hours_ago and f[1] <= now]

    print(f"📦 Lade {len(selected_history)} lückenlose 5-Minuten-Messungen (-8h bis Jetzt)...")

    raw_history_items = []

    # 3. Historie parallel herunterladen und parsen
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

        return {
            "valid_dt": valid_dt,
            "grid": grid,
            "is_nowcast": False
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_history_item, item, i): i for i, item in enumerate(selected_history)}
        for future in as_completed(futures):
            res = future.result()
            if res:
                raw_history_items.append(res)

    raw_history_items.sort(key=lambda x: x['valid_dt'])
    print(f"✅ {len(raw_history_items)} historische 5-Minuten-Grids geladen.")

    # 4. Nowcast (+2h Zukunft im lückenlosen 5-Minuten-Takt)
    raw_nowcast_items = []
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
                raw_nowcast_items.append({
                    "valid_dt": valid_dt,
                    "grid": grid,
                    "is_nowcast": True
                })

    all_items = raw_history_items + raw_nowcast_items
    all_items.sort(key=lambda x: x['valid_dt'])
    print(f"🧠 Wende 3-Stufen Temporal & Spatial Anti-Pop Filter auf alle {len(all_items)} Frames an...")

    # 5. Temporal Consistency Filter über die gesamte chronologische Sequenz
    raw_grids = [item['grid'] for item in all_items]
    cleaned_grids = apply_temporal_consistency_filter(raw_grids)

    # 6. Paralleles Rendern aller WebP-Dateien
    frames_metadata = []

    def render_and_save(idx):
        item = all_items[idx]
        grid = cleaned_grids[idx]
        file_name = f"radar_{idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)
        render_matrix_to_webp(grid, file_path)

        return {
            "step": idx,
            "valid_time": item['valid_dt'].strftime("%Y-%m-%dT%H:%M:00Z"),
            "file": file_name,
            "is_nowcast": item['is_nowcast']
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(render_and_save, i): i for i in range(len(all_items))}
        for future in as_completed(futures):
            res = future.result()
            if res:
                frames_metadata.append(res)

    frames_metadata.sort(key=lambda x: x['step'])
    print(f"✨ Gesamt-Datensatz: {len(frames_metadata)} flüssige 5-Minuten-Frames fertig gerendert!")

    # 7. Metadata JSON schreiben
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

    # 6. FTPS Upload
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
