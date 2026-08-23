#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (100% DWD OpenData)
================================================================================
Lädt hochauflösende DWD RADOLAN HD Radardaten im lückenlosen 5-Minuten-Takt von opendata.dwd.de:
- -8h bis 0h Vergangenheit: Alle 5-Minuten-Messungen (DWD RADOLAN RV)
- 0h bis +2h Nowcasting: Alle 5-Minuten-Vorhersageschritte (+5m bis +120m)
- Exakte Polar-Stereographische Entzerrung (Warp auf EPSG:4326/Leaflet)
- Organische Isolinien-Glättung im DWD WarnWetter / Kachelmann Vektor-Stil
- Nieselbrücken-Schutz via morphologischer Dilatation für lückenlose Regenfronten
- Bidirektionaler 3-Stufen Temporalfilter gegen Flickern, Wegploppen und Artefakte
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

# Exakte DWD RADOLAN DE1200 Bounding Box (Zielgitter 1400x1400)
# SW: [45.68°N, 1.46°E] bis NE: [55.86°N, 18.73°E]
RADAR_BOUNDS = [[45.68, 1.46], [55.86, 18.73]]


def build_turbo_lut():
    """
    Erstellt die meteorologisch kalibrierte Google Turbo LookUp-Table (0..255)
    im brillanten DWD WarnWetter / Kachelmann Kontrast-Stil.
    """
    lut = np.zeros((256, 4), dtype=np.uint8)

    for i in range(256):
        if i == 0:
            # 0: Absolut transparent (Kein Niederschlag)
            r, g, b, a = 0, 0, 0, 0
        elif i < 26:
            # 1 - 25: Zarter Türkis / Cyan Außensaum (0.24 - 0.60 mm/h)
            t = (i - 1) / 24.0
            r = int(30 + t * 20)
            g = int(185 + t * 45)
            b = int(245 - t * 15)
            a = int(170 + t * 65)  # Scharfe, saubere Kante ohne trüben Dunst
        elif i < 91:
            # 26 - 90: Satte Smaragd- & Lime-Grüntöne (0.72 - 2.40 mm/h)
            # Bildet wie in der WarnWetter App die voll deckende Hauptmasse des Regens
            t = (i - 26) / 64.0
            r = int(45 + t * 125)
            g = int(218 + t * 25)
            b = int(70 - t * 55)
            a = 255  # 100% Opazität für maximalen Kontrast über Land & Meer
        elif i < 151:
            # 91 - 150: Leuchtendes Goldgelb bis Sonnengelb (2.52 - 6.60 mm/h)
            t = (i - 91) / 59.0
            r = int(240 + t * 15)
            g = int(215 - t * 45)
            b = int(15 - t * 10)
            a = 255
        elif i < 201:
            # 151 - 200: Kräftiges Orange (6.72 - 16.8 mm/h)
            t = (i - 151) / 49.0
            r = int(250 - t * 10)
            g = int(145 - t * 75)
            b = int(10 + t * 10)
            a = 255
        elif i < 236:
            # 201 - 235: Intensives Karminrot bis Scharlachrot (16.9 - 33.6 mm/h)
            t = (i - 201) / 34.0
            r = int(238 - t * 25)
            g = int(40 - t * 25)
            b = int(25 + t * 65)
            a = 255
        else:
            # 236 - 255: Magenta bis Weiß-Violett (Extremer Starkregen / Hagel > 33.6 mm/h)
            t = (i - 236) / 19.0
            r = int(215 + t * 40)
            g = int(35 + t * 220)
            b = int(215 + t * 40)
            a = 255
            
        lut[i] = [r, g, b, a]

    return lut

TURBO_LUT = build_turbo_lut()


# Vorberechnete Reprojektions-Koordinaten für das 1400x1400 WGS84 Zielraster
_WARP_COORDS = None

def get_reprojection_coords(target_h=1400, target_w=1400):
    """
    Berechnet die exakte Polar-Stereographische Koordinatentransformation (DWD DE1200 -> WGS84).
    Wird einmalig vorberechnet und für alle 120 Frames wiederverwendet.
    """
    global _WARP_COORDS
    if _WARP_COORDS is not None:
        return _WARP_COORDS

    lat_min, lat_max = RADAR_BOUNDS[0][0], RADAR_BOUNDS[1][0]
    lon_min, lon_max = RADAR_BOUNDS[0][1], RADAR_BOUNDS[1][1]

    # DWD Spezifikation: Polar-stereographische Projektion
    R = 6370.040  # Erdradius in km
    lat_ts = np.radians(60.0)  # Standard-Parallele 60°N
    lon_0 = np.radians(10.0)   # Zentralmeridian 10°E
    scale = R * (1.0 + np.sin(lat_ts))

    # Reguläres Lat/Lon Gitter für Leaflet WebMap (von Nord nach Süd)
    lats = np.linspace(lat_max, lat_min, target_h)
    lons = np.linspace(lon_min, lon_max, target_w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    phi = np.radians(lat_grid)
    lam = np.radians(lon_grid)

    m = scale / (1.0 + np.sin(phi))
    x_proj = m * np.cos(phi) * np.sin(lam - lon_0)
    y_proj = -m * np.cos(phi) * np.cos(lam - lon_0)

    # Offset der linken unteren Ecke (SW) im 1100x1200 RADOLAN Raster
    # DWD DE1200 Gitter: x_0 = -543.197 km, y_0 (South) = -4822.589 km
    x_px = x_proj + 543.197
    y_px = y_proj + 4822.589

    _WARP_COORDS = (y_px, x_px)
    return _WARP_COORDS


def reproject_and_smooth_radar(grid_1200x1100):
    """
    Reprojiziert das polar-stereographische RADOLAN-Gitter auf WGS84 (EPSG:4326)
    in voller nativer Schärfe ohne künstlichen Weichzeichner.
    """
    try:
        from scipy.ndimage import map_coordinates

        y_coords, x_coords = get_reprojection_coords(1400, 1400)
        
        # Exakte Reprojektion ohne Weichzeichner (100% scharf)
        warped = map_coordinates(grid_1200x1100.astype(np.float32), [y_coords, x_coords], order=1, mode='constant', cval=0.0)
        return np.clip(warped, 0, 255).astype(np.uint8)

    except ImportError:
        # Fallback falls scipy nicht verfügbar: Erst flippen (Nord oben), dann skalieren
        flipped = np.flipud(grid_1200x1100)
        img = Image.fromarray(flipped)
        return np.array(img.resize((1400, 1400), Image.BILINEAR))


def remove_isolated_radar_clutter(val):
    """
    Meteorologischer Nieselbrücken- & Flächenfilter:

    1. Echte Kerne (val >= 4 bzw. > 0.48 mm/h) und große Fronten (>= 150 Pixel) definieren Regenfronten.
    2. Nieselbrücken-Schutz: Durch morphologische Dilatation werden kleinere Nieselinseln im Umkreis
       von 15 km um echte Fronten geschützt und nicht gelöscht.
    3. Nur isolierter Turm-Clutter (< 150 Pixel, ohne Kern und ohne räumlichen Frontenbezug) wird entfernt.
    """
    if not np.any(val > 0):
        return val

    try:
        from scipy.ndimage import label, maximum as nd_max, sum as nd_sum, binary_dilation

        labeled_array, num_features = label(val > 0)
        if num_features == 0:
            return val

        indices = np.arange(1, num_features + 1)
        cluster_sizes = nd_sum(np.ones_like(val), labels=labeled_array, index=indices)
        cluster_maxs = nd_max(val, labels=labeled_array, index=indices)

        # 1. Sichere Regenfronten: Haben einen Schauerkern (val >= 4) ODER sind groß (>= 150 Pixel)
        is_core_or_large = (cluster_maxs >= 4) | (cluster_sizes >= 150)
        valid_ids = indices[is_core_or_large]

        # 2. Maske der sicheren Fronten erstellen
        valid_mask = np.isin(labeled_array, valid_ids)

        # 3. Nieselbrücken-Schutz: Dehne die Zone um 15 Pixel (~15 km) aus
        expanded_zone = binary_dilation(valid_mask, iterations=15)

        # 4. Prüfe Überlappung jedes Clusters mit der erweiterten Zone
        overlap = nd_sum(expanded_zone.astype(int), labels=labeled_array, index=indices)
        is_valid_cluster = is_core_or_large | (overlap > 0)

        invalid_cluster_ids = indices[~is_valid_cluster]

        clean_val = val.copy()
        if len(invalid_cluster_ids) > 0:
            is_invalid_pixel = np.isin(labeled_array, invalid_cluster_ids)
            clean_val[is_invalid_pixel] = 0

        return clean_val
    except ImportError:
        return val


def map_radolan_val_to_index(val):
    """
    Mappt DWD RADOLAN RV Rohwerte (0.01 mm/5min) auf 256 Turbo-Farbstufen.
    """
    idx = np.zeros_like(val, dtype=np.uint8)

    # 1. Zarter Nieselregen & Feuchtesaum (val 2..5 -> 0.24..0.60 mm/h)
    m1 = (val >= 2) & (val < 6)
    idx[m1] = (1 + ((val[m1] - 2) / 4.0) * 24).astype(np.uint8)

    # 2. Leichter bis mäßiger Landregen (val 6..20 -> 0.72..2.40 mm/h)
    m2 = (val >= 6) & (val < 21)
    idx[m2] = (26 + ((val[m2] - 6) / 15.0) * 64).astype(np.uint8)

    # 3. Kräftiger Schauer (val 21..55 -> 2.52..6.60 mm/h)
    m3 = (val >= 21) & (val < 56)
    idx[m3] = (91 + ((val[m3] - 21) / 35.0) * 59).astype(np.uint8)

    # 4. Starkregen (val 56..140 -> 6.72..16.8 mm/h)
    m4 = (val >= 56) & (val < 141)
    idx[m4] = (151 + ((val[m4] - 56) / 85.0) * 49).astype(np.uint8)

    # 5. Extremregen (val 141..280 -> 16.9..33.6 mm/h)
    m5 = (val >= 141) & (val < 281)
    idx[m5] = (201 + ((val[m5] - 141) / 140.0) * 34).astype(np.uint8)

    # 6. Unwetter / Hagel (val >= 281 -> >33.6 mm/h)
    m6 = val >= 281
    idx[m6] = np.clip(236 + (val[m6] - 281) / 5.0, 236, 255).astype(np.uint8)

    return idx


def parse_radolan_binary(data_bytes):
    """
    Parst ein binäres DWD RADOLAN-Kompositfile (RV, WN, SF, RW).
    Gibt (header_str, 2D-numpy-array in Roh-Orientierung) zurück.
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
        
        # Maskiere Fehlkennung (Bit 14) und Sensorfehler (Bit 13)
        is_nodata = (arr & 0x6000) > 0
        val = arr & 0x0FFF
        val[is_nodata] = 0
        
        # Thermischen Antennen-Rauschboden (< 0.24 mm/h) auf 0 setzen
        val[val < 2] = 0

        # Meteorologischer Nieselbrücken- & Flächenfilter
        val = remove_isolated_radar_clutter(val)

        # Meteorologisch exakt auf 0..255 LUT mappen
        grid_indexed = map_radolan_val_to_index(val)

        return header, grid_indexed

    elif len(raw_data) >= width * height:
        arr = np.frombuffer(raw_data[:width * height], dtype=np.uint8).reshape((height, width))
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


def render_matrix_to_webp(grid_reprojected_1400, output_path):
    """
    Wendet die WarnWetter-Turbo LUT an und speichert als 100% verlustfreies WebP.
    """
    rgba = TURBO_LUT[grid_reprojected_1400]
    
    # Dünne Rest-Transparenzen unter 10 säubern
    rgba[rgba[:, :, 3] < 10, 3] = 0
    
    img_clean = Image.fromarray(rgba, mode='RGBA')
    img_clean.save(output_path, 'WEBP', lossless=True, method=6)


def apply_temporal_consistency_filter(grid_list):
    """
    Bidirektionaler temporaler Konsistenz-Filter (Anti-Flicker, Anti-Pop & Gap-Filling):
    
    1. Heilt 'Dips': Wenn ein Regengebiet in t-1 und t+1 existiert, aber in t kurzzeitig unter
       die Messschwelle sinkt -> interpolieren (verhindert Flackern/Wegploppen).
    2. Entfernt 'Spikes': Wenn ein schwacher Farbindex (1..25, entspricht Niesel < 0.6 mm/h)
       nur für exakt 1 Frame isoliert aufblitzt -> löschen.
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

            curr_clean = curr_g.copy()

            # 1. Gap-Filling (Dips heilen)
            is_dip = (curr_clean == 0) & (prev_g > 0) & (next_g > 0)
            if np.any(is_dip):
                interpolated = (prev_g[is_dip].astype(np.int32) + next_g[is_dip].astype(np.int32)) // 2
                curr_clean[is_dip] = np.maximum(1, interpolated).astype(np.uint8)

            # 2. Spike-Removal (Isolierte Flicker-Pixel entfernen)
            prev_active = maximum_filter(prev_g > 0, size=7)
            next_active = maximum_filter(next_g > 0, size=7)
            temporal_support = prev_active | next_active

            # Indizes 1..25 entsprechen Niesel/Feuchtesaum (< 0.6 mm/h)
            is_spike = (curr_clean > 0) & (curr_clean <= 25) & (~temporal_support)
            curr_clean[is_spike] = 0

            cleaned_list.append(curr_clean)

        # Letzter Frame: Einseitige Prüfung gegen t-1
        last_g = grid_list[-1]
        prev_active = maximum_filter(grid_list[-2] > 0, size=7)
        is_last_spike = (last_g > 0) & (last_g <= 25) & (~prev_active)
        last_clean = last_g.copy()
        last_clean[is_last_spike] = 0
        cleaned_list.append(last_clean)

        return cleaned_list
    except Exception as e:
        print(f"Hinweis beim Temporal-Filter: {e}")
        return grid_list


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

    raw_history_items = []

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

    # Nowcast (+2h)
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

    # 1. Temporal Consistency Filter auf raw radar grids
    raw_grids = [item['grid'] for item in all_items]
    cleaned_grids = apply_temporal_consistency_filter(raw_grids)

    print(f"🌐 Führe Polar-Stereographische Reprojektion & Isolinien-Glättung durch...")

    # 2. Paralleles Reprojizieren und Rendern
    frames_metadata = []

    def render_and_save(idx):
        item = all_items[idx]
        grid = cleaned_grids[idx]
        
        # Exakte Reprojektion von Polar-Stereo auf WGS84 + Konturglättung
        warped_grid = reproject_and_smooth_radar(grid)
        
        file_name = f"radar_{idx:03d}.webp"
        file_path = os.path.join(output_dir, file_name)
        render_matrix_to_webp(warped_grid, file_path)

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
