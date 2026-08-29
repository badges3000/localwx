#!/usr/bin/env python3
"""
DWD ICON-EU Synoptik- & Modellkarten Generator (localwx PRO) - CRASH-PROOF & FAST
================================================================================
Zuverlässiger, thread- und prozesssicherer Renderer für DWD ICON-EU Modellkarten.

Features:
- Robuste GRIB2-Dekodierung mit Koordinaten-Normalisierung (-180° bis +180°)
- Automatischer Scan-Richtungsabgleich (Nord-Süd / Süd-Nord)
- Deadlock-freies Multiprocessing pro Zeitschritt
- Automatischer FTPS-Upload nach /data/synoptic/
"""

import os
import sys
import json
import time
import bz2
import zipfile
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image
import ftplib
import ssl

# Matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Cartopy Setup
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

# Eccodes für DWD GRIB2 Dekodierung
try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False


# ==============================================================================
# 1. OFFLINE-DOWNLOAD DER KARTEN-GEOMETRIEN (KEIN NETWORK-HANG / DEADLOCK)
# ==============================================================================

def setup_cartopy_offline():
    """
    Lädt Küsten und Grenzen einmalig vorab herunter und entpackt sie im Cartopy-Cache.
    Verhindert Lock-Konflikte und Endlos-Hänger in Subprozessen.
    """
    if not CARTOPY_AVAILABLE:
        return

    data_dir = cartopy.config.get('data_dir', os.path.expanduser('~/.local/share/cartopy'))
    target_dir = os.path.join(data_dir, 'shapefiles', 'natural_earth', 'physical')
    target_cultural = os.path.join(data_dir, 'shapefiles', 'natural_earth', 'cultural')
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_cultural, exist_ok=True)

    downloads = [
        ("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip", target_dir, "ne_50m_coastline.shp"),
        ("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_boundary_lines_land.zip", target_cultural, "ne_50m_admin_0_boundary_lines_land.shp"),
        ("https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_lakes.zip", target_dir, "ne_50m_lakes.shp")
    ]

    print("🗺️ Initialisiere Geometrie-Features...")
    for url, out_folder, check_file in downloads:
        if os.path.exists(os.path.join(out_folder, check_file)):
            continue
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            zip_path = os.path.join(out_folder, "temp.zip")
            with urllib.request.urlopen(req, timeout=15) as resp, open(zip_path, 'wb') as f:
                f.write(resp.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(out_folder)
            os.remove(zip_path)
        except Exception as e:
            print(f"⚠️ Hinweis bei {check_file}: {e} (Fallback aktiv)")


# ==============================================================================
# FARBSKALEN
# ==============================================================================

def get_localwx_temp_cmap():
    colors = [
        (0.00, '#6b21a8'), (0.08, '#9333ea'), (0.15, '#c084fc'), (0.23, '#1d4ed8'),
        (0.31, '#2563eb'), (0.38, '#38bdf8'), (0.44, '#7dd3fc'), (0.46, '#e0f2fe'),
        (0.51, '#86efac'), (0.58, '#22c55e'), (0.65, '#eab308'), (0.73, '#f97316'),
        (0.81, '#ef4444'), (0.88, '#dc2626'), (0.95, '#991b1b'), (1.00, '#ec4899')
    ]
    return mcolors.LinearSegmentedColormap.from_list('localwx_temp', [(pos, col) for pos, col in colors], N=256)

def get_localwx_jet_cmap():
    colors = [
        (0.00, '#0f172a'), (0.15, '#0369a1'), (0.30, '#0d9488'), (0.45, '#16a34a'),
        (0.60, '#eab308'), (0.75, '#ea580c'), (0.88, '#dc2626'), (1.00, '#d946ef')
    ]
    return mcolors.LinearSegmentedColormap.from_list('localwx_jet', [(pos, col) for pos, col in colors], N=256)

TEMP_CMAP = get_localwx_temp_cmap()
JET_CMAP = get_localwx_jet_cmap()


# ==============================================================================
# DWD GRIB2 DOWNLOAD & PARSING
# ==============================================================================

def get_latest_icon_eu_run():
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


def fetch_dwd_file(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (localwx PRO)'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                compressed = resp.read()
                return bz2.decompress(compressed)
        except Exception:
            time.sleep(0.5)
    return None


def parse_grib_array(grib_bytes):
    """
    Parst GRIB2-Nachrichten und normalisiert Koordinaten sowie Scanrichtung.
    """
    if not ECCODES_AVAILABLE or grib_bytes is None:
        return None, None, None
    msg_id = None
    try:
        msg_id = eccodes.codes_new_from_message(grib_bytes)
        ni = eccodes.codes_get(msg_id, 'Ni')
        nj = eccodes.codes_get(msg_id, 'Nj')

        try:
            lat_first = eccodes.codes_get_double(msg_id, 'latitudeOfFirstGridPointInDegrees')
            lat_last = eccodes.codes_get_double(msg_id, 'latitudeOfLastGridPointInDegrees')
            lon_first = eccodes.codes_get_double(msg_id, 'longitudeOfFirstGridPointInDegrees')
            lon_last = eccodes.codes_get_double(msg_id, 'longitudeOfLastGridPointInDegrees')
        except Exception:
            lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint') / 1e6
            lat_last = eccodes.codes_get(msg_id, 'latitudeOfLastGridPoint') / 1e6
            lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint') / 1e6
            lon_last = eccodes.codes_get(msg_id, 'longitudeOfLastGridPoint') / 1e6

        # Längengrade von 0..360 auf -180..+180 normalisieren
        if lon_first > 180.0:
            lon_first -= 360.0
        if lon_last > 180.0:
            lon_last -= 360.0

        j_scans_pos = eccodes.codes_get(msg_id, 'jScansPositively')
        values = eccodes.codes_get_values(msg_id)
        arr = values.reshape((nj, ni))

        lats = np.linspace(min(lat_first, lat_last), max(lat_first, lat_last), nj)
        lons = np.linspace(min(lon_first, lon_last), max(lon_first, lon_last), ni)

        # Wenn von Nord nach Süd gescannt wurde, Array umdrehen
        if j_scans_pos == 0:
            arr = np.flipud(arr)

        return arr, lats, lons
    except Exception:
        return None, None, None
    finally:
        if msg_id is not None:
            try:
                eccodes.codes_release(msg_id)
            except Exception:
                pass


# ==============================================================================
# KARTEN-RENDERING EINES KOMPLETTEN ZEITSCHRITTS (TASK)
# ==============================================================================

def render_step_task(task_args):
    """
    Rendert alle 8 Karten (4 Parameter x 2 Domains) eines Zeitschritts autark im Worker-Prozess.
    """
    lead_h, date_str, run_str, run_date_iso, base_url, output_base = task_args
    run_date = datetime.fromisoformat(run_date_iso)
    valid_date = run_date + timedelta(hours=lead_h)
    lead_str = f"{lead_h:03d}"

    url_map = {
        't850': f"{base_url}/t/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_850_T.grib2.bz2",
        'fi850': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_850_FI.grib2.bz2",
        'fi500': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_500_FI.grib2.bz2",
        'fi300': f"{base_url}/fi/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_FI.grib2.bz2",
        'pmsl': f"{base_url}/pmsl/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_PMSL.grib2.bz2",
        'u300': f"{base_url}/u/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_U.grib2.bz2",
        'v300': f"{base_url}/v/icon-eu_europe_regular-lat-lon_pressure-level_{date_str}{run_str}_{lead_str}_300_V.grib2.bz2",
        't2m': f"{base_url}/t_2m/icon-eu_europe_regular-lat-lon_single-level_{date_str}{run_str}_{lead_str}_T_2M.grib2.bz2"
    }

    raw_bytes = {}
    with ThreadPoolExecutor(max_workers=8) as dl_exec:
        fut_map = {dl_exec.submit(fetch_dwd_file, url): key for key, url in url_map.items()}
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            raw_bytes[key] = fut.result()

    t850_arr, lats, lons = parse_grib_array(raw_bytes.get('t850'))
    fi850_arr, _, _ = parse_grib_array(raw_bytes.get('fi850'))
    fi500_arr, _, _ = parse_grib_array(raw_bytes.get('fi500'))
    fi300_arr, _, _ = parse_grib_array(raw_bytes.get('fi300'))
    pmsl_arr, _, _ = parse_grib_array(raw_bytes.get('pmsl'))
    u300_arr, _, _ = parse_grib_array(raw_bytes.get('u300'))
    v300_arr, _, _ = parse_grib_array(raw_bytes.get('v300'))
    t2m_arr, _, _ = parse_grib_array(raw_bytes.get('t2m'))

    if lats is None or lons is None:
        return lead_h, []

    data = {
        't850': (t850_arr - 273.15) if t850_arr is not None else None,
        'fi850': (fi850_arr / 98.0665) if fi850_arr is not None else None,
        'fi500': (fi500_arr / 98.0665) if fi500_arr is not None else None,
        'fi300': (fi300_arr / 98.0665) if fi300_arr is not None else None,
        'pmsl': (pmsl_arr / 100.0) if pmsl_arr is not None else None,
        'jet300': (np.sqrt(u300_arr**2 + v300_arr**2) * 3.6) if (u300_arr is not None and v300_arr is not None) else None,
        't2m': (t2m_arr - 273.15) if t2m_arr is not None else None
    }

    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=50.0) if CARTOPY_AVAILABLE else None

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    generated_frames = []

    for p in params:
        if p == 't850_gp':
            field_val, geopot_gpdm = data['t850'], data['fi850']
            title_str, unit_str, cmap = "850 hPa Geopotential (gpdm) & Temperatur (°C)", "°C", TEMP_CMAP
            levels, contour_levels, contour_fmt = np.arange(-36, 43, 2), np.arange(120, 175, 5), "%d"
        elif p == 'z500_mslp':
            field_val, geopot_gpdm = data['fi500'], data['pmsl']
            title_str, unit_str, cmap = "500 hPa Geopotentialhöhe (gpdm) & Bodendruck (hPa)", "gpdm", TEMP_CMAP
            levels, contour_levels, contour_fmt = np.arange(496, 604, 4), np.arange(960, 1060, 5), "%d"
        elif p == 'jet300':
            field_val, geopot_gpdm = data['jet300'], data['fi300']
            title_str, unit_str, cmap = "300 hPa Wind & Jetstream (km/h) & 300 hPa Isohypsen", "km/h", JET_CMAP
            levels, contour_levels, contour_fmt = np.arange(40, 330, 20), np.arange(840, 980, 8), "%d"
        else:
            field_val, geopot_gpdm = data['t2m'], data['pmsl']
            title_str, unit_str, cmap = "2m Temperatur (°C) & Bodendruck (hPa)", "°C", TEMP_CMAP
            levels, contour_levels, contour_fmt = np.arange(-30, 45, 2), np.arange(970, 1050, 5), "%d"

        for dom in domains:
            fig = plt.figure(figsize=(13.33, 8.33), dpi=96, facecolor='#0b0f19')

            if CARTOPY_AVAILABLE:
                ax = fig.add_axes([0.02, 0.08, 0.96, 0.83], projection=proj)
                if dom == 'germany':
                    ax.set_extent([4.5, 16.5, 46.5, 55.8], crs=ccrs.PlateCarree())
                else:
                    ax.set_extent([-16.0, 36.0, 34.0, 68.0], crs=ccrs.PlateCarree())

                cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
                
                try:
                    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#0f172a', linewidth=0.9, zorder=3)
                    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#334155', linewidth=0.6, linestyle=':', zorder=3)
                    ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='#0f172a', facecolor='none', linewidth=0.5, zorder=3)
                except Exception:
                    ax.coastlines(resolution='110m', color='#0f172a', linewidth=0.8)

                if geopot_gpdm is not None:
                    cs = ax.contour(lon_mesh, lat_mesh, geopot_gpdm, levels=contour_levels, colors='#ffffff', linewidths=1.3, zorder=4, transform=ccrs.PlateCarree())
                    ax.clabel(cs, inline=True, fmt=contour_fmt, fontsize=9, colors='#ffffff', inline_spacing=8)

                ax.gridlines(draw_labels=False, linewidth=0.4, color='#ffffff', alpha=0.2, linestyle='--')
            else:
                ax = fig.add_axes([0.05, 0.10, 0.90, 0.80], facecolor='#0b0f19')
                cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=levels, cmap=cmap, extend='both')
                if geopot_gpdm is not None:
                    cs = ax.contour(lon_mesh, lat_mesh, geopot_gpdm, levels=contour_levels, colors='#ffffff', linewidths=1.2)
                    ax.clabel(cs, inline=True, fmt=contour_fmt, fontsize=9, colors='#ffffff')

            # Titel & Footer
            init_str = run_date.strftime("%a, %d. %b %H:00 UTC")
            valid_str = valid_date.strftime("%a, %d. %b %H:00 UTC")
            fig.text(0.03, 0.96, f"localwx PRO  •  DWD ICON-EU  •  {title_str}", color='#ffffff', fontsize=14, fontweight='bold')
            fig.text(0.03, 0.925, f"Modell-Lauf: {init_str}   |   Gültig: {valid_str} (+{lead_h:02d}h)", color='#94a3b8', fontsize=11)

            cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.025])
            cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=9, colors='#cbd5e1')
            cbar.set_label(f"Skala ({unit_str})", color='#cbd5e1', fontsize=10, labelpad=-35, x=-0.08)

            filename = f"frame_{lead_h:03d}.webp"
            filepath = os.path.join(output_base, p, dom, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.savefig(filepath, format='webp', dpi=96, facecolor='#0b0f19', edgecolor='none')
            plt.close(fig)

            generated_frames.append({
                "param": p,
                "domain": dom,
                "lead_h": lead_h,
                "file": f"{p}/{dom}/{filename}",
                "valid_time": valid_date.isoformat(),
                "valid_label": valid_date.strftime("%a %d.%m. %H:%M")
            })

    return lead_h, generated_frames


# ==============================================================================
# PIPELINE GENERATOR
# ==============================================================================

def generate_synoptic_dataset():
    start_time = time.time()
    print("🚀 Starte DWD ICON Synoptik- & Modellkarten Generator (localwx PRO)...")

    setup_cartopy_offline()

    output_base = "./dist/synoptic"
    os.makedirs(output_base, exist_ok=True)

    run_date = get_latest_icon_eu_run()
    date_str = run_date.strftime("%Y%m%d")
    run_str = f"{run_date.hour:02d}"
    print(f"📡 Verwende Modell-Lauf: {run_date.strftime('%Y-%m-%d %H:00 UTC')}")

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    steps = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 72, 84, 96, 120]
    print(f"📦 Verarbeite {len(steps)} Zeitschritte (0h bis +120h) mit {len(steps)*8} Karten...")

    manifest = {
        "model": "DWD ICON-EU",
        "init_time": run_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step_interval_h": 3,
        "max_lead_h": 120,
        "parameters": params,
        "domains": domains,
        "steps": steps,
        "frames": {p: {dom: [] for dom in domains} for p in params}
    }

    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{run_str}"

    task_args_list = [
        (lead_h, date_str, run_str, run_date.isoformat(), base_url, output_base)
        for lead_h in steps
    ]

    max_procs = min(os.cpu_count() or 4, 4)
    print(f"⚡ Starte Multiprocess-Rendering mit {max_procs} Workern...")

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = {executor.submit(render_step_task, args): args[0] for args in task_args_list}
        completed = 0
        for future in as_completed(futures):
            lead_h = futures[future]
            try:
                _, frames = future.result()
                for f in frames:
                    manifest["frames"][f["param"]][f["domain"]].append(f)
                completed += 1
                print(f"   ↳ [{completed}/{len(steps)}] Zeitschritt +{lead_h:02d}h fertiggestellt.")
            except Exception as e:
                print(f"⚠️ Fehler bei Zeitschritt +{lead_h}h: {e}")

    for p in params:
        for dom in domains:
            manifest["frames"][p][dom].sort(key=lambda x: x['lead_h'])

    manifest_path = os.path.join(output_base, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✅ Alle 168 synoptischen Modellkarten in {duration}s erfolgreich generiert!")

    upload_synoptic_to_ftp(output_base)


def upload_synoptic_to_ftp(local_dir):
    server = os.environ.get('FTP_SERVER')
    user = os.environ.get('FTP_USERNAME')
    password = os.environ.get('FTP_PASSWORD')

    if not server or not user or not password:
        print("ℹ️ Keine FTP-Zugangsdaten. Überspringe Upload.")
        return

    print("\n📡 Verbinde mit FTP-Server für 'data/synoptic/'...")
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
    for folder in ["data", "synoptic"]:
        try:
            ftp.cwd(folder)
        except ftplib.error_perm:
            try:
                ftp.mkd(folder)
                ftp.cwd(folder)
            except Exception:
                pass

    uploaded_count = 0
    for root, dirs, files in os.walk(local_dir):
        rel_dir = os.path.relpath(root, local_dir)
        if rel_dir != ".":
            for part in rel_dir.split(os.sep):
                try:
                    ftp.mkd(part)
                except Exception:
                    pass
                try:
                    ftp.cwd(part)
                except Exception:
                    pass

        for file in files:
            if file.endswith('.webp') or file.endswith('.json'):
                full_path = os.path.join(root, file)
                with open(full_path, 'rb') as f_in:
                    ftp.storbinary(f"STOR {file}", f_in)
                    uploaded_count += 1

        if rel_dir != ".":
            depth = len(rel_dir.split(os.sep))
            for _ in range(depth):
                ftp.cwd('..')

    ftp.quit()
    print(f"✅ {uploaded_count} synoptische Modellkarten erfolgreich nach data/synoptic/ hochgeladen!")


if __name__ == "__main__":
    generate_synoptic_dataset()
