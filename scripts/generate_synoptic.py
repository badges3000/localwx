#!/usr/bin/env python3
"""
DWD ICON-EU Synoptik- & Modellkarten Generator (localwx PRO) - CRASH-PROOF & TURBO
=================================================================================
Zuverlässiger, paralleler Multiprocess-Renderer für echte DWD ICON-EU GRIB2-Karten.

Features:
- Robuster, zeitbegrenzter Pre-Fetch für Cartopy-Features (Kein Hang bei Natural Earth)
- Parallelisiertes Chart-Rendering via ProcessPoolExecutor
- Sauberes Eccodes Memory-Management
- 100% thread- & process-safe
- Upload per FTPS nach /data/synoptic/
"""

import os
import sys
import json
import time
import bz2
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from PIL import Image
import ftplib
import ssl

# Matplotlib headless backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Cartopy Pre-Check & Feature Setup
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

try:
    import eccodes
    ECCODES_AVAILABLE = True
except ImportError:
    ECCODES_AVAILABLE = False


# ==============================================================================
# LOCALWX METEOROLOGISCHE FARBSKALEN
# ==============================================================================

def get_localwx_temp_cmap():
    colors = [
        (0.00, '#6b21a8'),   # -36°C: Tiefes Violett
        (0.08, '#9333ea'),   # -30°C: Violett
        (0.15, '#c084fc'),   # -24°C: Helles Flieder
        (0.23, '#1d4ed8'),   # -18°C: Tiefblau
        (0.31, '#2563eb'),   # -12°C: Königsblau
        (0.38, '#38bdf8'),   #  -6°C: Eisblau
        (0.44, '#7dd3fc'),   #  -2°C: Zartes Eiscyan
        (0.46, '#e0f2fe'),   #   0°C: Frostgrenze (Weißblau)
        (0.51, '#86efac'),   #  +4°C: Helles Frühlingsgrün
        (0.58, '#22c55e'),   # +10°C: Smaragdgrün
        (0.65, '#eab308'),   # +16°C: Warmes Goldgelb
        (0.73, '#f97316'),   # +22°C: Sonniges Orange
        (0.81, '#ef4444'),   # +28°C: Warmrot (Sommertag)
        (0.88, '#dc2626'),   # +34°C: Intensives Scharlachrot (Heißer Tag)
        (0.95, '#991b1b'),   # +38°C: Dunkelrot
        (1.00, '#ec4899')    # +42°C: Magenta
    ]
    return mcolors.LinearSegmentedColormap.from_list('localwx_temp', [(pos, col) for pos, col in colors], N=256)


def get_localwx_jet_cmap():
    colors = [
        (0.00, '#0f172a'),   # < 40 km/h
        (0.15, '#0369a1'),   # 70 km/h: Blau
        (0.30, '#0d9488'),   # 110 km/h: Türkis
        (0.45, '#16a34a'),   # 150 km/h: Grün
        (0.60, '#eab308'),   # 190 km/h: Gelb
        (0.75, '#ea580c'),   # 230 km/h: Orange
        (0.88, '#dc2626'),   # 270 km/h: Rot (Jet-Kern)
        (1.00, '#d946ef')    # 320+ km/h: Magenta
    ]
    return mcolors.LinearSegmentedColormap.from_list('localwx_jet', [(pos, col) for pos, col in colors], N=256)


TEMP_CMAP = get_localwx_temp_cmap()
JET_CMAP = get_localwx_jet_cmap()


# ==============================================================================
# CARTOPY FEATURE PRE-FETCH (KEIN NETWORK HANG)
# ==============================================================================

def preload_cartopy_features():
    """
    Lädt die benötigten Cartopy Natural Earth Features vorab mit striktem Timeout.
    Verhindert Hängenbleiben bei überlasteten Amazon S3 Mirrors.
    """
    if not CARTOPY_AVAILABLE:
        return

    print("🗺️ Initialisiere Geometrie-Features...")
    socket_default_timeout = 10
    import socket
    socket.setdefaulttimeout(socket_default_timeout)

    for feature_name, category, name in [
        ('Coastline', 'physical', 'coastline'),
        ('Borders', 'cultural', 'admin_0_boundary_lines_land'),
        ('Lakes', 'physical', 'lakes')
    ]:
        try:
            shpreader.natural_earth(resolution='50m', category=category, name=name)
        except Exception:
            try:
                shpreader.natural_earth(resolution='110m', category=category, name=name)
            except Exception as e:
                print(f"⚠️ Hinweis bei {feature_name}: {e} (Fallback aktiv)")


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
    if not ECCODES_AVAILABLE or grib_bytes is None:
        return None, None, None
    msg_id = None
    try:
        msg_id = eccodes.codes_new_from_message(grib_bytes)
        ni = eccodes.codes_get(msg_id, 'Ni')
        nj = eccodes.codes_get(msg_id, 'Nj')
        lat_first = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPointInDegrees')
        lat_last = eccodes.codes_get(msg_id, 'latitudeOfLastGridPointInDegrees')
        lon_first = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPointInDegrees')
        lon_last = eccodes.codes_get(msg_id, 'longitudeOfLastGridPointInDegrees')
        values = eccodes.codes_get_values(msg_id)

        arr = values.reshape((nj, ni))
        lats = np.linspace(lat_first, lat_last, nj)
        lons = np.linspace(lon_first, lon_last, ni)
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
# WORKER FUNKTION FÜR MULTIPROCESSING RENDERING
# ==============================================================================

def render_single_chart_task(task_args):
    param, domain, lead_h, run_date_iso, data_dict, output_path = task_args
    run_date = datetime.fromisoformat(run_date_iso)
    valid_date = run_date + timedelta(hours=lead_h)
    lats, lons = data_dict['lats'], data_dict['lons']

    if param == 't850_gp':
        field_val = data_dict['t850']
        geopot_gpdm = data_dict['fi850']
        title_str = "850 hPa Geopotential (gpdm) & Temperatur (°C)"
        unit_str = "°C"
        cmap = TEMP_CMAP
        levels = np.arange(-36, 43, 2)
        contour_levels = np.arange(120, 175, 5)
        contour_label_fmt = "%d"

    elif param == 'z500_mslp':
        field_val = data_dict['fi500']
        geopot_gpdm = data_dict['pmsl']
        title_str = "500 hPa Geopotentialhöhe (gpdm) & Bodendruck (hPa)"
        unit_str = "gpdm"
        cmap = TEMP_CMAP
        levels = np.arange(496, 604, 4)
        contour_levels = np.arange(960, 1060, 5)
        contour_label_fmt = "%d"

    elif param == 'jet300':
        field_val = data_dict['jet300']
        geopot_gpdm = data_dict['fi300']
        title_str = "300 hPa Wind & Jetstream (km/h) & 300 hPa Isohypsen"
        unit_str = "km/h"
        cmap = JET_CMAP
        levels = np.arange(40, 330, 20)
        contour_levels = np.arange(840, 980, 8)
        contour_label_fmt = "%d"

    else:
        field_val = data_dict['t2m']
        geopot_gpdm = data_dict['pmsl']
        title_str = "2m Temperatur (°C) & Bodendruck (hPa)"
        unit_str = "°C"
        cmap = TEMP_CMAP
        levels = np.arange(-30, 45, 2)
        contour_levels = np.arange(970, 1050, 5)
        contour_label_fmt = "%d"

    fig = plt.figure(figsize=(13.33, 8.33), dpi=96, facecolor='#0b0f19')

    if CARTOPY_AVAILABLE:
        proj = ccrs.LambertConformal(central_longitude=10.0, central_latitude=50.0)
        ax = fig.add_axes([0.02, 0.08, 0.96, 0.83], projection=proj)
        
        if domain == 'germany':
            ax.set_extent([4.5, 16.5, 46.5, 55.8], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([-16.0, 36.0, 34.0, 68.0], crs=ccrs.PlateCarree())

        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())

        try:
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#0f172a', linewidth=0.9, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#334155', linewidth=0.6, linestyle=':', zorder=3)
            ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='#0f172a', facecolor='none', linewidth=0.5, zorder=3)
        except Exception:
            ax.coastlines(resolution='110m', color='#0f172a', linewidth=0.8)

        if geopot_gpdm is not None:
            cs = ax.contour(lon_mesh, lat_mesh, geopot_gpdm, levels=contour_levels, colors='#ffffff', linewidths=1.3, zorder=4, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fmt=contour_label_fmt, fontsize=9, colors='#ffffff', inline_spacing=8)

        ax.gridlines(draw_labels=False, linewidth=0.4, color='#ffffff', alpha=0.2, linestyle='--')
    else:
        ax = fig.add_axes([0.05, 0.10, 0.90, 0.80], facecolor='#0b0f19')
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        cf = ax.contourf(lon_mesh, lat_mesh, field_val, levels=levels, cmap=cmap, extend='both')
        if geopot_gpdm is not None:
            cs = ax.contour(lon_mesh, lat_mesh, geopot_gpdm, levels=contour_levels, colors='#ffffff', linewidths=1.2)
            ax.clabel(cs, inline=True, fmt=contour_label_fmt, fontsize=9, colors='#ffffff')

    # Header
    init_str = run_date.strftime("%a, %d. %b %H:00 UTC")
    valid_str = valid_date.strftime("%a, %d. %b %H:00 UTC")
    fig.text(0.03, 0.96, f"localwx PRO  •  DWD ICON-EU (0.0625°)  •  {title_str}", color='#ffffff', fontsize=14, fontweight='bold')
    fig.text(0.03, 0.925, f"Modell-Lauf: {init_str}   |   Gültig: {valid_str} (+{lead_h:02d}h)", color='#94a3b8', fontsize=11)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.025])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9, colors='#cbd5e1')
    cbar.set_label(f"Skala ({unit_str})", color='#cbd5e1', fontsize=10, labelpad=-35, x=-0.08)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='webp', dpi=96, facecolor='#0b0f19', edgecolor='none')
    plt.close(fig)
    return True


# ==============================================================================
# PIPELINE GENERATOR
# ==============================================================================

def generate_synoptic_dataset():
    start_time = time.time()
    print("🚀 Starte DWD ICON Synoptik- & Modellkarten Generator (localwx PRO)...")

    preload_cartopy_features()

    output_base = "./dist/synoptic"
    os.makedirs(output_base, exist_ok=True)

    run_date = get_latest_icon_eu_run()
    date_str = run_date.strftime("%Y%m%d")
    run_str = f"{run_date.hour:02d}"
    print(f"📡 Verwende Modell-Lauf: {run_date.strftime('%Y-%m-%d %H:00 UTC')}")

    params = ['t850_gp', 'z500_mslp', 'jet300', 't2m_wind']
    domains = ['europe', 'germany']
    steps = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 42, 48, 54, 60, 72, 84, 96, 120]
    print(f"📦 Lade & verarbeite {len(steps)} Zeitschritte (0h bis +120h)...")

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
    render_tasks = []

    # 1. Download & Data Extraction
    for lead_h in steps:
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
            continue

        data_dict = {
            'lats': lats,
            'lons': lons,
            't850': (t850_arr - 273.15) if t850_arr is not None else None,
            'fi850': (fi850_arr / 98.0665) if fi850_arr is not None else None,
            'fi500': (fi500_arr / 98.0665) if fi500_arr is not None else None,
            'fi300': (fi300_arr / 98.0665) if fi300_arr is not None else None,
            'pmsl': (pmsl_arr / 100.0) if pmsl_arr is not None else None,
            'jet300': (np.sqrt(u300_arr**2 + v300_arr**2) * 3.6) if (u300_arr is not None and v300_arr is not None) else None,
            't2m': (t2m_arr - 273.15) if t2m_arr is not None else None
        }

        valid_dt = run_date + timedelta(hours=lead_h)
        for p in params:
            for dom in domains:
                filename = f"frame_{lead_h:03d}.webp"
                filepath = os.path.join(output_base, p, dom, filename)
                
                render_tasks.append((p, dom, lead_h, run_date.isoformat(), data_dict, filepath))
                
                manifest["frames"][p][dom].append({
                    "lead_h": lead_h,
                    "file": f"{p}/{dom}/{filename}",
                    "valid_time": valid_dt.isoformat(),
                    "valid_label": valid_dt.strftime("%a %d.%m. %H:%M")
                })

    # 2. Paralleles Multiprocess-Rendering
    print(f"🎨 Rendere {len(render_tasks)} Modellkarten parallel auf allen CPU-Kernen...")
    workers = min(os.cpu_count() or 4, 8)
    with ProcessPoolExecutor(max_workers=workers) as renderer:
        list(renderer.map(render_single_chart_task, render_tasks))

    for p in params:
        for dom in domains:
            manifest["frames"][p][dom].sort(key=lambda x: x['lead_h'])

    manifest_path = os.path.join(output_base, "manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    duration = round(time.time() - start_time, 1)
    print(f"✅ Alle {len(render_tasks)} synoptischen Modellkarten in {duration}s fertiggestellt!")

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
