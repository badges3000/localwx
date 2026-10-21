#!/usr/bin/env python3
"""
DWD RADOLAN HD Turbo-Niederschlagsradar Generator & Uploader (100% DWD OpenData)
================================================================================
Lädt hochauflösende DWD RADOLAN HD Radardaten im lückenlosen 5-Minuten-Takt von opendata.dwd.de:
- -8h bis 0h Vergangenheit: Alle 5-Minuten-Messungen (DWD RADOLAN RV)
- 0h bis +2h Nowcasting: Alle 5-Minuten-Vorhersageschritte (+5m bis +120m)
- Automatischer Aktualitäts-Check zwischen ODIM_H5 (.tar) und RADOLAN Binary (.tar.bz2)
- Exakte Polar-Stereographische Entzerrung (Warp auf EPSG:4326/Leaflet)
- Organische Isolinien-Glättung im DWD WarnWetter / Kachelmann Vektor-Stil
- Lückenloser Durchzug über Radartürme (inkl. Totkegel-Heilung / Cone-of-Silence Inpainting)
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

try:
    import h5py
except ImportError:
    h5py = None

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
            # 1 - 25: Leuchtender Türkis / Cyan Nieselsaum (0.12 - 0.60 mm/h)
            t = (i - 1) / 24.0
            r = int(15 + t * 25)
            g = int(205 + t * 25)
            b = int(255 - t * 15)
            a = int(210 + t * 45)
        elif i < 91:
            # 26 - 90: Satte Smaragd- & Lime-Grüntöne (0.72 - 2.40 mm/h)
            t = (i - 26) / 64.0
            r = int(45 + t * 125)
            g = int(218 + t * 25)
            b = int(70 - t * 55)
            a = 255
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
    Berechnet die exakte Polar-Stereographische Koordinatentransformation (DWD DE1200 -> Web Mercator EPSG:3857).
    Wird einmalig vorberechnet und für alle Frames wiederverwendet.
    """
    global _WARP_COORDS
    if _WARP_COORDS is not None:
        return _WARP_COORDS

    lat_min, lat_max = RADAR_BOUNDS[0][0], RADAR_BOUNDS[1][0]
    lon_min, lon_max = RADAR_BOUNDS[0][1], RADAR_BOUNDS[1][1]

    R = 6370.040
    lat_ts = np.radians(60.0)
    lon_0 = np.radians(10.0)
    scale = R * (1.0 + np.sin(lat_ts))

    y_merc_max = np.log(np.tan(np.pi / 4.0 + np.radians(lat_max) / 2.0))
    y_merc_min = np.log(np.tan(np.pi / 4.0 + np.radians(lat_min) / 2.0))
    y_merc_grid = np.linspace(y_merc_max, y_merc_min, target_h)
    lats = np.degrees(2.0 * np.arctan(np.exp(y_merc_grid)) - np.pi / 2.0)

    lons = np.linspace(lon_min, lon_max, target_w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    phi = np.radians(lat_grid)
    lam = np.radians(lon_grid)

    m = scale / (1.0 + np.sin(phi))
    x_proj = m * np.cos(phi) * np.sin(lam - lon_0)
    y_proj = -m * np.cos(phi) * np.cos(lam - lon_0)

    x_px = x_proj + 543.462
    y_px = y_proj + 4808.645

    _WARP_COORDS = (y_px, x_px)
    return _WARP_COORDS


def reproject_and_smooth_radar(grid_1200x1100):
    """
    1. Reprojiziert das polar-stereographische RADOLAN-Gitter auf WGS84 (EPSG:4326).
    2. Wendet organische Konturglättung (Isolinien-Filter) an.
    """
    try:
        from scipy.ndimage import map_coordinates, gaussian_filter

        y_coords, x_coords = get_reprojection_coords(1400, 1400)

        warped = map_coordinates(
            grid_1200x1100.astype(np.float32),
            [y_coords, x_coords],
            order=1,
            mode='constant',
            cval=0.0
        )

        smoothed = gaussian_filter(warped, sigma=0.75)

        smoothed_out = np.clip(np.round(smoothed), 0, 255).astype(np.uint8)
        smoothed_out[smoothed < 0.6] = 0
        return smoothed_out

    except ImportError:
        flipped = np.flipud(grid_1200x1100)
        img = Image.fromarray(flipped)
        return np.array(img.resize((1400, 1400), Image.BILINEAR))


# 17 offizielle DWD-Radarstationen mit ihren Standorten
DWD_RADAR_STATIONS = [
    {"id": "ASB", "name": "Aurich", "lat": 53.5640, "lon": 7.2337},
    {"id": "BOO", "name": "Boostedt", "lat": 54.0044, "lon": 10.0469},
    {"id": "DRS", "name": "Dresden", "lat": 51.1245, "lon": 13.7686},
    {"id": "EIS", "name": "Eisberg", "lat": 49.5407, "lon": 9.9045},
    {"id": "ESS", "name": "Essen", "lat": 51.4056, "lon": 6.9671},
    {"id": "FBG", "name": "Feldberg", "lat": 47.8736, "lon": 8.0048},
    {"id": "FLD", "name": "Flechtdorf", "lat": 51.3112, "lon": 8.8020},
    {"id": "HNR", "name": "Hannover", "lat": 52.4601, "lon": 9.6945},
    {"id": "ISN", "name": "Isen", "lat": 48.1747, "lon": 12.1018},
    {"id": "MEM", "name": "Memmingen", "lat": 48.0421, "lon": 10.2192},
    {"id": "NEU", "name": "Neuhaus", "lat": 50.5001, "lon": 11.1350},
    {"id": "NHB", "name": "Neuheilenbach", "lat": 50.1100, "lon": 6.5485},
    {"id": "OFT", "name": "Offenbach", "lat": 50.0611, "lon": 8.7061},
    {"id": "PRO", "name": "Prötzel", "lat": 52.6487, "lon": 13.8582},
    {"id": "ROS", "name": "Rostock", "lat": 54.1759, "lon": 12.0581},
    {"id": "TUR", "name": "Türkheim", "lat": 48.5853, "lon": 9.7828},
    {"id": "UMD", "name": "Ummendorf", "lat": 52.1601, "lon": 11.1761},
]

_TOWER_ZONE_CACHE = {}
_TOWER_PIXEL_COORDS = None


def get_tower_pixel_coords(height=1200, width=1100):
    """Berechnet die exakten (cy, cx) Gitterkoordinaten der 17 DWD-Radartürme."""
    global _TOWER_PIXEL_COORDS
    if _TOWER_PIXEL_COORDS is not None:
        return _TOWER_PIXEL_COORDS

    R = 6370.040
    lat_ts = np.radians(60.0)
    lon_0 = np.radians(10.0)
    scale = R * (1.0 + np.sin(lat_ts))

    coords = []
    for st in DWD_RADAR_STATIONS:
        phi = np.radians(st["lat"])
        lam = np.radians(st["lon"])
        m = scale / (1.0 + np.sin(phi))
        xp = m * np.cos(phi) * np.sin(lam - lon_0)
        yp = -m * np.cos(phi) * np.cos(lam - lon_0)

        cx = int(round(xp + 543.462))
        cy = int(round(yp + 4808.645))
        if 0 <= cx < width and 0 <= cy < height:
            coords.append((cy, cx))

    _TOWER_PIXEL_COORDS = coords
    return coords


def get_radar_tower_zone_mask(height=1200, width=1100, radius_km=6):
    """
    Erstellt eine boolesche Maske (1200x1100) für den unmittelbaren Nahbereich (6 km)
    der 17 DWD-Radartürme.
    """
    cache_key = (height, width, radius_km)
    if cache_key in _TOWER_ZONE_CACHE:
        return _TOWER_ZONE_CACHE[cache_key]

    mask = np.zeros((height, width), dtype=bool)
    y_grid, x_grid = np.ogrid[:height, :width]
    r_sq = radius_km ** 2

    for cy, cx in get_tower_pixel_coords(height, width):
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        mask |= (dist_sq <= r_sq)

    _TOWER_ZONE_CACHE[cache_key] = mask
    return mask


def heal_radar_cone_of_silence(val):
    """
    Schließt das physikalische 'Totkegel-Loch' (Cone of Silence / DWD-Standortfilter)
    direkt über den 17 Radartürmen (z. B. Rostock), wenn ein Regengebiet über den Turm zieht.
    Wenn der Außenring (4 bis 9 km) um einen Turm zu >= 45 % im Regen liegt, werden
    künstliche 0-Pixel im inneren 4-km-Kreis sanft mit dem Median des umgebenden Regens gefüllt.
    """
    h, w = val.shape
    out = val.copy()

    for cy, cx in get_tower_pixel_coords(h, w):
        ymin, ymax = max(0, cy - 9), min(h, cy + 10)
        xmin, xmax = max(0, cx - 9), min(w, cx + 10)

        sub = out[ymin:ymax, xmin:xmax]
        sy, sx = np.ogrid[ymin:ymax, xmin:xmax]
        dist_sq = (sx - cx) ** 2 + (sy - cy) ** 2

        outer_ring = (dist_sq > 16) & (dist_sq <= 81)  # 4 km bis 9 km Ring
        inner_core = dist_sq <= 16                     # 0 km bis 4 km Totkegel

        ring_vals = sub[outer_ring]
        if ring_vals.size > 0:
            active_ring = ring_vals[ring_vals > 0]
            # Zieht gerade ein Regengebiet über den Turm hinweg?
            if active_ring.size >= 0.45 * ring_vals.size:
                fill_val = max(1, int(np.median(active_ring)))
                hole_mask = inner_core & (sub == 0)
                if np.any(hole_mask):
                    sub[hole_mask] = fill_val

    return out


def remove_isolated_radar_clutter(val):
    """
    Meteorologischer Niesel-Erhalt & Radarturm-Filter OHNE Ausstanzen:
    - Es werden NIEMALS Löcher in durchziehende Regengebiete oder Nieselfelder gestanzt.
    - Jedes Niederschlagsfeld wird immer nur als GANZES behalten oder (falls reines Störecho) entfernt.
    - Am Radarturm (6 km Radius) werden nur winzige, komplett isolierte Stör-Flecken (< 35 Pixel,
      val <= 2) gelöscht, sofern im Umkreis von 18 km kein echtes Regengebiet existiert.
    - Zieht ein Regengebiet über einen Radarturm, schließt heal_radar_cone_of_silence()
      sogar das physikalische Totkegel-Loch direkt über der Antenne.
    """
    if not np.any(val > 0):
        return val

    val = val.copy()

    try:
        from scipy.ndimage import label, maximum as nd_max, sum as nd_sum, binary_dilation

        structure_8 = np.ones((3, 3), dtype=bool)
        labeled, num_features = label(val >= 1, structure=structure_8)
        if num_features == 0:
            return val

        idx = np.arange(1, num_features + 1)
        sizes = nd_sum(np.ones_like(val), labels=labeled, index=idx)
        maxs = nd_max(val, labels=labeled, index=idx)

        # Unmittelbarer 6-km-Antennenbereich
        tower_mask_6km = get_radar_tower_zone_mask(val.shape[0], val.shape[1], radius_km=6)
        touches_tower = nd_sum(tower_mask_6km.astype(int), labels=labeled, index=idx) > 0

        # 1. Sichere echte Regengebiete / Nieselfelder:
        # - Jedes zusammenhängende Gebiet >= 35 Pixel (
