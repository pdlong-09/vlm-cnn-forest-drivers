"""Yearly Landsat 5/7/8/9 Collection 2 Level-2 composites around a ForestNet event, via Microsoft Planetary Computer.

For year t (loss year + k, k = -2..+4) every scene of that calendar year over the chip is cloud-masked with QA_PIXEL
(clear or water bits only) and reduced to a per-pixel median. The chip is a 3 km x 3 km square at 30 m centred on the
event; summary statistics are computed inside the loss polygon and in a 150 m ring around it.
"""
from __future__ import annotations

import math

import numpy as np
import planetary_computer as pc
import pystac_client
from odc.stac import load
from shapely import contains_xy
from shapely.geometry import box

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel"]
OFFSETS = range(-2, 5)
HALF_M = 1500
RES_DEG = 30 / 111320.0
_cat = None


def catalog():
    global _cat
    if _cat is None:
        _cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    return _cat


def chip_bbox(lat, lon):
    dlat = HALF_M / 110574.0
    dlon = HALF_M / (111320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


MAX_SCENES = 6  # least cloudy scenes per year; enough for a median composite, keeps reads per event bounded
POOL = 16


def year_composite(bbox, year, max_cloud=70):
    items = list(catalog().search(collections=["landsat-c2-l2"], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31",
                                  query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    n_all = len(items)
    if not items:
        return None, 0
    items = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[:MAX_SCENES]
    ds = load(items, bands=BANDS, bbox=bbox, crs="EPSG:4326", resolution=RES_DEG, groupby="solar_day",
              chunks=None, fail_on_error=False, pool=POOL)
    qa = ds["qa_pixel"].values.astype(np.uint16)
    clear = ((qa >> 6) & 1).astype(bool) | ((qa >> 7) & 1).astype(bool)
    clear &= qa != 1  # fill
    sr = np.stack([ds[b].values for b in BANDS[:-1]], axis=1).astype(np.float32) * 0.0000275 - 0.2
    sr[~np.repeat(clear[:, None], 6, axis=1)] = np.nan
    sr[(sr < -0.05) | (sr > 1.2)] = np.nan
    with np.errstate(all="ignore"):
        comp = np.nanmedian(sr, axis=0)
    nobs = np.isfinite(sr[:, 3]).sum(0)
    return {"comp": comp, "nobs": nobs, "lat": ds.latitude.values, "lon": ds.longitude.values,
            "scenes": len(ds.time)}, n_all


def indices(c):
    b, g, r, n, s1, s2 = c
    with np.errstate(all="ignore"):
        return {"ndvi": (n - r) / (n + r), "nbr": (n - s2) / (n + s2), "ndmi": (n - s1) / (n + s1)}


def event_series(row, polygon_lonlat):
    """Composites and summary rows for one event (row of rsdrivers.io.events())."""
    bbox = chip_bbox(row.latitude, row.longitude)
    ring = polygon_lonlat.buffer(150 / 111320.0).difference(polygon_lonlat)
    out, stats = {}, []
    for k in OFFSETS:
        y = int(row.year) + k
        res, nitems = (None, 0) if y > 2025 else year_composite(bbox, y)
        rec = {"eid": row.eid, "k": k, "year": y, "items": nitems}
        if res is not None:
            lon, lat = np.meshgrid(res["lon"], res["lat"])
            m_in = contains_xy(polygon_lonlat, lon, lat)
            m_ring = contains_xy(ring, lon, lat)
            if not m_in.any():  # tiny polygons: nearest pixel to the centroid
                c = polygon_lonlat.centroid
                m_in = (np.abs(lon - c.x) == np.abs(lon - c.x).min()) & (np.abs(lat - c.y) == np.abs(lat - c.y).min())
            idx = indices(res["comp"])
            rec.update(scenes=res["scenes"], valid_in=float(np.isfinite(idx["ndvi"][m_in]).mean()),
                       nobs_in=float(np.median(res["nobs"][m_in])))
            for name, arr in idx.items():
                with np.errstate(all="ignore"):
                    rec[f"{name}_in"] = float(np.nanmedian(arr[m_in])) if np.isfinite(arr[m_in]).any() else np.nan
                    rec[f"{name}_ring"] = float(np.nanmedian(arr[m_ring])) if np.isfinite(arr[m_ring]).any() else np.nan
            out[k] = res
        stats.append(rec)
    return out, stats
