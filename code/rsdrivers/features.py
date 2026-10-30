"""Tabular features for step k: yearly index series up to year k (later years masked) and static auxiliary layers.

Static layers come with ForestNet (auxiliary/): SRTM (3 channels), peat flag, 84 NCEP climate values (aggregated
over the five years before the event) and Hansen treecover2000 on the image grid; all predate the event.
EXCLUDED as future information (found 21/10/2026, before any model was scored on test): Hansen gain (2000-2012
gain includes planting after events of 2001-2011) and the OSM distances to the closest street and town (OSM as
downloaded by ForestNet around 2020, so roads built after the loss, e.g. plantation road grids, are included).
They are still computed (for a sensitivity check) but listed in FUTURE and dropped by usable(). Coordinates are
not used as features.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from shapely import contains_xy

from .io import FN, PIX_M, SIZE, load_region

INDICES = ["ndvi", "nbr", "ndmi"]
YEARS = range(-2, 5)
LABELS4 = ["Plantation", "Smallholder agriculture", "Grassland shrubland", "Other"]
SPLIT_SEED = 20261021
FUTURE = ["gain_frac", "closest_street_km", "closest_city_km"]


def usable(static: pd.DataFrame) -> pd.DataFrame:
    """Static features allowed at every step k (no information from after the loss year)."""
    return static.drop(columns=[c for c in FUTURE if c in static.columns])


def val_split(ev: pd.DataFrame) -> pd.Series:
    """'train', 'val_sel', 'val_cal' or 'test' per event: val split in halves, stratified by merged_label."""
    rng = np.random.default_rng(SPLIT_SEED)
    part = ev.split.copy()
    for lab, g in ev[ev.split == "val"].groupby("merged_label"):
        idx = g.index.to_numpy()[rng.permutation(len(g))]
        part.loc[idx[: len(idx) // 2]] = "val_sel"
        part.loc[idx[len(idx) // 2:]] = "val_cal"
    return part


def _dist_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def static_features(row) -> dict:
    ex = FN / row.example_path / "auxiliary"
    g = load_region(row.example_path)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    m = contains_xy(g, xs + 0.5, ys + 0.5)
    if not m.any():
        c = g.centroid
        m = (np.abs(xs + 0.5 - c.x) < 1) & (np.abs(ys + 0.5 - c.y) < 1)
    f = {"area_ha": g.area * PIX_M ** 2 / 1e4, "n_parts": len(g.geoms) if hasattr(g, "geoms") else 1}
    srtm = np.load(ex / "srtm.npy").astype(float)
    for i in range(srtm.shape[0]):
        f[f"srtm{i}_mean"] = float(srtm[i][m].mean())
        f[f"srtm{i}_std"] = float(srtm[i][m].std())
    gfc = np.load(ex / "gfc.npy").astype(float)
    f["gain_frac"] = float(gfc[0, 0][m].mean())
    f["tc2000_in"] = float(gfc[1, 0][m].mean())
    f["tc2000_img"] = float(gfc[1, 0].mean())
    f["peat"] = float(bool(json.loads((ex / "peat.json").read_text()).get("peat")))
    ncep = np.load(ex / "ncep.npy").astype(float)
    for i, v in enumerate(ncep):
        f[f"ncep{i}"] = float(v)
    osm = json.loads((ex / "osm.json").read_text(encoding="utf-8"))
    for key in ("closest_street", "closest_city"):
        o = osm.get(key) or {}
        try:
            f[f"{key}_km"] = _dist_km(row.latitude, row.longitude, float(o["lat"]), float(o["lon"]))
        except (KeyError, TypeError, ValueError):
            f[f"{key}_km"] = np.nan
    return f


def series_wide(series: pd.DataFrame) -> pd.DataFrame:
    """series_gee.csv (long) -> one row per event: <index>_<zone>_y<k> and nobs_<zone>_y<k>."""
    s = series.copy()
    s = s.rename(columns={f"{i}_median": i for i in INDICES} | {"nobs_median": "nobs"})
    w = s.pivot_table(index="eid", columns=["zone", "k"], values=INDICES + ["nobs"])
    w.columns = [f"{v}_{z}_y{k}" for v, z, k in w.columns]
    return w


def step_features(wide: pd.DataFrame, k: int) -> pd.DataFrame:
    """Features available at step k: years -2..k kept, later years set to NaN, plus change summaries."""
    out = wide.copy()
    for col in out.columns:
        if int(col.rsplit("_y", 1)[1]) > k:
            out[col] = np.nan
    for i in INDICES:
        for z in ("in", "ring"):
            pre = out[[f"{i}_{z}_y{t}" for t in (-2, -1) if f"{i}_{z}_y{t}" in out]].mean(1)
            post = out[[f"{i}_{z}_y{t}" for t in range(0, k + 1) if f"{i}_{z}_y{t}" in out]]
            out[f"{i}_{z}_drop"] = pre - post.min(1)
            out[f"{i}_{z}_last_minus_pre"] = post.iloc[:, -1] - pre
            out[f"{i}_{z}_recovery"] = post.iloc[:, -1] - post.min(1)
        out[f"{i}_in_minus_ring_last"] = out[f"{i}_in_y{k}"] - out[f"{i}_ring_y{k}"] if f"{i}_in_y{k}" in out else np.nan
    out["k"] = k
    return out
