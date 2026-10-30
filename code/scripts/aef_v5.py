"""design notes: AlphaEarth Foundations embeddings (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL) per event.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/aef_v5.py fetch --set fn|e2|e3 [--workers 4]
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/aef_v5.py pack

One snapshot per event, year a = min(max(Y + 4, 2017), 2025) with Y the event year; mean of the 64 bands at 10 m inside
the loss polygon and in a 150 m ring around it (as the v1 Landsat series); a ring without pixels takes the polygon value.
ForestNet polygons come from region_lonlat, E2 and E3 polygons from polygons.geojson. Reads no label.
fetch -> data/interim/rs_forestnet_v5/aef_<set>.csv (eid, year_a, zone, A00..A63), resumable
pack  -> data/interim/rs_forestnet_v5/aef.npz (eid, feat [n, 128] = polygon then ring, year_a)
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))
from rsdrivers import gee  # noqa: E402
from rsdrivers.io import events, region_lonlat  # noqa: E402
from sims_build import RING_DEG, retry  # noqa: E402

OUT = ROOT / "data/interim/rs_forestnet_v5"
SETS = {"e2": (ROOT / "data/interim/rs_forestnet_sims2", 200000), "e3": (ROOT / "data/interim/rs_forestnet_sims3", 300000)}
BANDS = [f"A{i:02d}" for i in range(64)]


def snapshot_year(y):
    return int(min(max(int(y) + 4, 2017), 2025))


def polygons(which):
    """(eid, event year, polygon in lon/lat) for one set."""
    if which == "fn":
        return [(int(r.eid), int(r.year), region_lonlat(r)) for r in events().itertuples()]
    d, off = SETS[which]
    ev = pd.read_csv(d / "events.csv", usecols=["sid", "year"])
    gj = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((d / "polygons.geojson").read_text())["features"]}
    return [(off + int(r.sid), int(r.year), gj[r.sid]) for r in ev.itertuples()]


def fetch(which, workers):
    ee = gee.init()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"aef_{which}.csv"
    done = set(pd.read_csv(out, usecols=["eid"]).eid) if out.exists() else set()
    jobs = {}
    for eid, y, poly in polygons(which):
        if eid in done:
            continue
        poly = poly.buffer(0)
        ring = poly.buffer(RING_DEG).difference(poly)
        a = snapshot_year(y)
        for zone, g in (("in", poly), ("ring", ring)):
            jobs.setdefault(a, []).append({"id": f"{eid}|{zone}", "geom": mapping(g.simplify(0.00005))})
    tasks = [(a, f[i:i + 100]) for a, f in sorted(jobs.items()) for i in range(0, len(f), 100)]
    print(f"{sum(len(f) for f in jobs.values())} features in {len(tasks)} requests", flush=True)
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    def run(t):
        a, part = t
        img = col.filterDate(f"{a}-01-01", f"{a + 1}-01-01").mosaic().select(BANDS)
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(p["geom"]), {"fid": p["id"]}) for p in part])
        got = retry(lambda: img.reduceRegions(fc, ee.Reducer.mean(), 10, tileScale=4).getInfo(), f"year {a}")
        return [{"eid": f["properties"]["fid"].split("|")[0], "year_a": a, "zone": f["properties"]["fid"].split("|")[1],
                 **{b: f["properties"].get(b) for b in BANDS}} for f in got["features"]]
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(workers) as pool:
        w = csv.DictWriter(f, fieldnames=["eid", "year_a", "zone"] + BANDS)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.map(run, tasks), 1):
            w.writerows(rows)
            f.flush()
            if n % 10 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)}", flush=True)


def pack():
    eids, feats, years, n_ring_fill, n_missing = [], [], [], 0, 0
    for which in ("fn", "e2", "e3"):
        if not (OUT / f"aef_{which}.csv").exists():  # E3 comes later than the training batch; its rows are added then
            print(f"no aef_{which}.csv yet")
            continue
        t = pd.read_csv(OUT / f"aef_{which}.csv").drop_duplicates(["eid", "zone"], keep="last")
        want = [e for e, _, _ in polygons(which)]
        by = {z: g.set_index("eid") for z, g in t.groupby("zone")}
        for e in want:
            v_in = by["in"].loc[e, BANDS].to_numpy(float) if e in by["in"].index else np.full(64, np.nan)
            v_ring = by["ring"].loc[e, BANDS].to_numpy(float) if e in by["ring"].index else np.full(64, np.nan)
            if np.isnan(v_in).any():
                n_missing += 1
                continue
            if np.isnan(v_ring).any():
                v_ring, n_ring_fill = v_in, n_ring_fill + 1
            eids.append(e)
            feats.append(np.concatenate([v_in, v_ring]))
            years.append(int(by["in"].loc[e, "year_a"]))
    np.savez_compressed(OUT / "aef.npz", eid=np.array(eids), feat=np.array(feats, np.float32), year_a=np.array(years))
    print(f"packed {len(eids)} events; ring filled from polygon {n_ring_fill}; no polygon value {n_missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "pack"])
    ap.add_argument("--set", choices=["fn", "e2", "e3"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    fetch(a.set, a.workers) if a.cmd == "fetch" else pack()


if __name__ == "__main__":
    main()
