"""Yearly Landsat index series (k = -2..+4 years around the loss year) for ForestNet events, via Earth Engine.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/build_series_gee.py [--limit N] [--batch 150]

Output: data/interim/rs_forestnet/series_gee.csv, one row per event x year x zone (in = loss polygon, ring = 150 m
around it). Requests already answered are skipped, so the run resumes after an interruption.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shapely.geometry import mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsdrivers import gee  # noqa: E402
from rsdrivers.io import INTERIM, events, region_lonlat  # noqa: E402

OFFSETS = range(-2, 5)
LAST_YEAR = 2024
RING_DEG = 150 / 111320.0
COLS = ["eid", "k", "year", "zone", "ndvi_median", "nbr_median", "ndmi_median", "nobs_median", "ndvi_count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = INTERIM / ("series_gee.csv" if a.limit is None else f"series_gee_test{a.limit}.csv")
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(int(r["eid"]), int(r["year"]), r["zone"]) for r in csv.DictReader(f)}
    d = events()
    if a.limit:
        d = d.sample(a.limit, random_state=1)
    jobs = defaultdict(list)  # calendar year -> features
    for r in d.itertuples():
        poly = region_lonlat(r).buffer(0)
        ring = poly.buffer(RING_DEG).difference(poly)
        for k in OFFSETS:
            y = int(r.year) + k
            if y > LAST_YEAR:
                continue
            for zone, g in (("in", poly), ("ring", ring)):
                if (r.eid, y, zone) not in done:
                    jobs[y].append({"id": f"{r.eid}|{k}|{zone}", "geom": mapping(g.simplify(0.00005))})
    gee.init()
    tasks = [(y, feats[i:i + a.batch]) for y, feats in sorted(jobs.items()) for i in range(0, len(feats), a.batch)]
    print(f"{sum(len(f) for f in jobs.values())} features in {len(tasks)} requests", flush=True)

    def run(task):
        y, part = task
        for attempt in range(6):
            try:
                return y, gee.reduce_year(y, part)
            except Exception as e:  # quota / transient errors: back off and retry
                print(f"year {y}: {str(e)[:160]}; retry in {30 * (attempt + 1)} s", flush=True)
                time.sleep(30 * (attempt + 1))
        print(f"year {y}: gave up on a batch of {len(part)} (rerun to retry)", flush=True)
        return y, []

    new = not out.exists()
    t0 = time.time()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for n, (y, props) in enumerate(pool.map(run, tasks), 1):
            for p in props:
                eid, k, zone = p["fid"].split("|")
                w.writerow({"eid": eid, "k": k, "year": y, "zone": zone, **{c: p.get(c) for c in COLS[4:]}})
            f.flush()
            print(f"{n}/{len(tasks)} requests (year {y}), {time.time() - t0:.0f} s", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
