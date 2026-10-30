"""Yearly false-colour chips (k = -1..+4) for each ForestNet event, for the VLM evidence card.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/build_chips_gee.py [--limit N] [--threads 8]

Output: data/interim/rs_forestnet/chips/<eid>/k<k>_<year>.png (128 x 128 px over 3 km). Existing files are skipped.
"""
import argparse
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsdrivers import gee  # noqa: E402
from rsdrivers.io import INTERIM, events  # noqa: E402

OFFSETS = range(-1, 5)
LAST_YEAR = 2024


def one(job):
    eid, k, year, lon, lat, path = job
    for attempt in range(5):
        try:
            url = gee.chip_url(year, lon, lat)
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return True
        except Exception as e:
            time.sleep(20 * (attempt + 1))
            err = e
    print(f"failed {eid} k={k}: {str(err)[:120]}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    d = events()
    if a.limit:
        d = d.sample(a.limit, random_state=1)
    root = INTERIM / "chips"
    jobs = []
    for r in d.itertuples():
        for k in OFFSETS:
            y = int(r.year) + k
            p = root / str(r.eid) / f"k{k}_{y}.png"
            if y <= LAST_YEAR and not p.exists():
                jobs.append((r.eid, k, y, r.longitude, r.latitude, p))
    gee.init()
    print(f"{len(jobs)} chips to fetch", flush=True)
    ok = 0
    with ThreadPoolExecutor(a.threads) as pool:
        for n, res in enumerate(pool.map(one, jobs)):
            ok += res
            if n % 100 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    print(f"done: {ok}/{len(jobs)}")


if __name__ == "__main__":
    main()
