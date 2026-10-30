"""design notes-2: external set E from the Sims et al. 2025 plots in Indonesia, inputs built with the v1 pipeline.
--pool trop (design notes): set E2, tropical plots (|lat| <= 23.5) outside Indonesia (LSIB 2017) in three region groups,
the same event rules applied to every candidate, then 500 cells per group drawn at random (seed 20261027);
written to data/interim/rs_forestnet_sims2/.
--pool e3 (design notes): set E3, every valid cell of the same pool that is not in E2 (matched on src and ID), after
checking that the v4 draw is reproduced; written to data/interim/rs_forestnet_sims3/.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/sims_build.py [--pool idn|trop|e3] --what events|series|static|chips

events -> data/interim/rs_forestnet_sims/events.csv + polygons.geojson: per 0.01 deg cell the Hansen GFC v1.13 year with
          the largest loss area 2001-2022 (ties: earlier), loss pixels of that year vectorised at 30 m; cells with that
          year in 2021-2022 or less than 0.5 ha of it are dropped. Labels are carried along but not used for any choice.
series -> series_gee.csv (v1 sensors and cloud mask, years -2..+4, polygon and 150 m ring, with nobs)
static -> static.csv (area_ha, n_parts, tc2000_in, srtm0_mean)
chips  -> chips/<sid>/k<t>_<year>.png (v1 chips: all Landsat, 128 px over 3 km, centred on the cell)
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from shapely.geometry import mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsdrivers import gee  # noqa: E402
from rsdrivers.io import ROOT  # noqa: E402

RAW = ROOT / "data/raw/sims2025"
OUT = ROOT / "data/interim/rs_forestnet_sims"  # --pool trop switches this to data/interim/rs_forestnet_sims2
AGRI = {"Permanent agriculture", "Shifting cultivation"}
RING_DEG = 150 / 111320.0
HALF = 0.005  # half of the 0.01 deg cell


def retry(fn, what, tries=6):
    for a in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"{what}: {str(e)[:150]}; retry", flush=True)
            time.sleep(30 * (a + 1))
    raise RuntimeError(f"{what} failed")


GROUPS = {"Africa": "Africa", "Latin America": "Latin America", "Southeast Asia": "Asia-Pacific",
          "Oceania": "Asia-Pacific", "Asia": "Asia-Pacific"}


def trop_candidates(ee):
    """v4 §1: tropical plots outside Indonesia, country by LSIB 2017; labels carried along, not used for any choice."""
    import numpy as np
    from sklearn.neighbors import BallTree
    from rsdrivers.io import events as fn_events
    v = pd.read_csv(RAW / "validation_2001_2022.csv").assign(src="validation")
    t = pd.read_csv(RAW / "training_2001_2022_v1.csv").assign(src="training")
    s = pd.concat([v, t], ignore_index=True)
    s = s[(s.Driver_primary != "Noise/non-forest") & (s.Latitude.abs() <= 23.5) & s.Region.isin(GROUPS)].reset_index(drop=True)
    s["key"] = s.src + ":" + s.ID.astype(str)
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    cmap = {}
    for i in range(0, len(s), 400):
        part = s.iloc[i:i + 400]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(float(r.Longitude), float(r.Latitude)), {"key": r.key})
                                   for r in part.itertuples()])
        j = ee.Join.saveFirst("c").apply(fc, lsib, ee.Filter.intersects(leftField=".geo", rightField=".geo"))
        j = j.map(lambda f: f.set("country", ee.Feature(f.get("c")).get("country_na")))
        got = retry(lambda: (j.aggregate_array("key").getInfo(), j.aggregate_array("country").getInfo()), "lsib")
        cmap.update(dict(zip(*got)))
        print(f"countries {min(i + 400, len(s))}/{len(s)}", flush=True)
    s["country"] = s.key.map(cmap)
    s = s[s.country.notna() & (s.country != "Indonesia")].reset_index(drop=True)
    s["group"] = s.Region.map(GROUPS)
    tr = fn_events()
    tr = tr[tr.split == "train"]
    d = BallTree(np.radians(tr[["latitude", "longitude"]].to_numpy()), metric="haversine") \
        .query(np.radians(s[["Latitude", "Longitude"]].to_numpy()), k=1)[0][:, 0] * 6371
    s["dist_fn_train_km"] = d
    return s[["ID", "src", "Latitude", "Longitude", "Driver_primary", "Confidence_primary", "Tags", "dist_fn_train_km",
              "country", "group"]]


def events(ee, pool, workers=1):
    if pool == "idn":
        s = pd.read_csv(RAW / "sims_idn_plots_hansen.csv")  # Indonesia (LSIB 2017), noise removed; earlier 1 km-buffer stats
        s = s[["ID", "src", "Latitude", "Longitude", "Driver_primary", "Confidence_primary", "Tags", "dist_fn_train_km"]]
    else:
        s = trop_candidates(ee)
    s = s.reset_index(drop=True)
    s.insert(0, "sid", range(len(s)))
    ly = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("lossyear")
    area = ee.Image.pixelArea().divide(1e4)
    rows, geoms = [], {}

    def hist_of(i):  # one request per 50 cells; several at once with --workers (24/10/2026, E3: same results)
        part = s.iloc[i:i + 50]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle([r.Longitude - HALF, r.Latitude - HALF,
                                                                      r.Longitude + HALF, r.Latitude + HALF]),
                                              {"sid": int(r.sid)}) for r in part.itertuples()])
        return retry(lambda: fc.map(lambda f: f.set("groups", area.addBands(ly.updateMask(ly.gte(1).And(ly.lte(22))))
                                                    .reduceRegion(ee.Reducer.sum().group(1, "y"), f.geometry(), 30,
                                                                  maxPixels=1e8).get("groups"))).getInfo(), "hist")
    with ThreadPoolExecutor(workers) as pool_:
        hists = list(pool_.map(hist_of, range(0, len(s), 50)))
    for i, hist in zip(range(0, len(s), 50), hists):
        for f in hist["features"]:
            sid = f["properties"]["sid"]
            g = {int(x["y"]): x["sum"] for x in (f["properties"].get("groups") or [])}
            if not g:
                continue
            best = sorted(g, key=lambda y: (-g[y], y))[0]
            rows.append({"sid": sid, "year": 2000 + best, "year_ha": g[best], "loss_ha_2001_2022": sum(g.values())})
        print(f"histograms {min(i + 50, len(s))}/{len(s)}", flush=True)
    h = pd.DataFrame(rows)
    s = s.merge(h, on="sid", how="left")
    keep = s.year.between(2001, 2020) & (s.year_ha >= 0.5)
    print(f"cells {len(s)} | dropped: year 2021-2022 {int(s.year.between(2021, 2022).sum())}, "
          f"<0.5 ha {int((s.year_ha < 0.5).sum())}, no loss {int(s.year.isna().sum())}", flush=True)
    s = s[keep].reset_index(drop=True)
    if pool in ("trop", "e3"):  # v4 §1: 500 valid cells per region group, drawn after the event rules (no labels used)
        import numpy as np
        rng = np.random.default_rng(20261027)
        print("valid cells per group:", s.group.value_counts().to_dict(), flush=True)
        drawn = pd.concat([g.iloc[np.sort(rng.permutation(len(g))[:500])] for _, g in s.groupby("group", sort=True)])
        if pool == "trop":
            s = drawn
        else:  # v5 §1: E3 = valid cells not in E2, matched on (src, ID); the reproduced draw is only a check
            e2 = pd.read_csv(ROOT / "data/interim/rs_forestnet_sims2/events.csv")
            k2 = set(zip(e2.src, e2.ID))
            kd = set(zip(drawn.src, drawn.ID))
            print(f"E2 check: {len(k2)} E2 cells, {len(k2 & set(zip(s.src, s.ID)))} in the recomputed pool, "
                  f"draw reproduced exactly: {kd == k2}", flush=True)
            s = s[[k not in k2 for k in zip(s.src, s.ID)]]
            print("E3 cells per group:", s.group.value_counts().to_dict(), flush=True)
        s = s.drop(columns="sid").reset_index(drop=True)
        s.insert(0, "sid", range(len(s)))

    def vec_of(i):  # vectorise the event-year loss pixels per cell
        part = s.iloc[i:i + 25]

        def vec(r):
            box = ee.Geometry.Rectangle([r.Longitude - HALF, r.Latitude - HALF, r.Longitude + HALF, r.Latitude + HALF])
            m = ly.eq(int(r.year) - 2000).selfMask()
            v = m.reduceToVectors(geometry=box, scale=30, geometryType="polygon", eightConnected=True, maxPixels=1e8)
            return ee.Feature(v.geometry(), {"sid": int(r.sid)})
        fc = ee.FeatureCollection([vec(r) for r in part.itertuples()])
        return retry(lambda: fc.getInfo(), "vectors")
    with ThreadPoolExecutor(workers) as pool_:
        vecs = list(pool_.map(vec_of, range(0, len(s), 25)))
    for i, got in zip(range(0, len(s), 25), vecs):
        for f in got["features"]:
            geoms[f["properties"]["sid"]] = shape(f["geometry"]).buffer(0)
        print(f"polygons {min(i + 25, len(s))}/{len(s)}", flush=True)
    s["label_agri"] = s.Driver_primary.isin(AGRI).astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "events.csv", index=False)
    (OUT / "polygons.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"sid": int(k)}, "geometry": mapping(v)} for k, v in geoms.items()]}))
    print(f"events: {len(s)} cells, polygons {len(geoms)}", flush=True)


def load():
    ev = pd.read_csv(OUT / "events.csv")
    gj = json.loads((OUT / "polygons.geojson").read_text())
    return ev, {f["properties"]["sid"]: shape(f["geometry"]) for f in gj["features"]}


def series(ee, a):
    ev, geoms = load()
    out = OUT / "series_gee.csv"
    cols = ["eid", "k", "year", "zone", "ndvi_median", "nbr_median", "ndmi_median", "nobs_median", "ndvi_count"]
    done = set()
    if out.exists():
        done = {(int(r["eid"]), int(r["k"]), r["zone"]) for r in csv.DictReader(open(out))}
    jobs = defaultdict(list)
    for r in ev.itertuples():
        poly = geoms[r.sid]
        ring = poly.buffer(RING_DEG).difference(poly)
        for k in range(-2, 5):
            for zone, g in (("in", poly), ("ring", ring)):
                if (r.sid, k, zone) not in done:
                    jobs[int(r.year) + k].append({"id": f"{r.sid}|{k}|{zone}", "geom": mapping(g.simplify(0.00005))})
    tasks = [(y, f[i:i + 200]) for y, f in sorted(jobs.items()) for i in range(0, len(f), 200)]
    print(f"{sum(len(f) for f in jobs.values())} features in {len(tasks)} requests", flush=True)

    def run(t):
        y, part = t
        props = retry(lambda: gee.reduce_year(y, part), f"year {y}")
        return [{"eid": p["fid"].split("|")[0], "k": p["fid"].split("|")[1], "year": y, "zone": p["fid"].split("|")[2],
                 **{c: p.get(c) for c in cols[4:]}} for p in props]
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.map(run, tasks), 1):
            w.writerows(rows)
            f.flush()
            print(f"{n}/{len(tasks)}", flush=True)


def static(ee):
    ev, geoms = load()
    img = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select("treecover2000").rename("tc2000_in") \
        .addBands(ee.Image("USGS/SRTMGL1_003").rename("srtm0_mean"))
    feats = []
    for i in range(0, len(ev), 300):  # 24/10/2026 (E3): 2,306 polygons in one request exceeded the 10 MB payload limit
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry(mapping(geoms[r.sid].simplify(0.00005))), {"sid": int(r.sid)})
                                   for r in ev.iloc[i:i + 300].itertuples()])
        feats += retry(lambda: img.reduceRegions(fc, ee.Reducer.mean(), 30, tileScale=4).getInfo(), "static")["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        g = geoms[p["sid"]]
        lat = g.centroid.y
        import math
        area = g.area * (111320.0 * math.cos(math.radians(lat))) * 110574.0 / 1e4
        rows.append({"sid": p["sid"], "area_ha": area, "n_parts": len(g.geoms) if hasattr(g, "geoms") else 1,
                     "tc2000_in": p.get("tc2000_in"), "srtm0_mean": p.get("srtm0_mean")})
    pd.DataFrame(rows).to_csv(OUT / "static.csv", index=False)
    print("static rows", len(rows))


def chips(a):
    ev, _ = load()
    jobs = []
    for r in ev.itertuples():
        for t in range(-1, 5):
            y = int(r.year) + t
            p = OUT / "chips" / str(r.sid) / f"k{t}_{y}.png"
            if y <= 2024 and not p.exists():
                jobs.append((r.sid, y, float(r.Longitude), float(r.Latitude), p))

    def one(j):
        sid, y, lon, lat, p = j
        for att in range(5):
            try:
                with urllib.request.urlopen(gee.chip_url(y, lon, lat, bounded=a.pool == "e3"), timeout=180) as resp:
                    data = resp.read()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                return True
            except Exception:
                time.sleep(20 * (att + 1))
        print("failed", sid, y, flush=True)
        return False
    print(len(jobs), "chips to fetch", flush=True)
    with ThreadPoolExecutor(a.threads) as pool:
        ok = sum(pool.map(one, jobs))
    print(f"done {ok}/{len(jobs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True, choices=["events", "series", "static", "chips"])
    ap.add_argument("--pool", default="idn", choices=["idn", "trop", "e3"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    global OUT
    if a.pool == "trop":
        OUT = ROOT / "data/interim/rs_forestnet_sims2"
    if a.pool == "e3":
        OUT = ROOT / "data/interim/rs_forestnet_sims3"
    ee = gee.init()
    {"events": lambda: events(ee, a.pool, a.workers), "series": lambda: series(ee, a), "static": lambda: static(ee), "chips": lambda: chips(a)}[a.what]()


if __name__ == "__main__":
    main()
