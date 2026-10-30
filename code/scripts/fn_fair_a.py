"""Driver attribution fair-comparison batches (26/10/2026): does the VLM's cross-continent advantage hold when the CNN sees the same
3 km chips, when the CNN also gets the numbers of the fact sheet, and when the VLM is trained on images only?
No choice is made on the external sets; every model is selected on the ForestNet selection half as M2 and M3c.
Every batch: package -> upload -> run -> pull -> delete on the server.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_fair_a.py run --phase cnn               (server2, RTX 3060)
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_fair_a.py run --phase vlmimg --gpus 1,0   (server1)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_fair_a.py evaluate

cnn: server/cnn_fair.py, variants img3km (full 128 px chips) and facts (full chips + fact-sheet numbers), seeds 0-2.
vlmimg: the M3c recipe (qlora_train.py, seed 0) on cards whose fact sheet has every number replaced by n/a, in training
        and at inference; ForestNet test, Indonesia, E2 and E3 at k = 4.
Results: runs/rs_forestnet_robust/raw/fair_<phase>/; evaluate writes runs/rs_forestnet_robust/fair_comparison.json.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))
from fn_v4 import ORDERS  # noqa: E402
from fn_v5 import k4_probs, queued, sims_job  # noqa: E402
from rsdrivers.card import fact_sheet, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, events, load_region  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402
from rsdrivers.render import job_for  # noqa: E402

SETS = {"sidn": (ROOT / "data/interim/rs_forestnet_sims", 100000, "sims1"),
        "e2": (ROOT / "data/interim/rs_forestnet_sims2", 200000, "sims2"),
        "e3": (ROOT / "data/interim/rs_forestnet_sims3", 300000, "sims3")}
OUT = ROOT / "runs/rs_forestnet_robust"
HOSTS = {"server2": {"remote": "/home/user/llm4deforest_v4/runs", "py": "/home/user/llm4deforest_v4/venv/bin/python"},
         "server1": {"remote": "/data/tmp/llm4deforest_paperA_robust", "py": "/home/user/llm4deforest/paperA/venv/bin/python"}}
CODE = {"cnn": ["server/cnn_fair.py", "server/cnn_train.py", "server/pipeline_fair.py", "server/gpu_pipeline.py"],
        "vlmimg": ["server/qlora_train.py", "server/vlm_infer.py", "server/render_cards.py", "rsdrivers/render.py",
                   "server/pipeline_fair.py", "server/gpu_pipeline.py"]}
R18 = ("resnet18-f37072fd.pth", "https://download.pytorch.org/models/resnet18-f37072fd.pth")
CACHE = ROOT / "data/cache/robust_a"
INDICES = ["ndvi", "nbr", "ndmi"]
YEARS = range(-2, 5)
TAB_COLS = [f"{i}_{z}_y{t}" for i in INDICES for z in ("in", "ring") for t in YEARS] + [f"nobs_in_y{t}" for t in YEARS]
NUMS = re.compile(r"Loss area: .*?Mean elevation: [^.]*\.")
BLANK = ("Loss area: n/a ha in n/a patch(es). Tree cover in 2000 inside the area: n/a%. Peat soil: unknown. "
         "Mean elevation: n/a m.")


# ---------------------------------------------------------------- CNN inputs
def fn_frame():
    return events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")


def chips_full(paths_by_row, n):
    from PIL import Image
    X = np.zeros((n, 6, 128, 128, 3), np.uint8)
    for i, paths in enumerate(paths_by_row):
        for j, p in enumerate(paths):
            if p.exists():
                with Image.open(p) as im:
                    X[i, j] = np.asarray(im.convert("RGB").resize((128, 128), Image.BILINEAR))
    return X


def tab_array(wide, static, ids, peat):
    """T [n, 5, F]: the fact-sheet numbers available at step k (years after k missing) and the static values."""
    w = wide.reindex(ids)
    for c in TAB_COLS:
        assert c in w.columns, c
    st = static.reindex(ids)
    stat = np.stack([np.log1p(st.area_ha), np.log1p(st.n_parts), st.tc2000_in, st.srtm0_mean,
                     peat if peat is not None else np.full(len(ids), np.nan)], 1)
    T = np.full((len(ids), 5, len(TAB_COLS) + stat.shape[1]), np.nan, np.float32)
    base = w[TAB_COLS].to_numpy(np.float32)
    for k in range(5):
        v = base.copy()
        for j, c in enumerate(TAB_COLS):
            if int(c.rsplit("_y", 1)[1]) > k:
                v[:, j] = np.nan
        T[:, k] = np.concatenate([v, stat], 1)
    return T


def pack_cnn(pkg: Path):
    fn = fn_frame()
    X = chips_full(([INTERIM / "chips" / str(r.eid) / f"k{t}_{int(r.year) + t}.png" for t in range(-1, 5)]
                    for r in fn.itertuples()), len(fn))
    y = fn.merged_label.map(LABELS4.index).to_numpy()
    np.savez_compressed(pkg / "chips_fn.npz", X=X, y=y, part=fn.part.to_numpy(), eid=fn.eid.to_numpy())
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    T = tab_array(wide, static, fn.eid.to_numpy(), static.reindex(fn.eid).peat.astype(float).to_numpy())
    np.savez_compressed(pkg / "tab_fn.npz", T=T, eid=fn.eid.to_numpy())
    for t, (base, off, _) in SETS.items():
        ev = pd.read_csv(base / "events.csv", usecols=["sid", "year"])
        X = chips_full(([base / "chips" / str(r.sid) / f"k{u}_{int(r.year) + u}.png" for u in range(-1, 5)]
                        for r in ev.itertuples()), len(ev))
        eid = (off + ev.sid).to_numpy()
        np.savez_compressed(pkg / f"chips_{t}.npz", X=X, y=np.full(len(ev), -1), part=np.array(["ext"] * len(ev)), eid=eid)
        wide = series_wide(pd.read_csv(base / "series_gee.csv"))
        static = pd.read_csv(base / "static.csv").set_index("sid")
        np.savez_compressed(pkg / f"tab_{t}.npz", T=tab_array(wide, static, ev.sid.to_numpy(), None), eid=eid)
    dst = CACHE / "torch" / R18[0]
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", "--max-time", "1800", "-o", str(dst), R18[1]], check=True)
    ck = pkg / "torch_home/hub/checkpoints"
    ck.mkdir(parents=True)
    shutil.copy(dst, ck / R18[0])


# ---------------------------------------------------------------- image-only VLM inputs
def pack_vlmimg(pkg: Path):
    fn = fn_frame()
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    jobs, files = [], {}

    def w(name, row):
        if name not in files:
            files[name] = open(pkg / name, "w", encoding="utf-8")
        files[name].write(json.dumps(row) + "\n")
    qs = [question(o) for o in ORDERS]
    for r in fn[fn.part.isin(["train", "val_sel", "test"])].itertuples():
        src = INTERIM / "chips" / str(r.eid)
        if src.exists():
            shutil.copytree(src, pkg / "chips" / str(r.eid), dirs_exist_ok=True)
        poly = load_region(r.example_path)
        for k in (range(5) if r.part != "test" else [4]):
            jobs.append(job_for(r, k, poly, INTERIM / "chips"))
            facts = NUMS.sub(BLANK, fact_sheet(r, {}, static.loc[r.eid], k))
            assert "n/a ha" in facts
            if r.part == "train":
                w("img_train.jsonl", {"eid": int(r.eid), "k": k, "image": f"{r.eid}_k{k}.jpg",
                                      "texts": [facts + "\n\n" + q for q, _ in qs],
                                      "answers": [next(L for L, lab in mp.items() if lab == r.merged_label) for _, mp in qs]})
            else:
                for o, (q, _) in enumerate(qs):
                    w("img_valsel.jsonl" if r.part == "val_sel" else "img_fntest.jsonl",
                      {"id": f"{r.eid}|{k}|{o}", "text": facts + "\n\n" + q, "image": f"{r.eid}_k{k}.jpg", "shots": []})
    vs = fn[fn.part == "val_sel"]
    pd.DataFrame({"eid": vs.eid, "y": vs.merged_label.map(LABELS4.index)}).to_csv(pkg / "meta_valsel.csv", index=False)
    from fn_v4 import e2_facts
    for t, (base, off, prefix) in SETS.items():
        ev = pd.read_csv(base / "events.csv", usecols=["sid", "year", "Latitude", "Longitude"])  # no label column
        geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((base / "polygons.geojson").read_text())["features"]}
        sstat = pd.read_csv(base / "static.csv").set_index("sid")
        (pkg / "chips" / prefix).mkdir(parents=True, exist_ok=True)
        for r in ev.itertuples():
            src = base / "chips" / str(r.sid)
            if src.exists():
                shutil.copytree(src, pkg / "chips" / prefix / str(r.sid), dirs_exist_ok=True)
            j = sims_job(base, prefix, r, 4, geoms[r.sid])
            jobs.append(j)
            facts = NUMS.sub(BLANK, e2_facts({}, sstat.loc[r.sid], 4))
            assert "n/a ha" in facts
            for o, (q, _) in enumerate(qs):
                w(f"img_{t}.jsonl", {"id": f"{off + int(r.sid)}|4|{o}", "text": facts + "\n\n" + q, "image": j["name"],
                                     "shots": []})
    for f in files.values():
        f.close()
    (pkg / "jobs.json").write_text(json.dumps(jobs))


def build(pkg: Path, phase: str):
    (pkg / "code").mkdir(parents=True)
    for d in ("out", "adapters", "chips"):
        (pkg / d).mkdir()
    for f in CODE["vlmimg" if phase == "vlmimgseeds" else phase]:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    pack_cnn(pkg) if phase == "cnn" else pack_vlmimg(pkg)
    print(f"package {phase} ready", flush=True)


def run(a):
    host = "server1" if a.phase in ("vlmimg", "vlmimgseeds") else "server2"
    H = HOSTS[host]
    lab = Lab("lab" if host == "server1" else host)
    run_id, R = a.run_id or f"fair{a.phase}_{time.strftime('%Y%m%d%H%M%S')}", H["remote"]
    raw = OUT / "raw" / f"fair_{a.phase}"
    keep = False
    try:
        if a.run_id:  # resume a batch that is already unpacked on the server (stages already done are skipped)
            keep = True
            if lab.ssh(f"test -d {R}/{run_id}/code && echo ok || echo no").strip() != "ok":
                raise RuntimeError(f"{run_id} is not on {host}")
            lab.ssh(f"rm -f {R}/{run_id}.exit {R}/{run_id}.exit.pid {R}/{run_id}_out.tar")
        else:
            lab.ssh(f"mkdir -p {R} && chmod 700 {R}")
            with tempfile.TemporaryDirectory() as td:
                pkg = Path(td) / run_id
                build(pkg, a.phase)
                tar = Path(td) / f"{run_id}.tar"
                with tarfile.open(tar, "w") as t:
                    t.add(pkg, arcname=run_id)
                print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
                lab.put_chunked(tar, f"{R}/{run_id}.tar", chunk_mb=64 if host != "server1" else 4, workers=6)
            if lab.run_detached(R, f"tar xf {run_id}.tar && rm {run_id}.tar", f"{R}/{run_id}.unpack", poll_s=15) != 0:
                raise RuntimeError("unpack failed")
        keep = True
        pipe = f"{H['py']} {R}/{run_id}/code/pipeline_fair.py --phase {a.phase} --gpus {a.gpus} --seeds 1,2"
        if host == "server1":
            pipe = queued(lab, a.gpus, f"{R}/{run_id}", pipe)
        code = lab.run_detached(R, f"cd {R}/{run_id} && {pipe} >> {R}/{run_id}/pipeline_log.txt 2>&1", f"{R}/{run_id}.exit",
                                poll_s=120, timeout_h=10)
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -3 {R}/{run_id}/pipeline_log.txt"), "log"), flush=True)
        pull = "out logs state pipeline_log.txt"
        lab.retry(lambda: lab.ssh(f"cd {R}/{run_id} && tar cf ../{run_id}_out.tar {pull}"), "pack")
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir(parents=True)
        lab.get_chunked(f"{R}/{run_id}_out.tar", raw / "out.tar")
        with tarfile.open(raw / "out.tar") as t:
            t.extractall(raw, filter="data")
        (raw / "out.tar").unlink()
        (raw / "RUN_ID").write_text(run_id)
        if code != 0:
            raise RuntimeError(f"pipeline failed (exit {code}); data kept on {host} in {R}/{run_id}")
        keep = False
    finally:
        if keep:
            print(f"server data kept on {host} in {R}/{run_id}", flush=True)
        else:
            left = lab.retry(lambda: lab.ssh(f"rm -rf {R}/{run_id} {R}/{run_id}.tar* {R}/{run_id}_out.tar {R}/{run_id}.exit* "
                                             f"{R}/{run_id}.unpack*; ls -A {R} | wc -l"), "clean", tries=40)
            print(f"server data left on {host}:", left.strip(), flush=True)
        lab.close()


# ---------------------------------------------------------------- evaluation
def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def macro_f1(y, p):
    f = []
    for c in range(4):
        tp, fp, fn_ = ((p == c) & (y == c)).sum(), ((p == c) & (y != c)).sum(), ((p != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn_))
    return float(np.mean(f))


def load(path, ids, k4=True):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    P = np.stack([probs[pos[int(e)]] for e in ids])
    return P[:, 4] if (k4 and P.ndim == 3) else P


def agri(P):
    return ((P[:, 0] + P[:, 1]) >= 0.5).astype(int)


def boot(y, a1, a0, groups, rng, B=2000):
    n = len(y)
    cell = [(lambda i: ba(y[i], a1[i]) - ba(y[i], a0[i]))(rng.integers(0, n, n)) for _ in range(B)]
    out = {"diff": round(ba(y, a1) - ba(y, a0), 4), "ci_cell": [round(float(v), 4) for v in np.percentile(cell, [2.5, 97.5])]}
    if groups is not None:
        uc = np.unique(groups)
        mem = {c: np.flatnonzero(groups == c) for c in uc}
        g = []
        for _ in range(B):
            i = np.concatenate([mem[c] for c in rng.choice(uc, len(uc), replace=True)])
            g.append(ba(y[i], a1[i]) - ba(y[i], a0[i]))
        out["ci_country"] = [round(float(v), 4) for v in np.percentile(g, [2.5, 97.5])]
    return out


def evaluate():
    rng = np.random.default_rng(20261021)
    cnn_raw, img_raw = OUT / "raw/fair_cnn/out", OUT / "raw/fair_vlmimg/out"
    fn = fn_frame()
    te = fn[fn.part == "test"]
    yfn, ybin = te.merged_label.map(LABELS4.index).to_numpy(), te.merged_label.isin(LABELS4[:2]).astype(int).to_numpy()
    M = {"CNN (2.25 km)": {"fn": ROOT / "runs/rs_forestnet_v3/probs/M2.npz", "sidn": ROOT / "runs/rs_forestnet_v3/probs/M2.npz",
                           "e2": ROOT / "runs/rs_forestnet_v4/probs/M2_e2.npz", "e3": ROOT / "runs/rs_forestnet_v5/probs/M2_E3.npz"},
         "VLM": {"fn": ROOT / "runs/rs_forestnet/probs/M3c_val_cal+test.npz", "sidn": ROOT / "runs/rs_forestnet_v3/probs/M3c.npz",
                 "e2": ROOT / "runs/rs_forestnet_v4/probs/M3c.npz", "e3": ROOT / "runs/rs_forestnet_v5/probs/M3c_E3.npz"}}
    P = {m: {t: load(p, te.eid if t == "fn" else (SETS[t][1] + pd.read_csv(SETS[t][0] / "events.csv").sid))
             for t, p in d.items()} for m, d in M.items()}
    for v, lab in (("img3km", "CNN (3 km)"), ("facts", "CNN (3 km) + fact-sheet numbers")):
        P[lab] = {}
        for t in ("fn", "sidn", "e2", "e3"):
            ids = te.eid if t == "fn" else SETS[t][1] + pd.read_csv(SETS[t][0] / "events.csv").sid
            P[lab][t] = np.mean([load(cnn_raw / f"{v}_s{s}_{t}.npz", ids) for s in (0, 1, 2)], 0)
    P["VLM, images only"] = {}  # mean probabilities of seeds 0 (fair_vlmimg) and 1, 2 (fair_vlmimgseeds)
    for t, f in (("fn", "fntest"), ("sidn", "sidn"), ("e2", "e2"), ("e3", "e3")):
        ids = te.eid if t == "fn" else SETS[t][1] + pd.read_csv(SETS[t][0] / "events.csv").sid
        runs = []
        for path in [img_raw / f"img_{f}.jsonl"] + [OUT / f"raw/fair_vlmimgseeds/out/img_s{s}_{f}.jsonl" for s in (1, 2)]:
            eids, Q = k4_probs(path)
            pos = {int(e): i for i, e in enumerate(eids)}
            runs.append(np.stack([np.asarray(Q)[pos[int(e)]] for e in ids]))
        P["VLM, images only"][t] = np.mean(runs, 0)
    res = {"models": {}, "differences": {}, "vlm_img_choice": json.loads((img_raw / "img_choice.json").read_text())}
    ev = {t: pd.read_csv(SETS[t][0] / "events.csv") for t in SETS}
    for m, d in P.items():
        r = {"fn_macro_f1": round(macro_f1(yfn, d["fn"].argmax(1)), 4), "fn_BA": round(ba(ybin, agri(d["fn"])), 4)}
        for t in SETS:
            r[f"{t}_BA"] = round(ba(ev[t].label_agri.to_numpy(), agri(d[t])), 4)
            if "group" in ev[t]:
                r[f"{t}_BA_by_group"] = {g: round(ba(ev[t].label_agri.to_numpy()[m_], agri(d[t])[m_]), 4)
                                         for g in ("Africa", "Asia-Pacific", "Latin America") for m_ in [(ev[t].group == g).to_numpy()]}
        res["models"][m] = r
    for a1, a0 in (("VLM", "CNN (3 km)"), ("VLM", "CNN (3 km) + fact-sheet numbers"), ("VLM, images only", "CNN (3 km)"),
                   ("VLM, images only", "CNN (2.25 km)"), ("VLM", "VLM, images only"), ("CNN (3 km)", "CNN (2.25 km)"),
                   ("CNN (3 km) + fact-sheet numbers", "CNN (3 km)")):
        for t in SETS:
            y = ev[t].label_agri.to_numpy()
            res["differences"][f"{t}: {a1} - {a0}"] = boot(y, agri(P[a1][t]), agri(P[a0][t]),
                                                           ev[t].country.to_numpy() if "country" in ev[t] else None, rng)
    (OUT / "fair_comparison.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "evaluate"])
    ap.add_argument("--phase", choices=["cnn", "vlmimg", "vlmimgseeds"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--run-id")
    a = ap.parse_args()
    run(a) if a.what == "run" else evaluate()


if __name__ == "__main__":
    main()
