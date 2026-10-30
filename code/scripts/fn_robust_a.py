"""Driver attribution robustness batches (25/10/2026, after E3 was scored; no model or threshold is chosen on E3).
Every batch: package -> upload -> run -> pull -> delete on the server.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_robust_a.py setup                    (server2: CPU wheels)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_robust_a.py run --phase cpu          (server2, CPU)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_robust_a.py run --phase backbones    (server2, RTX 3060)
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_robust_a.py run --phase ablate --gpus 1   (server1 queue)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_robust_a.py convert

cpu: server/robust_a.py (block bootstrap, repeated calibration, tabular baseline M1x) on an anonymous bundle (ids,
     labels, probabilities, index series and four static layers; no coordinates except the 1-degree cell and country).
backbones: server/cnn_backbones.py (ResNet-50, ConvNeXt-Tiny; the M2 recipe) on ForestNet, Sims Indonesia, E2, E3.
ablate: the v1 M3c adapter on E2 and E3 at k = 4 with the full card, image only and text only (pipeline_robust.py).
Results: runs/rs_forestnet_robust/raw/<phase>/, converted to runs/rs_forestnet_robust/.
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
from fn_v4 import M3C_ADAPTER, ORDERS, e2_facts, pack_e2  # noqa: E402
from fn_v5 import k4_probs, pack_e3, queued, sims_job  # noqa: E402
from rsdrivers.card import question  # noqa: E402
from rsdrivers.features import series_wide  # noqa: E402,F401
from rsdrivers.lab import Lab  # noqa: E402

SIDN, E2, E3 = (ROOT / f"data/interim/{d}" for d in ("rs_forestnet_sims", "rs_forestnet_sims2", "rs_forestnet_sims3"))
INTERIM = ROOT / "data/interim/rs_forestnet"
OFF = {"sidn": 100000, "e2": 200000, "e3": 300000}
BASE = {"sidn": SIDN, "e2": E2, "e3": E3}
OUT = ROOT / "runs/rs_forestnet_robust"
HOSTS = {"server2": {"remote": "/home/user/llm4deforest_v4/runs", "py": "/home/user/llm4deforest_v4/venv/bin/python"},
         "server1": {"remote": "/data/tmp/llm4deforest_paperA_robust", "py": "/home/user/llm4deforest/paperA/venv/bin/python"}}
CODE = {"cpu": ["server/robust_a.py", "rsdrivers/earlystop_v2.py", "server/pipeline_robust.py", "server/gpu_pipeline.py"],
        "backbones": ["server/cnn_backbones.py", "server/cnn_train.py", "server/pipeline_robust.py", "server/gpu_pipeline.py"],
        "ablate": ["server/vlm_infer.py", "server/render_cards.py", "rsdrivers/render.py", "server/pipeline_robust.py",
                   "server/gpu_pipeline.py"]}
WEIGHTS = {"resnet50-11ad3fa6.pth": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
           "convnext_tiny-983f1562.pth": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"}
WHEELS = ["pandas", "scipy", "scikit-learn", "lightgbm"]
CACHE = ROOT / "data/cache/robust_a"  # downloaded weights and wheels (gitignored data/cache)


def pack_sims(base, off, path):
    """CNN input of a Sims set as fn_v4.pack_e2: years -1..4, each chip resized to 128 x 128, central 96 px."""
    from PIL import Image
    se = pd.read_csv(base / "events.csv", usecols=["sid", "year"])
    X = np.zeros((len(se), 6, 96, 96, 3), np.uint8)
    for i, r in enumerate(se.itertuples()):
        for j, t in enumerate(range(-1, 5)):
            p = base / "chips" / str(r.sid) / f"k{t}_{int(r.year) + t}.png"
            if p.exists():
                with Image.open(p) as im:
                    X[i, j] = np.asarray(im.convert("RGB").resize((128, 128), Image.BILINEAR))[16:112, 16:112]
    np.savez_compressed(path, X=X, y=np.full(len(se), -1), part=np.array(["ext"] * len(se)), eid=(off + se.sid).to_numpy())


def bundle_cpu(b: Path):
    """Ids, labels, probabilities, index series and four static layers; country and the 1-degree cell only."""
    b.mkdir(parents=True)
    z = np.load(ROOT / "runs/rs_forestnet/probs/M2.npz", allow_pickle=True)
    eid, part, y, probs = z["eid"], z["part"], z["y"], z["probs"]  # read each key once
    pd.DataFrame({"eid": eid, "part": part, "y": y}).to_csv(b / "fn_events.csv", index=False)
    np.savez_compressed(b / "fn_M2.npz", probs=probs, eid=eid)
    shutil.copy(ROOT / "runs/rs_forestnet/probs/M3c_val_cal+test.npz", b / "fn_M3c.npz")
    pd.read_csv(INTERIM / "series_gee.csv").to_csv(b / "fn_series.csv", index=False)
    pd.read_parquet(INTERIM / "static_features.parquet")[["eid", "area_ha", "n_parts", "tc2000_in", "srtm0_mean"]].to_csv(
        b / "fn_static.csv", index=False)
    for t, base in BASE.items():
        ev = pd.read_csv(base / "events.csv")
        keep = ["eid", "label_agri", "Latitude", "Longitude"] + [c for c in ("country", "group") if c in ev]
        ev = ev.assign(eid=OFF[t] + ev.sid)
        ev = ev.assign(Latitude=np.floor(ev.Latitude), Longitude=np.floor(ev.Longitude))[keep]  # 1-degree cell only
        ev.to_csv(b / f"{t}_events.csv", index=False)
        s = pd.read_csv(base / "series_gee.csv")
        s.assign(eid=OFF[t] + s.eid).to_csv(b / f"{t}_series.csv", index=False)
        st = pd.read_csv(base / "static.csv")
        st.assign(eid=OFF[t] + st.sid).drop(columns=["sid"]).to_csv(b / f"{t}_static.csv", index=False)
    # model probabilities as used in the article
    v3 = np.load(ROOT / "runs/rs_forestnet_v3/probs/M2.npz", allow_pickle=True)
    v3p, v3e = v3["probs"], v3["eid"]
    ext = v3e >= OFF["sidn"]
    np.savez_compressed(b / "sidn_M2.npz", probs=v3p[ext], eid=v3e[ext])
    np.savez_compressed(b / "fn3_M2.npz", probs=v3p[~ext], eid=v3e[~ext])  # the same retrained CNN on ForestNet
    shutil.copy(ROOT / "runs/rs_forestnet_v3/probs/M3c.npz", b / "sidn_M3c.npz")
    shutil.copy(ROOT / "runs/rs_forestnet_v4/probs/M2_e2.npz", b / "e2_M2.npz")
    shutil.copy(ROOT / "runs/rs_forestnet_v4/probs/M3c.npz", b / "e2_M3c.npz")
    for m in ("M2", "M3c", "M6"):
        shutil.copy(ROOT / f"runs/rs_forestnet_v5/probs/{m}_E3.npz", b / f"e3_{m}.npz")
    seeds = ROOT / "runs/rs_forestnet_v5/raw/seeds/out"
    for s in (1, 2):
        for t in ("e2", "e3"):
            eids, P = k4_probs(seeds / f"m3c_s{s}_{t}.jsonl")
            np.savez_compressed(b / f"{t}_M3c_s{s}.npz", probs=P, eid=eids)


def blank_facts(st):
    """Fact sheet of the E2/E3 cards with every number replaced by n/a (image-only ablation)."""
    text = e2_facts({}, st, 4)
    return re.sub(r"Loss area: .*?Mean elevation: [^.]*\.",
                  "Loss area: n/a ha in n/a patch(es). Tree cover in 2000 inside the area: n/a%. Peat soil: unknown. "
                  "Mean elevation: n/a m.", text)


def build_ablate(pkg: Path):
    jobs = [{"name": "blank_k4.jpg", "chips": [None] * 6, "labels": [f"year {t:+d}" for t in range(-1, 5)], "polys": []}]
    files = {}
    for t in ("e2", "e3"):
        base = BASE[t]
        ev = pd.read_csv(base / "events.csv", usecols=["sid", "year", "Latitude", "Longitude"])
        geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((base / "polygons.geojson").read_text())["features"]}
        swide = series_wide(pd.read_csv(base / "series_gee.csv"))
        sstat = pd.read_csv(base / "static.csv").set_index("sid")
        prefix = {"e2": "sims2", "e3": "sims3"}[t]
        (pkg / "chips" / prefix).mkdir(parents=True, exist_ok=True)
        for v in ("full", "img", "txt"):
            files[(t, v)] = open(pkg / f"m3c_{t}_{v}.jsonl", "w", encoding="utf-8")
        for r in ev.itertuples():
            src = base / "chips" / str(r.sid)
            if src.exists():
                shutil.copytree(src, pkg / "chips" / prefix / str(r.sid), dirs_exist_ok=True)
            j = sims_job(base, prefix, r, 4, geoms[r.sid])
            jobs.append(j)
            full = e2_facts(swide.loc[r.sid].to_dict() if r.sid in swide.index else {}, sstat.loc[r.sid], 4)
            img = blank_facts(sstat.loc[r.sid])
            for o, order in enumerate(ORDERS):
                q, _ = question(order)
                rid = f"{OFF[t] + int(r.sid)}|4|{o}"
                for v, text, image in (("full", full, j["name"]), ("img", img, j["name"]), ("txt", full, "blank_k4.jpg")):
                    files[(t, v)].write(json.dumps({"id": rid, "text": text + "\n\n" + q, "image": image, "shots": []}) + "\n")
    for f in files.values():
        f.close()
    (pkg / "jobs.json").write_text(json.dumps(jobs))
    shutil.copytree(M3C_ADAPTER, pkg / "adapters/m3c")


def fetch_weights():
    (CACHE / "torch").mkdir(parents=True, exist_ok=True)
    for name, url in WEIGHTS.items():
        dst = CACHE / "torch" / name
        if not dst.exists():
            subprocess.run(["curl", "-sL", "--max-time", "1800", "-o", str(dst), url], check=True)
        print(name, dst.stat().st_size // 2 ** 20, "MB", flush=True)


def build(pkg: Path, phase: str):
    (pkg / "code").mkdir(parents=True)
    for f in CODE[phase]:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    if phase == "cpu":
        bundle_cpu(pkg / "bundle")
    elif phase == "backbones":
        fetch_weights()
        ck = pkg / "torch_home/hub/checkpoints"
        ck.mkdir(parents=True)
        for name in WEIGHTS:
            shutil.copy(CACHE / "torch" / name, ck / name)
        shutil.copy(INTERIM / "chips_packed.npz", pkg / "chips_fn.npz")
        pack_e2(pkg / "chips_e2.npz")
        pack_e3(pkg / "chips_e3.npz")
        pack_sims(SIDN, OFF["sidn"], pkg / "chips_sidn.npz")
    else:
        build_ablate(pkg)
    print(f"package {phase} ready", flush=True)


def setup():
    """CPU packages for the project venv on server2 (wheels fetched here, installed offline over the LAN)."""
    whl = CACHE / "wheels"
    whl.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--python-version", "3.12",
                    "--implementation", "cp", "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
                    "--platform", "manylinux_2_17_x86_64", "-d", str(whl), *WHEELS], check=True)
    lab = Lab("server2")
    try:
        lab.ssh("rm -rf ~/llm4deforest_v4/wheels_tmp && mkdir -p ~/llm4deforest_v4/wheels_tmp")
        for f in sorted(whl.glob("*.whl")):
            lab.put(f, f"/home/user/llm4deforest_v4/wheels_tmp/{f.name}")
        print(lab.ssh("~/.local/bin/uv pip install --python ~/llm4deforest_v4/venv/bin/python --no-index "
                      f"--find-links ~/llm4deforest_v4/wheels_tmp {' '.join(WHEELS)} 2>&1 | tail -5; "
                      "rm -rf ~/llm4deforest_v4/wheels_tmp", timeout=900))
        print(lab.ssh("~/llm4deforest_v4/venv/bin/python -c 'import pandas, scipy, sklearn, lightgbm, torch; "
                      "print(pandas.__version__, scipy.__version__, sklearn.__version__, lightgbm.__version__, torch.__version__)'"))
    finally:
        lab.close()


def run(a):
    host = "server1" if a.phase == "ablate" else "server2"
    H = HOSTS[host]
    lab = Lab("lab" if host == "server1" else host)
    run_id = a.run_id or f"rob{a.phase}_{time.strftime('%Y%m%d%H%M%S')}"
    R = H["remote"]
    raw = OUT / "raw" / a.phase
    keep = False
    try:
        if a.run_id:  # resume: the batch is already unpacked on the server; stale exit markers go
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
        pipe = f"{H['py']} {R}/{run_id}/code/pipeline_robust.py --phase {a.phase} --gpus {a.gpus} --parts {a.parts}"
        if host == "server1":
            pipe = queued(lab, a.gpus, f"{R}/{run_id}", pipe)
        code = lab.run_detached(R, f"cd {R}/{run_id} && {pipe} >> {R}/{run_id}/pipeline_log.txt 2>&1", f"{R}/{run_id}.exit",
                                poll_s=120, timeout_h=a.hours)
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -3 {R}/{run_id}/pipeline_log.txt"), "log"), flush=True)
        lab.retry(lambda: lab.ssh(f"cd {R}/{run_id} && tar cf ../{run_id}_out.tar out logs state pipeline_log.txt"), "pack")
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


def convert():
    """Letter probabilities of the ablation -> 4-class k = 4 npz; backbone seeds -> mean probabilities per set."""
    OUT.mkdir(parents=True, exist_ok=True)
    ab = OUT / "raw/ablate/out"
    for f in sorted(ab.glob("m3c_*.jsonl")) if ab.exists() else []:
        eids, P = k4_probs(f)
        np.savez_compressed(OUT / f"{f.stem}.npz", probs=P, eid=eids)
        print("saved", f.stem, P.shape)
    bb = OUT / "raw/backbones/out"
    for arch in ("resnet50", "convnext_tiny"):
        for t in ("fn", "sidn", "e2", "e3"):
            zs = [np.load(f, allow_pickle=True) for f in sorted(bb.glob(f"{arch}_s*_{t}.npz"))] if bb.exists() else []
            if zs:
                np.savez_compressed(OUT / f"{arch}_{t}.npz", probs=np.mean([z["probs"] for z in zs], 0), eid=zs[0]["eid"],
                                    n_seeds=len(zs))
                print("saved", arch, t, len(zs), "seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "run", "convert"])
    ap.add_argument("--phase", choices=["cpu", "backbones", "ablate"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--hours", type=float, default=30)
    ap.add_argument("--parts", default="1,3,2", help="cpu phase: which analyses of robust_a.py")
    ap.add_argument("--run-id", help="resume a batch already unpacked on the server")
    a = ap.parse_args()
    {"setup": setup, "run": lambda: run(a), "convert": convert}[a.cmd]()


if __name__ == "__main__":
    main()
