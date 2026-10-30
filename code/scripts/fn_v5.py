"""design notes driver: packages, batches on server1 and server2, conversion. Every batch: upload -> run -> pull -> delete.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py run --phase train [--gpus 0,1] [--smoke]
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py run --phase test   (after the E2 choice is hashed)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py run --phase cnn_e3            (server2, after the hash)
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py run --phase train --run-id v5train..._... (resume)
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py stage --phase test  (E3 inputs ahead of time)
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v5.py convert

On server1 every pipeline goes through the GPU queue (tools/gpu_queue.sh, first GPU of --gpus): our jobs one at a time,
each starting once that GPU has been free for 30 s. --run-id continues a batch still on the server (resumable stages).

train: M6 p = 0 and p = 0.5 on the ForestNet training events (cards k = 0..4 as M5e), then E2 (development) at k = 4;
       the server picks the candidate (pipeline_v5.py) and only that adapter comes back.
test:  the chosen M6 and the v1 M3c on E3 at k = 4. cnn_e3: M2 seeds 0-2 on E3.
Results land in runs/rs_forestnet_v5/raw/<phase>/. E3 events get eid 300000 + sid. E3 labels never enter a package.
"""
import argparse
import json
import math
import shutil
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
from fn_v4 import E2, M3C_ADAPTER, ORDERS, V3W, e2_facts, fn_events  # noqa: E402
from rsdrivers.card import OPTIONS, fact_sheet, question  # noqa: E402
from rsdrivers.features import LABELS4  # noqa: E402
from rsdrivers.features import series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, load_region  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402
from rsdrivers.render import SCALE, job_for  # noqa: E402

E3 = ROOT / "data/interim/rs_forestnet_sims3"
V5 = ROOT / "data/interim/rs_forestnet_v5"
RUNS5 = ROOT / "runs/rs_forestnet_v5"
OFF2, OFF3 = 200000, 300000
PH6 = "Satellite embedding features (AlphaEarth Foundations, the event area and its surroundings): " + "<|fim_pad|>" * 4
QUEUE = "/data/tmp/llm4deforest_gpuq/gpu_queue.sh"  # tools/gpu_queue.sh (24/10/2026)
ALIVE = 'p=$(cat {R}/{run_id}.exit.pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && echo yes || echo no'
HOSTS = {"server2": {"remote": "/home/user/llm4deforest_v4/runs", "py": "/home/user/llm4deforest_v4/venv/bin/python"},
         "server1": {"remote": "/data/tmp/llm4deforest_paperA_v5", "py": "/home/user/llm4deforest/paperA/venv/bin/python"}}
CODE = {"seeds": ["server/qlora_train.py", "server/vlm_infer.py", "server/gpu_pipeline.py", "server/pipeline_v5.py",
                  "server/render_cards.py", "rsdrivers/render.py"],
        "cnn_e3": ["server/cnn_train.py", "server/cnn_v3.py", "server/gpu_pipeline.py", "server/pipeline_v5.py"],
        "vlm": ["server/fn_m6.py", "server/fn_m5e.py", "server/fn_qlora_v3.py", "server/vlm_infer.py", "server/gpu_pipeline.py",
                "server/pipeline_v5.py", "server/render_cards.py", "rsdrivers/render.py"]}


def sims_job(base, prefix, r, k, poly):
    """Card job of a Sims cell (fn_v4.e2_job for any of E2 / E3)."""
    dlat = 1500 / 110574.0
    dlon = 1500 / (111320.0 * math.cos(math.radians(r.Latitude)))
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    pts = [[[(x - r.Longitude + dlon) / (2 * dlon) * 128 * SCALE, (r.Latitude + dlat - y) / (2 * dlat) * 128 * SCALE]
            for x, y in p.exterior.coords] for p in polys]
    chips, labels = [], []
    for t in range(-1, k + 1):
        y = int(r.year) + t
        chips.append(f"{prefix}/{r.sid}/k{t}_{y}.png" if (base / "chips" / str(r.sid) / f"k{t}_{y}.png").exists() else None)
        labels.append(f"year {t:+d}")
    return {"name": f"{prefix}_{r.sid}_k{k}.jpg", "chips": chips, "labels": labels, "polys": pts}


def sims_items(base, prefix, off, w, model, jobs, pkg, extra, limit=None):
    """k = 4 prompts (4 option orders) of every cell of one Sims set; copies the chips. Reads no label column."""
    ev = pd.read_csv(base / "events.csv", usecols=["sid", "year", "Latitude", "Longitude"])
    ev = ev.head(limit) if limit else ev
    geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((base / "polygons.geojson").read_text())["features"]}
    swide = series_wide(pd.read_csv(base / "series_gee.csv"))
    sstat = pd.read_csv(base / "static.csv").set_index("sid")
    (pkg / "chips" / prefix).mkdir(parents=True, exist_ok=True)
    for r in ev.itertuples():
        src = base / "chips" / str(r.sid)
        if src.exists():
            shutil.copytree(src, pkg / "chips" / prefix / str(r.sid), dirs_exist_ok=True)
        j = sims_job(base, prefix, r, 4, geoms[r.sid])
        jobs.append(j)
        facts = e2_facts(swide.loc[r.sid].to_dict() if r.sid in swide.index else {}, sstat.loc[r.sid], 4)
        for o, order in enumerate(ORDERS):
            q, _ = question(order)
            w(f"{model}.jsonl", {"id": f"{off + int(r.sid)}|4|{o}", "text": facts + "\n\n" + (extra + "\n\n" if extra else "") + q,
                                 "image": j["name"], "shots": []})
    return len(ev)


def aef_subset(dst, ranges):
    """The AlphaEarth rows whose eid falls in one of `ranges` ([lo, hi) pairs)."""
    z = np.load(V5 / "aef.npz")
    eid, feat = z["eid"], z["feat"]  # read each key once
    keep = np.zeros(len(eid), bool)
    for lo, hi in ranges:
        keep |= (eid >= lo) & (eid < hi)
    np.savez_compressed(dst, eid=eid[keep], feat=feat[keep])
    return int(keep.sum())


def pack_e3(path):
    """v1 CNN input for E3, exactly as fn_v4.pack_e2 (resize to 128 x 128, central 96 px, years -1..4)."""
    from PIL import Image
    se = pd.read_csv(E3 / "events.csv", usecols=["sid", "year"])
    X = np.zeros((len(se), 6, 96, 96, 3), np.uint8)
    for i, r in enumerate(se.itertuples()):
        for j, t in enumerate(range(-1, 5)):
            p = E3 / "chips" / str(r.sid) / f"k{t}_{int(r.year) + t}.png"
            if p.exists():
                with Image.open(p) as im:
                    X[i, j] = np.asarray(im.convert("RGB").resize((128, 128), Image.BILINEAR))[16:112, 16:112]
    np.savez_compressed(path, X=X, y=np.full(len(se), -1), part=np.array(["ext"] * len(se)), eid=(OFF3 + se.sid).to_numpy())


def build(pkg: Path, phase: str, smoke: bool, stage: bool = False):
    (pkg / "code").mkdir(parents=True)
    for d in ("chips", "out", "weights", "adapters"):
        (pkg / d).mkdir()
    for f in CODE[phase if phase in ("cnn_e3", "seeds") else "vlm"]:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    if phase == "cnn_e3":
        pack_e3(pkg / "chips_e3.npz")
        for s_ in (0, 1, 2):
            shutil.copy(V3W / f"M2_s{s_}.pt", pkg / f"weights/M2_s{s_}.pt")
        print("package server2/cnn_e3 ready", flush=True)
        return
    jobs, files = [], {}

    def w(name, row):
        if name not in files:
            files[name] = open(pkg / name, "w", encoding="utf-8")
        files[name].write(json.dumps(row) + "\n")
    if phase == "seeds":  # design notes (post hoc): the v1 M3c recipe with other seeds
        fn = fn_events()
        wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
        static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
        for r in fn[fn.part.isin(["train", "val_sel"])].itertuples():
            if smoke and r.Index % 60:
                continue
            src = INTERIM / "chips" / str(r.eid)
            if src.exists():
                shutil.copytree(src, pkg / "chips" / str(r.eid), dirs_exist_ok=True)
            for k in range(5):
                jobs.append(job_for(r, k, load_region(r.example_path), INTERIM / "chips"))
                facts = fact_sheet(r, wide.loc[r.eid].to_dict() if r.eid in wide.index else {}, static.loc[r.eid], k)
                qs = [question(o) for o in ORDERS]
                if r.part == "train":  # as m3c_qlora.py (v1): facts + question, 4 option orders
                    w("m3c_train.jsonl", {"eid": int(r.eid), "k": k, "image": f"{r.eid}_k{k}.jpg",
                                          "texts": [facts + "\n\n" + q for q, _ in qs],
                                          "answers": [next(L for L, lab in mp.items() if lab == r.merged_label) for _, mp in qs]})
                else:
                    for o, (q, _) in enumerate(qs):
                        w("m3c_valsel.jsonl", {"id": f"{r.eid}|{k}|{o}", "text": facts + "\n\n" + q, "image": f"{r.eid}_k{k}.jpg",
                                               "shots": []})
        vs = fn[fn.part == "val_sel"]
        pd.DataFrame({"eid": vs.eid, "y": vs.merged_label.map(LABELS4.index)}).to_csv(pkg / "meta_valsel.csv", index=False)
        n2 = sims_items(E2, "sims2", OFF2, w, "m3c_e2", jobs, pkg, None, 6 if smoke else None)
        n3 = sims_items(E3, "sims3", OFF3, w, "m3c_e3", jobs, pkg, None, 6 if smoke else None)
        print(f"package server1/seeds: E2 {n2} cells, E3 {n3} cells", flush=True)
    elif phase == "train":
        fn = fn_events()
        tr = fn[fn.part == "train"]
        if smoke:
            tr = tr.head(24)
        wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
        static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
        for r in tr.itertuples():
            src = INTERIM / "chips" / str(r.eid)
            if src.exists():
                shutil.copytree(src, pkg / "chips" / str(r.eid), dirs_exist_ok=True)
            for k in range(5):
                jobs.append(job_for(r, k, load_region(r.example_path), INTERIM / "chips"))
                facts = fact_sheet(r, wide.loc[r.eid].to_dict() if r.eid in wide.index else {}, static.loc[r.eid], k)
                qs = [question(o) for o in ORDERS]
                w("m6_train.jsonl", {"eid": int(r.eid), "k": k, "image": f"{r.eid}_k{k}.jpg", "ph": PH6,
                                     "texts": [facts + "\n\n" + PH6 + "\n\n" + q for q, _ in qs],
                                     "answers": [next(L for L, lab in mp.items() if lab == r.merged_label) for _, mp in qs]})
        n = sims_items(E2, "sims2", OFF2, w, "m6_dev", jobs, pkg, PH6, 6 if smoke else None)
        e2 = pd.read_csv(E2 / "events.csv", usecols=["sid", "label_agri"])  # development labels (E2), for the choice only
        e2 = e2.head(6) if smoke else e2
        pd.DataFrame({"eid": OFF2 + e2.sid, "y": e2.label_agri}).to_csv(pkg / "meta_e2.csv", index=False)
        nf = aef_subset(pkg / "aef.npz", [(0, OFF2), (OFF2, OFF3)])
        print(f"package server1/train: ForestNet train {len(tr)} events, E2 {n} cells, AEF rows {nf}", flush=True)
    else:
        n = sims_items(E3, "sims3", OFF3, w, "m6_e3", jobs, pkg, PH6, 6 if smoke else None)
        sims_items(E3, "sims3", OFF3, w, "m3c_e3", [], pkg, None, 6 if smoke else None)
        if stage:  # staged before the E2 choice: the train batch copies the chosen adapter in on the server
            which = "M6 adapter added later on the server"
        else:
            ch = json.loads((RUNS5 / "raw/train/out/m6_choice.json").read_text())
            shutil.copytree(RUNS5 / f"raw/train/adapters/m6_{ch['tag']}/epoch{ch['epoch']}", pkg / "adapters/m6")
            which = f"M6 = {ch['tag']} epoch {ch['epoch']}"
        shutil.copytree(M3C_ADAPTER, pkg / "adapters/m3c")
        nf = aef_subset(pkg / "aef.npz", [(OFF3, OFF3 + 100000)])
        print(f"package server1/test: E3 {n} cells, {which}, AEF rows {nf}", flush=True)
    for f in files.values():
        f.close()
    (pkg / "jobs.json").write_text(json.dumps(jobs))


def queued(lab, gpus, workdir, cmd):
    """cmd wrapped in the server1 GPU queue for the first GPU of `gpus` (installed from tools/gpu_queue.sh if missing)."""
    if lab.ssh(f"test -x {QUEUE} && echo ok || echo no").strip() != "ok":
        lab.ssh(f"mkdir -p {Path(QUEUE).parent.as_posix()}")
        lab.put(ROOT / "tools/gpu_queue.sh", QUEUE)
        lab.ssh(f"chmod 700 {QUEUE}")
    return f"{QUEUE} {gpus.split(',')[0]} {workdir} {cmd}"


STAGE = RUNS5 / "test_stage.json"  # the E3 batch uploaded ahead of time (run id on server1)


def stage(a):
    """Upload and unpack the E3 inputs now (no model runs, no labels), so the test batch can start as soon as a GPU is
    free after the E2 choice is hashed."""
    lab, H = Lab("lab"), HOSTS["server1"]
    R, run_id, ok = H["remote"], f"v5test_{time.strftime('%Y%m%d%H%M%S')}", False
    try:
        lab.ssh(f"mkdir -p {R} && chmod 700 {R}")
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / run_id
            build(pkg, "test", a.smoke, stage=True)
            tar = Path(td) / f"{run_id}.tar"
            with tarfile.open(tar, "w") as t:
                t.add(pkg, arcname=run_id)
            print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
            lab.put_chunked(tar, f"{R}/{run_id}.tar", chunk_mb=4, workers=6)
        if lab.run_detached(R, f"tar xf {run_id}.tar && rm {run_id}.tar", f"{R}/{run_id}.unpack", poll_s=15) != 0:
            raise RuntimeError("unpack failed")
        STAGE.write_text(json.dumps({"run_id": run_id, "staged": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        ok = True
        print(f"staged on server1: {R}/{run_id}", flush=True)
    finally:
        if not ok:
            lab.retry(lambda: lab.ssh(f"rm -rf {R}/{run_id} {R}/{run_id}.tar* {R}/{run_id}.unpack*"), "clean", tries=40)
        lab.close()


def run(a):
    host = "server2" if a.phase == "cnn_e3" else "server1"
    H = HOSTS[host]
    lab = Lab("lab" if host == "server1" else host)
    run_id = a.run_id or f"v5{a.phase}{'smoke' if a.smoke else ''}_{time.strftime('%Y%m%d%H%M%S')}"
    R = H["remote"]
    keep = False
    raw = RUNS5 / "raw" / (a.phase + ("_smoke" if a.smoke else ""))
    try:
        if a.run_id:  # resume: the batch is still on the server; stale markers go unless its job is still running
            keep = True  # never delete a batch we did not upload in this call before its results are back
            if a.phase == "test" and lab.ssh(f"test -d {R}/{run_id}/adapters/m6 && echo ok || echo no").strip() != "ok":
                raise RuntimeError(f"{run_id} has no M6 adapter yet (the train batch copies the chosen one in)")
            alive = lab.ssh(ALIVE.format(R=R, run_id=run_id)).strip() == "yes"
            if not alive:
                lab.ssh(f"rm -f {R}/{run_id}.exit {R}/{run_id}.exit.pid {R}/{run_id}_out.tar")
        else:
            lab.ssh(f"mkdir -p {R} && chmod 700 {R}")
            with tempfile.TemporaryDirectory() as td:
                pkg = Path(td) / run_id
                build(pkg, a.phase, a.smoke)
                tar = Path(td) / f"{run_id}.tar"
                with tarfile.open(tar, "w") as t:
                    t.add(pkg, arcname=run_id)
                print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
                lab.put_chunked(tar, f"{R}/{run_id}.tar", chunk_mb=64 if host != "server1" else 4, workers=6)
            if lab.run_detached(R, f"tar xf {run_id}.tar && rm {run_id}.tar", f"{R}/{run_id}.unpack", poll_s=15) != 0:
                raise RuntimeError("unpack failed")
        pipe = f"{H['py']} {R}/{run_id}/code/pipeline_v5.py --phase {a.phase} --gpus {a.gpus} --seeds {a.seeds}"
        if host == "server1":
            pipe = queued(lab, a.gpus, f"{R}/{run_id}", pipe)
        code = lab.run_detached(R, f"{pipe} >> {R}/{run_id}/pipeline_log.txt 2>&1", f"{R}/{run_id}.exit", poll_s=120,
                                timeout_h=a.hours)
        keep = True  # from here on the server copy is kept unless the results are safely back and the pipeline passed
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -3 {R}/{run_id}/pipeline_log.txt"), "log"), flush=True)
        pull = "out logs state pipeline_log.txt"
        st = json.loads(STAGE.read_text())["run_id"] if STAGE.exists() and not a.smoke else None
        copied = False
        if a.phase == "train" and code == 0 and st:  # the chosen adapter goes straight into the staged E3 batch
            try:
                print(lab.ssh(f"cd {R}/{run_id} && p=$(cat out/m6_choice_path.txt) && test -d {R}/{st} && "
                              f"rm -rf {R}/{st}/adapters/m6 && cp -r $p {R}/{st}/adapters/m6 && echo copied $p to {st}").strip(),
                      "- it comes back with the E3 results", flush=True)
                copied = True
            except RuntimeError as e:
                print(f"copy into the staged E3 batch failed ({str(e)[:80]}); pulling the adapter instead", flush=True)
        if a.phase == "train" and not copied:  # only the chosen candidate's adapter comes back (pipeline_v5.py)
            pull += " $(cat out/m6_choice_path.txt 2>/dev/null)"
        if a.phase == "test":  # 25/10/2026: the E3 batch of v5 left this out and the chosen adapter was lost with it
            pull += " adapters/m6"
        if a.phase == "seeds":  # the chosen epoch of every seed (paths written by pipeline_v5.py)
            pull += " $(cat out/m3c_seed_paths.txt 2>/dev/null)"
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


def k4_probs(path):
    """4-class probabilities at k = 4: softmax over the letters, mapped back to classes, mean over the 4 option orders."""
    acc, cnt = {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e, k, o = map(int, r["id"].split("|"))
        assert k == 4
        lp = np.array([r["logprobs"][L] for L in OPTIONS])
        p = np.exp(lp - np.max(lp)) if np.isfinite(lp).any() else np.full(4, 0.25)
        p = p / p.sum()
        _, mapping = question(ORDERS[o])
        v = acc.setdefault(e, np.zeros(4))
        for L, pr in zip(OPTIONS, p):
            v[LABELS4.index(mapping[L])] += pr
        cnt[e] = cnt.get(e, 0) + 1
    if any(c != 4 for c in cnt.values()):
        raise RuntimeError(f"{path}: events without 4 answers")
    eids = sorted(acc)
    return np.array(eids), np.stack([acc[e] / 4 for e in eids])


def convert():
    out = RUNS5 / "probs"
    out.mkdir(parents=True, exist_ok=True)
    tr, te, cn = (RUNS5 / "raw" / n for n in ("train", "test", "cnn_e3"))
    for name, path in [(f"M6_{t}_ep{e}_E2", tr / f"out/m6_{t}_dev_ep{e}.jsonl") for t in ("p0", "p05") for e in (1, 2)] + \
                      [("M6_E3", te / "out/m6_e3.jsonl"), ("M3c_E3", te / "out/m3c_e3.jsonl")]:
        if not path.exists():
            print(name, "missing")
            continue
        eids, P = k4_probs(path)
        np.savez_compressed(out / f"{name}.npz", probs=P, eid=eids)
        print("saved", name, P.shape)
    zs = [np.load(f, allow_pickle=True) for f in sorted(cn.glob("out/M2e3_s*.npz"))]
    if zs:
        np.savez_compressed(out / "M2_E3.npz", probs=np.mean([z["probs"] for z in zs], 0)[:, 4], eid=zs[0]["eid"])
        print("saved M2_E3", len(zs), "seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "stage", "convert"])
    ap.add_argument("--phase", choices=["train", "test", "cnn_e3", "seeds"])
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hours", type=float, default=30)
    ap.add_argument("--run-id")
    a = ap.parse_args()
    {"run": lambda: run(a), "stage": lambda: stage(a), "convert": convert}[a.cmd]()


if __name__ == "__main__":
    main()
