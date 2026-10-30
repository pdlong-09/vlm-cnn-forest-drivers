"""design notes GPU batch: pack -> upload to /data/tmp (NVMe) on server1 -> server/pipeline_v3.py -> pull -> delete.

    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v3_gpu.py --phase train [--smoke] [--resume <run>]
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v3_gpu.py --phase ext --train-run <run> [--smoke]

Two batches (design notes note in §7): phase train runs on ForestNet only while the Sims chips are fetched; phase ext
brings E, the frozen v1 M3c adapter, and the M2 weights and chosen M3v adapter pulled back from phase train.

Package: ForestNet chips (every event, v1 names) and the Sims chips (chips/sims/<sid>/), v1 card render jobs (ForestNet
cards exactly as v1; Sims cards with the polygon mapped from lon/lat onto the same 3 km chip), v1-format prompts
(train.jsonl with labels for the training part; valsel / valcaltest / ext without labels), meta_valsel.csv (val_sel
labels for the epoch choice), meta_cnn.csv, the frozen v1 M3c adapter and the code. Sims events get eid 100000 + sid.
Results: runs/rs_forestnet_v3/raw/<run>/ -> runs/rs_forestnet_v3/probs/{M2,M3c,M3v}.npz.
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
from rsdrivers.card import OPTIONS, fact_sheet, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, events, load_region  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402
from rsdrivers.render import SCALE, job_for  # noqa: E402

SIMS = ROOT / "data/interim/rs_forestnet_sims"
RUNS3 = ROOT / "runs/rs_forestnet_v3"
REMOTE = "/data/tmp/llm4deforest_paperA_v3"
PY = "/home/user/llm4deforest/paperA/venv/bin/python"
M3C_ADAPTER = ROOT / "runs/rs_forestnet/adapters/M3c_20261022015339/adapters/epoch2"
ORDERS = [LABELS4[i:] + LABELS4[:i] for i in range(4)]
OFFSET = 100000
CODE = ["server/cnn_v3.py", "server/cnn_train.py", "server/vlm_infer.py", "server/render_cards.py", "rsdrivers/render.py",
        "server/gpu_pipeline.py", "server/pipeline_v3.py", "server/fn_pack_v3.py", "server/fn_qlora_v3.py",
        "server/vlm_infer_v3.py", "server/check_tower_lora.py"]


def sims_job(r, k, poly):
    """v1 card job for a Sims cell: chip of 3 km centred on the cell (rsdrivers.gee._box), polygon in scaled pixels."""
    dlat = 1500 / 110574.0
    dlon = 1500 / (111320.0 * math.cos(math.radians(r.Latitude)))
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    pts = [[[(x - r.Longitude + dlon) / (2 * dlon) * 128 * SCALE, (r.Latitude + dlat - y) / (2 * dlat) * 128 * SCALE]
            for x, y in p.exterior.coords] for p in polys]
    chips, labels = [], []
    for t in range(-1, k + 1):
        y = int(r.year) + t
        rel = f"sims/{r.sid}/k{t}_{y}.png"
        chips.append(rel if (SIMS / "chips" / str(r.sid) / f"k{t}_{y}.png").exists() else None)
        labels.append(f"year {t:+d}")
    return {"name": f"s{r.sid}_k{k}.jpg", "chips": chips, "labels": labels, "polys": pts}


def sims_facts(wide_row, st, k):
    """v1 fact sheet; the peat map ForestNet used is not public, so the line says unknown (design notes)."""
    s = {"area_ha": st["area_ha"], "n_parts": st["n_parts"], "tc2000_in": st["tc2000_in"], "peat": 0,
         "srtm0_mean": st["srtm0_mean"]}
    text = fact_sheet(None, wide_row, s, k)
    assert "Peat soil: no." in text
    return text.replace("Peat soil: no.", "Peat soil: unknown.")


def build(pkg: Path, smoke: bool, phase: str, train_raw: Path | None = None):
    for d in ("chips/sims", "code", "adapters", "weights", "out"):
        (pkg / d).mkdir(parents=True, exist_ok=True)
    for f in CODE:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    se = pd.read_csv(SIMS / "events.csv")
    if phase == "train":
        se = se.iloc[0:0]
    else:
        fn = fn.iloc[0:0]
        shutil.copytree(M3C_ADAPTER, pkg / "adapters/m3c")
        choice = json.loads((train_raw / "out/m3v_choice.json").read_text())
        shutil.copy(train_raw / "out/m3v_choice.json", pkg / "out/m3v_choice.json")
        shutil.copytree(train_raw / f"adapters/m3v/epoch{choice['epoch']}", pkg / f"adapters/m3v/epoch{choice['epoch']}")
        for w in sorted((train_raw / "weights").glob("M2_s*.pt")):
            shutil.copy(w, pkg / "weights" / w.name)
    if smoke:
        if len(fn):
            fn = pd.concat([g.head(12 if p == "train" else 4) for p, g in fn.groupby("part")])
        se = se.head(6)
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    swide = series_wide(pd.read_csv(SIMS / "series_gee.csv"))
    sstat = pd.read_csv(SIMS / "static.csv").set_index("sid")
    geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((SIMS / "polygons.geojson").read_text())["features"]}
    jobs, files = [], {n: open(pkg / f"{n}.jsonl", "w", encoding="utf-8") for n in ("train", "valsel", "valcaltest", "ext")}
    cnn = []
    for r in fn.itertuples():
        src = INTERIM / "chips" / str(r.eid)
        if src.exists():
            shutil.copytree(src, pkg / "chips" / str(r.eid), dirs_exist_ok=True)
        y_idx = LABELS4.index(r.merged_label)
        cnn.append({"eid": r.eid, "part": r.part, "y": y_idx, "year": int(r.year), "chipdir": str(r.eid)})
        for k in range(5):
            jobs.append(job_for(r, k, load_region(r.example_path), INTERIM / "chips"))
            facts = fact_sheet(r, wide.loc[r.eid].to_dict() if r.eid in wide.index else {}, static.loc[r.eid], k)
            name = f"{r.eid}_k{k}.jpg"
            qs = [question(o) for o in ORDERS]
            if r.part == "train":
                files["train"].write(json.dumps({"eid": int(r.eid), "k": k, "image": name, "label": y_idx,
                                                 "texts": [facts + "\n\n" + q for q, _ in qs],
                                                 "answers": [next(L for L, lab in mp.items() if lab == r.merged_label)
                                                             for _, mp in qs]}) + "\n")
            else:
                target = "valsel" if r.part == "val_sel" else "valcaltest"
                for o, (q, _) in enumerate(qs):
                    files[target].write(json.dumps({"id": f"{r.eid}|{k}|{o}", "text": facts + "\n\n" + q, "image": name,
                                                    "shots": []}) + "\n")
    for r in se.itertuples():
        src = SIMS / "chips" / str(r.sid)
        if src.exists():
            shutil.copytree(src, pkg / "chips/sims" / str(r.sid), dirs_exist_ok=True)
        eid = OFFSET + int(r.sid)
        cnn.append({"eid": eid, "part": "ext", "y": -1, "year": int(r.year), "chipdir": f"sims/{r.sid}"})
        w = swide.loc[r.sid].to_dict() if r.sid in swide.index else {}
        for k in range(5):
            j = sims_job(r, k, geoms[r.sid])
            jobs.append(j)
            facts = sims_facts(w, sstat.loc[r.sid], k)
            for o, order in enumerate(ORDERS):
                q, _ = question(order)
                files["ext"].write(json.dumps({"id": f"{eid}|{k}|{o}", "text": facts + "\n\n" + q, "image": j["name"],
                                               "shots": []}) + "\n")
    for f in files.values():
        f.close()
    (pkg / "jobs.json").write_text(json.dumps(jobs))
    pd.DataFrame(cnn).to_csv(pkg / "meta_cnn.csv", index=False)
    if phase == "train":
        pd.DataFrame([c for c in cnn if c["part"] == "val_sel"])[["eid", "y"]].to_csv(pkg / "meta_valsel.csv", index=False)
    print(f"package ({phase}): ForestNet {len(fn)} events, Sims {len(se)} cells, {len(jobs)} cards", flush=True)


def letters_to_probs(rows):
    acc, cnt = {}, {}
    for r in rows:
        e, k, o = map(int, r["id"].split("|"))
        lp = np.array([r["logprobs"][L] for L in OPTIONS])
        p = np.exp(lp - np.max(lp)) if np.isfinite(lp).any() else np.full(4, 0.25)
        p = p / p.sum()
        v = acc.setdefault(e, np.zeros((5, 4)))
        c = cnt.setdefault(e, np.zeros(5))
        _, mapping = question(ORDERS[o])
        for L, pr in zip(OPTIONS, p):
            v[k, LABELS4.index(mapping[L])] += pr
        c[k] += 1
    if any((c != 4).any() for c in cnt.values()):
        raise RuntimeError("event-steps without 4 answers")
    eids = sorted(acc)
    return np.array(eids), np.stack([acc[e] / cnt[e][:, None] for e in eids])


def convert(train_raw: Path, ext_raw: Path):
    """Both batches -> runs/rs_forestnet_v3/probs: M2 (mean of the seeds; ForestNet from train, E from ext), M3c (E),
    M3v (val_sel of the chosen epoch, ForestNet val_cal + test, E)."""
    out = RUNS3 / "probs"
    out.mkdir(parents=True, exist_ok=True)
    tr = [np.load(f, allow_pickle=True) for f in sorted(train_raw.glob("out/M2_s*.npz"))]
    ex = [np.load(f, allow_pickle=True) for f in sorted(ext_raw.glob("out/M2ext_s*.npz"))]
    if len(tr) != len(ex) or not tr:
        raise RuntimeError(f"M2 seeds: {len(tr)} from train, {len(ex)} from ext")
    P = np.concatenate([np.mean([z["probs"] for z in tr], 0), np.mean([z["probs"] for z in ex], 0)])
    np.savez_compressed(out / "M2.npz", probs=P, eid=np.concatenate([tr[0]["eid"], ex[0]["eid"]]),
                        part=np.concatenate([tr[0]["part"], ex[0]["part"]]))
    print("saved M2", P.shape, len(tr), "seeds")
    ep = json.loads((train_raw / "out/m3v_choice.json").read_text())["epoch"]
    for name, paths in (("M3c", [ext_raw / "out/m3c_ext.jsonl"]),
                        ("M3v", [ext_raw / "out/m3v_ext.jsonl", train_raw / "out/m3v_valcaltest.jsonl",
                                 train_raw / f"out/m3v_valsel_ep{ep}.jsonl"])):
        rows = [json.loads(l) for p in paths for l in open(p, encoding="utf-8")]
        eids, Pm = letters_to_probs(rows)
        np.savez_compressed(out / f"{name}.npz", probs=Pm, eid=eids)
        print("saved", name, Pm.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "ext"], required=True)
    ap.add_argument("--train-run", help="phase ext: the finished phase-train run (its results in runs/rs_forestnet_v3/raw)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--hours", type=float, default=48)
    a = ap.parse_args()
    run = a.resume or f"v3{a.phase}{'smoke' if a.smoke else ''}_{time.strftime('%Y%m%d%H%M%S')}"
    raw = RUNS3 / "raw" / run
    train_raw = RUNS3 / "raw" / a.train_run if a.train_run else None
    if a.phase == "ext" and not (train_raw and (train_raw / "out/m3v_choice.json").exists()):
        raise SystemExit("phase ext needs --train-run with finished results")
    lab, keep = Lab(), False
    try:
        if not a.resume:
            with tempfile.TemporaryDirectory() as td:
                pkg = Path(td) / run
                build(pkg, a.smoke, a.phase, train_raw)
                tar = Path(td) / f"{run}.tar"
                with tarfile.open(tar, "w") as t:
                    t.add(pkg, arcname=run)
                print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
                lab.put_chunked(tar, f"{REMOTE}/{run}.tar")
            if lab.run_detached(REMOTE, f"tar xf {run}.tar && rm {run}.tar && chmod -R go-rwx {run}",
                                f"{REMOTE}/{run}.unpack", poll_s=20) != 0:
                raise RuntimeError("unpack failed")
        else:
            for f in CODE:
                lab.put(ROOT / "paper/rs_forestnet" / f, f"{REMOTE}/{run}/code/{Path(f).name}")
        code = lab.run_detached(REMOTE, f"{PY} {run}/code/pipeline_v3.py --phase {a.phase} --gpus 0,1 "
                                        f"{'--smoke' if a.smoke else ''} "
                                        f"> {run}/pipeline_log.txt 2>&1", f"{REMOTE}/{run}.exit{time.strftime('%H%M%S')}",
                                poll_s=300, timeout_h=a.hours)
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -4 {REMOTE}/{run}/pipeline_log.txt"), "log"), flush=True)
        lab.retry(lambda: lab.ssh(
            # phase train also returns the chosen M3v adapter and the M2 weights (needed by phase ext)
            f"cd {REMOTE}/{run} && A=''; [ {a.phase} = train ] && [ -f out/m3v_choice.json ] && "
            f"A=\"adapters/m3v/epoch$(grep -o '\"epoch\": [0-9]' out/m3v_choice.json | grep -o '[0-9]$') weights\"; "
            f"tar cf ../{run}_out.tar out logs state pipeline_log.txt $A"),
            "pack results")
        raw.mkdir(parents=True, exist_ok=True)
        lab.get(f"{REMOTE}/{run}_out.tar", raw / "out.tar", timeout=3600)
        with tarfile.open(raw / "out.tar") as t:
            t.extractall(raw, filter="data")
        if code != 0:
            keep = True
            raise RuntimeError(f"pipeline failed (exit {code}); data kept, fix and --resume {run}")
    finally:
        if not keep:
            left = lab.retry(lambda: lab.ssh(f"rm -rf {REMOTE}/{run} {REMOTE}/{run}.tar* {REMOTE}/{run}_out.tar "
                                             f"{REMOTE}/{run}.exit* {REMOTE}/{run}.unpack*; ls -A {REMOTE} | wc -l"), "clean")
            print("server data left in", REMOTE, ":", left.strip(), flush=True)
        lab.close()
    if a.phase == "ext" and not a.smoke:
        convert(train_raw, raw)


if __name__ == "__main__":
    main()
