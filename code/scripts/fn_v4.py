"""design notes driver: packages, batches on server2 and server1, conversion. Every batch: upload -> run -> pull -> delete.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v4.py folds
    [LAB_PW=...] .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v4.py run --host server1|server2 --phase cnn|cnn_e2 [--smoke]
    LAB_PW=... .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v4.py run --host server1 --phase train|ext [--smoke]
    .venv_rs/Scripts/python paper/rs_forestnet/scripts/fn_v4.py convert

Results land in runs/rs_forestnet_v4/raw/<host>_<phase>/ (the latest finished batch of each kind is used downstream).
E2 events get eid 200000 + sid. Expert text and retrieval use the CNN outputs of the server2 "cnn" batch.
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
from rsdrivers.card import OPTIONS, fact_sheet, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, events, load_region  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402
from rsdrivers.render import SCALE, job_for  # noqa: E402

E2 = ROOT / "data/interim/rs_forestnet_sims2"
V4 = ROOT / "data/interim/rs_forestnet_v4"
RUNS4 = ROOT / "runs/rs_forestnet_v4"
V3W = ROOT / "runs/rs_forestnet_v3/raw/v3train_20261023153551/weights"
M3C_ADAPTER = ROOT / "runs/rs_forestnet/adapters/M3c_20261022015339/adapters/epoch2"
OFF = 200000
ORDERS = [LABELS4[i:] + LABELS4[:i] for i in range(4)]
NAMES = {"Plantation": "large-scale plantation", "Smallholder agriculture": "smallholder agriculture",
         "Grassland shrubland": "grassland or shrubland", "Other": "other"}
PH_TEXT = "Remote-sensing encoder features (from a CNN trained on Landsat chips of other events): " + "<|fim_pad|>" * 4
HOSTS = {"server2": {"remote": "/home/user/llm4deforest_v4/runs", "py": "/home/user/llm4deforest_v4/venv/bin/python",
                        "gpus": "0"},
         "server1": {"remote": "/data/tmp/llm4deforest_paperA_v4", "py": "/home/user/llm4deforest/paperA/venv/bin/python",
                    "gpus": "0,1"}}
CODE = {"cnn": ["server/cnn_train.py", "server/cnn_v3.py", "server/fn_v4_cnn.py", "server/fn_v4_merge.py",
                "server/gpu_pipeline.py", "server/pipeline_v4.py"],
        "vlm": ["server/qlora_train.py", "server/vlm_infer.py", "server/fn_m5e.py", "server/fn_qlora_v3.py",
                   "server/gpu_pipeline.py", "server/pipeline_v4.py", "server/render_cards.py", "rsdrivers/render.py"]}


# ---------- inputs ----------
def fn_events():
    return events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")


def e2_events():
    return pd.read_csv(E2 / "events.csv")


def e2_job(r, k, poly):
    dlat = 1500 / 110574.0
    dlon = 1500 / (111320.0 * math.cos(math.radians(r.Latitude)))
    polys = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
    pts = [[[(x - r.Longitude + dlon) / (2 * dlon) * 128 * SCALE, (r.Latitude + dlat - y) / (2 * dlat) * 128 * SCALE]
            for x, y in p.exterior.coords] for p in polys]
    chips, labels = [], []
    for t in range(-1, k + 1):
        y = int(r.year) + t
        chips.append(f"sims2/{r.sid}/k{t}_{y}.png" if (E2 / "chips" / str(r.sid) / f"k{t}_{y}.png").exists() else None)
        labels.append(f"year {t:+d}")
    return {"name": f"t{r.sid}_k{k}.jpg", "chips": chips, "labels": labels, "polys": pts}


def e2_facts(wide_row, st, k):
    s = {"area_ha": st["area_ha"], "n_parts": st["n_parts"], "tc2000_in": st["tc2000_in"], "peat": 0, "srtm0_mean": st["srtm0_mean"]}
    text = fact_sheet(None, wide_row, s, k)
    return text.replace("Peat soil: no.", "Peat soil: unknown.")


def pack_e2(path):
    """v1 CNN input for E2: years -1..4, central 96 px of each 128 px chip.

    Chips away from the equator come back from GEE 128 px wide but only 128·cos(lat) px tall (the 3 km box is square on
    the ground, not in degrees; 23/10/2026: 4344 of 9000 E2 chips, 119-127 px). They are resized to 128 x 128 first, as
    render.py does for the cards, so the crop is centred and the ground is square like the ForestNet chips."""
    from PIL import Image
    se = e2_events()
    X = np.zeros((len(se), 6, 96, 96, 3), np.uint8)
    for i, r in enumerate(se.itertuples()):
        for j, t in enumerate(range(-1, 5)):
            p = E2 / "chips" / str(r.sid) / f"k{t}_{int(r.year) + t}.png"
            if p.exists():
                with Image.open(p) as im:
                    X[i, j] = np.asarray(im.convert("RGB").resize((128, 128), Image.BILINEAR))[16:112, 16:112]
    np.savez_compressed(path, X=X, y=np.full(len(se), -1), part=np.array(["ext"] * len(se)), eid=(OFF + se.sid).to_numpy())


# ---------- CNN outputs (from the server2 cnn batch) ----------
def cnn_outputs():
    raw = RUNS4 / "raw/cnn"
    full = np.load(raw / "cnnv4/full_fn.npz", allow_pickle=True)
    oof = np.load(raw / "cnnv4/oof_feats.npz", allow_pickle=True)
    prob = {int(e): p for e, p in zip(full["eid"], full["prob"])}
    prob.update({int(e): p for e, p in zip(oof["eid"], oof["prob"])})  # training events: out of fold
    fe2 = RUNS4 / "raw/cnn_e2/cnnv4/full_e2.npz"
    if fe2.exists():
        z = np.load(fe2, allow_pickle=True)
        prob.update({int(e): p for e, p in zip(z["eid"], z["prob"])})
    return raw, full, prob


def neighbours(raw, full, fn):
    """Labels of the 4 most similar training events per (event, k) in the matching feature space (design notes)."""
    from sklearn.neighbors import BallTree
    tr = fn[fn.part == "train"].set_index("eid")
    lab = tr.merged_label.to_dict()
    fold = pd.read_csv(V4 / "folds.csv").set_index("eid").fold.to_dict()
    ll = np.radians(tr[["latitude", "longitude"]].to_numpy())
    near = BallTree(ll, metric="haversine").query_radius(ll, 10 / 6371.0)
    close = {e: set(tr.index[i]) for e, i in zip(tr.index, near)}

    def norm(F):
        return F / (np.linalg.norm(F, axis=-1, keepdims=True) + 1e-8)
    out = {}
    fpos = {int(e): i for i, e in enumerate(full["eid"])}
    tr_ids = np.array(tr.index)
    F_full = full["feat"]  # read once: every access to an NpzFile key decompresses the whole array again
    Ftr = norm(F_full[[fpos[e] for e in tr_ids]])
    queries = {int(e): F_full[i].copy() for i, e in enumerate(full["eid"]) if int(e) not in lab}
    fe2 = RUNS4 / "raw/cnn_e2/cnnv4/full_e2.npz"
    if fe2.exists():
        z = np.load(fe2, allow_pickle=True)
        queries.update({int(e): f for e, f in zip(z["eid"], z["feat"])})
    for e, f in queries.items():  # non-training events: seed-0 CNN, index = all training events
        q = norm(f)
        out[e] = [[lab[tr_ids[j]] for j in np.argsort(-(Ftr[:, k] @ q[k]))[:4]] for k in range(5)]
    for f_ in range(5):  # training events of fold f: fold-f CNN, index = training events of the other folds
        z = np.load(raw / f"cnnv4/fold{f_}_trainfeats.npz", allow_pickle=True)
        ids, F = z["eid"].astype(int), norm(z["feat"])
        is_q = np.array([fold[e] == f_ for e in ids])
        idx_ids, Fi = ids[~is_q], F[~is_q]
        for qi in np.flatnonzero(is_q):
            e = ids[qi]
            ok = np.array([i not in close[e] and i != e for i in idx_ids])
            res = []
            for k in range(5):
                s = Fi[:, k] @ F[qi, k]
                s[~ok] = -np.inf
                res.append([lab[idx_ids[j]] for j in np.argsort(-s)[:4]])
            out[e] = res
    return out


def expert_text(p):
    return ("A remote-sensing CNN trained on Landsat chips of other events estimates: " +
            ", ".join(f"{NAMES[l]} {p[i]:.2f}" for i, l in enumerate(LABELS4)) + ".")


def neigh_text(n):
    return "The four most similar training events (by that CNN's image features) were: " + ", ".join(NAMES[l] for l in n) + "."


# ---------- packages ----------
def feats_with_e2(dst: Path):
    """feats_v4.npz of the cnn batch (out-of-fold on train, seed 0 elsewhere) plus the seed-0 E2 features if present."""
    z = dict(np.load(RUNS4 / "raw/cnn/feats_v4.npz", allow_pickle=True))
    fe2 = RUNS4 / "raw/cnn_e2/cnnv4/full_e2.npz"
    if fe2.exists():
        b = np.load(fe2, allow_pickle=True)
        z = {"eid": np.concatenate([z["eid"], b["eid"]]), "feat": np.concatenate([z["feat"], b["feat"]]),
             "prob": np.concatenate([z["prob"], b["prob"]]), "source": np.concatenate([z["source"], ["full"] * len(b["eid"])])}
    np.savez_compressed(dst, **z)


def build(pkg: Path, host: str, phase: str, smoke: bool):
    (pkg / "code").mkdir(parents=True)
    for d in ("chips", "out", "weights", "adapters"):
        (pkg / d).mkdir()
    for f in CODE["cnn" if phase in ("cnn", "cnn_e2") else "vlm"]:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    fn = fn_events()
    if smoke:
        fn = pd.concat([g.head(16 if p == "train" else 4) for p, g in fn.groupby("part")])
    se = e2_events().head(6) if smoke else e2_events()
    if phase in ("cnn", "cnn_e2"):  # CNN batches, on either server (design notes)
        if phase == "cnn":
            z = np.load(INTERIM / "chips_packed.npz", allow_pickle=True)
            keep = np.isin(z["eid"], fn.eid.to_numpy())
            np.savez(pkg / "chips_fn.npz", X=z["X"][keep], y=z["y"][keep], part=z["part"][keep], eid=z["eid"][keep])
            shutil.copy(V4 / "folds.csv", pkg / "folds.csv")
            shutil.copy(V3W / "M2_s0.pt", pkg / "weights/M2_s0.pt")
        else:
            pack_e2(pkg / "chips_e2.npz")
            for s_ in (0, 1, 2):
                shutil.copy(V3W / f"M2_s{s_}.pt", pkg / f"weights/M2_s{s_}.pt")
        print(f"package {host}/{phase} ready", flush=True)
        return
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    raw, full, prob = cnn_outputs()
    nb = neighbours(raw, full, fn_events())
    fn_phase = fn if phase == "train" else fn[fn.part == "test"]
    jobs, files = [], {}

    def w(name, row):
        if name not in files:  # open once (setdefault would reopen, i.e. truncate, the file on every call)
            files[name] = open(pkg / name, "w", encoding="utf-8")
        files[name].write(json.dumps(row) + "\n")
    for r in fn_phase.itertuples():
        src = INTERIM / "chips" / str(r.eid)
        if src.exists():
            shutil.copytree(src, pkg / "chips" / str(r.eid), dirs_exist_ok=True)
        for k in range(5):
            jobs.append(job_for(r, k, load_region(r.example_path), INTERIM / "chips"))
            facts = fact_sheet(r, wide.loc[r.eid].to_dict() if r.eid in wide.index else {}, static.loc[r.eid], k)
            name = f"{r.eid}_k{k}.jpg"
            variants = {"m5e": PH_TEXT} if phase == "ext" else {
                "m5e": PH_TEXT, "m5t": expert_text(prob[r.eid][k]), "m5tr": expert_text(prob[r.eid][k]) + " " + neigh_text(nb[r.eid][k])}
            for m, extra in variants.items():
                qs = [question(o) for o in ORDERS]
                texts = [facts + "\n\n" + extra + "\n\n" + q for q, _ in qs]
                if r.part == "train":
                    w(f"{m}_train.jsonl", {"eid": int(r.eid), "k": k, "image": name, "texts": texts,
                                           "answers": [next(L for L, lab in mp.items() if lab == r.merged_label) for _, mp in qs]})
                    continue
                if m == "m5e" and r.part == "test" and phase == "train":
                    continue  # M5e on the ForestNet test is exploratory and runs last, in the ext batch
                tgt = {"val_sel": "valsel", "val_cal": "valcal" if m == "m5e" else "valcaltest",
                       "test": "test" if m == "m5e" else "valcaltest"}[r.part]
                for o in range(4):
                    w(f"{m}_{tgt}.jsonl", {"id": f"{r.eid}|{k}|{o}", "text": texts[o], "image": name, "shots": []})
    if phase == "ext":
        geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((E2 / "polygons.geojson").read_text())["features"]}
        swide = series_wide(pd.read_csv(E2 / "series_gee.csv"))
        sstat = pd.read_csv(E2 / "static.csv").set_index("sid")
        (pkg / "chips/sims2").mkdir()
        for r in se.itertuples():
            src = E2 / "chips" / str(r.sid)
            if src.exists():
                shutil.copytree(src, pkg / "chips/sims2" / str(r.sid), dirs_exist_ok=True)
            eid = OFF + int(r.sid)
            wr = swide.loc[r.sid].to_dict() if r.sid in swide.index else {}
            for k in range(5):
                j = e2_job(r, k, geoms[r.sid])
                jobs.append(j)
                facts = e2_facts(wr, sstat.loc[r.sid], k)
                for m, extra in {"m3c": None, "m5e": PH_TEXT, "m5t": expert_text(prob[eid][k]),
                                 "m5tr": expert_text(prob[eid][k]) + " " + neigh_text(nb[eid][k])}.items():
                    for o, order in enumerate(ORDERS):
                        q, _ = question(order)
                        w(f"{m}_ext.jsonl", {"id": f"{eid}|{k}|{o}", "text": facts + "\n\n" + (extra + "\n\n" if extra else "") + q,
                                             "image": j["name"], "shots": []})
        shutil.copytree(M3C_ADAPTER, pkg / "adapters/m3c")
        tr = RUNS4 / "raw/server1_train"
        for m in ("m5e", "m5t", "m5tr"):
            e = json.loads((tr / f"out/{m}_choice.json").read_text())["epoch"]
            shutil.copy(tr / f"out/{m}_choice.json", pkg / f"out/{m}_choice.json")
            shutil.copytree(tr / f"adapters/{m}/epoch{e}", pkg / f"adapters/{m}/epoch{e}")
    for f in files.values():
        f.close()
    (pkg / "jobs.json").write_text(json.dumps(jobs))
    vs = fn_events()
    vs = vs[vs.part == "val_sel"]
    pd.DataFrame({"eid": vs.eid, "y": vs.merged_label.map(LABELS4.index)}).to_csv(pkg / "meta_valsel.csv", index=False)
    feats_with_e2(pkg / "feats_v4.npz")
    print(f"package {host}/{phase}: ForestNet {len(fn_phase)} events, E2 {len(se) if phase == 'ext' else 0}, {len(jobs)} cards, "
          f"files {sorted(files)}", flush=True)


def run(a):
    H = HOSTS[a.host]
    lab = Lab("lab" if a.host == "server1" else a.host)
    run_id = f"v4{a.host}{a.phase}{'smoke' if a.smoke else ''}_{time.strftime('%Y%m%d%H%M%S')}"
    R = H["remote"]
    keep = False
    kind = a.phase if a.phase in ("cnn", "cnn_e2") else f"{a.host}_{a.phase}"
    raw = RUNS4 / "raw" / (kind + ("_smoke" if a.smoke else ""))
    try:
        lab.ssh(f"mkdir -p {R} && chmod 700 {R}")
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / run_id
            build(pkg, a.host, a.phase, a.smoke)
            tar = Path(td) / f"{run_id}.tar"
            with tarfile.open(tar, "w") as t:
                t.add(pkg, arcname=run_id)
            print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
            lab.put_chunked(tar, f"{R}/{run_id}.tar", chunk_mb=64 if a.host != "server1" else 4)
        if lab.run_detached(R, f"tar xf {run_id}.tar && rm {run_id}.tar", f"{R}/{run_id}.unpack", poll_s=15) != 0:
            raise RuntimeError("unpack failed")
        code = lab.run_detached(R, f"{H['py']} {run_id}/code/pipeline_v4.py --role {a.host} --phase {a.phase} --gpus {H['gpus']} "
                                   f"{'--smoke' if a.smoke else ''} > {run_id}/pipeline_log.txt 2>&1", f"{R}/{run_id}.exit",
                                poll_s=120, timeout_h=a.hours)
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -3 {R}/{run_id}/pipeline_log.txt"), "log"), flush=True)
        pull = "out logs state pipeline_log.txt"
        if a.phase == "cnn":
            pull += " cnnv4/oof_feats.npz cnnv4/full_fn.npz cnnv4/fold*_trainfeats.npz feats_v4.npz"
        if a.phase == "cnn_e2":
            pull += " cnnv4/full_e2.npz"
        lab.retry(lambda: lab.ssh(
            f"cd {R}/{run_id} && A=''; for c in out/*_choice.json; do [ -f $c ] || continue; m=$(basename $c _choice.json); "
            f"e=$(grep -o '\"epoch\": [0-9]' $c | grep -o '[0-9]$'); [ -d adapters/$m/epoch$e ] && [ {a.phase} = train ] && "
            f"A=\"$A adapters/$m/epoch$e\"; done; tar cf ../{run_id}_out.tar {pull} $A"), "pack")
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir(parents=True)
        lab.get(f"{R}/{run_id}_out.tar", raw / "out.tar", timeout=3600)
        with tarfile.open(raw / "out.tar") as t:
            t.extractall(raw, filter="data")
        (raw / "RUN_ID").write_text(run_id)
        if code != 0:
            keep = True
            raise RuntimeError(f"pipeline failed (exit {code}); data kept on {a.host} in {R}/{run_id}")
    finally:
        if not keep:
            left = lab.retry(lambda: lab.ssh(f"rm -rf {R}/{run_id} {R}/{run_id}.tar* {R}/{run_id}_out.tar {R}/{run_id}.exit* "
                                             f"{R}/{run_id}.unpack*; ls -A {R} | wc -l"), "clean")
            print(f"server data left on {a.host}:", left.strip(), flush=True)
        lab.close()


def letters_to_probs(rows):
    acc, cnt = {}, {}
    for r in rows:
        e, k, o = map(int, r["id"].split("|"))
        lp = np.array([r["logprobs"][L] for L in OPTIONS])
        p = np.exp(lp - np.max(lp)) if np.isfinite(lp).any() else np.full(4, 0.25)
        p = p / p.sum()
        _, mapping = question(ORDERS[o])
        v = acc.setdefault(e, np.zeros((5, 4)))
        for L, pr in zip(OPTIONS, p):
            v[k, LABELS4.index(mapping[L])] += pr
        cnt.setdefault(e, np.zeros(5))[k] += 1
    if any((c != 4).any() for c in cnt.values()):
        raise RuntimeError("event-steps without 4 answers")
    eids = sorted(acc)
    return np.array(eids), np.stack([acc[e] / cnt[e][:, None] for e in eids])


def convert():
    out = RUNS4 / "probs"
    out.mkdir(parents=True, exist_ok=True)
    lb_t, lb_e, ce = (RUNS4 / "raw" / n for n in ("server1_train", "server1_ext", "cnn_e2"))
    sets = {"M5e": [lb_t / "out/m5e_valsel_ep{e}.jsonl", lb_t / "out/m5e_valcal.jsonl", lb_e / "out/m5e_ext.jsonl",
                    lb_e / "out/m5e_test.jsonl"],
            "M5t": [lb_t / "out/m5t_valsel_ep{e}.jsonl", lb_t / "out/m5t_valcaltest.jsonl", lb_e / "out/m5t_ext.jsonl"],
            "M5tr": [lb_t / "out/m5tr_valsel_ep{e}.jsonl", lb_t / "out/m5tr_valcaltest.jsonl", lb_e / "out/m5tr_ext.jsonl"],
            "M3c": [lb_e / "out/m3c_ext.jsonl"]}
    for name, paths in sets.items():
        ch = lb_t / f"out/{name.lower()}_choice.json"
        e = json.loads(ch.read_text())["epoch"] if ch.exists() else None
        paths = [Path(str(p).format(e=e)) for p in paths]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(name, "missing", [str(p) for p in missing])
            continue
        eids, P = letters_to_probs([json.loads(l) for p in paths for l in open(p, encoding="utf-8")])
        np.savez_compressed(out / f"{name}.npz", probs=P, eid=eids)
        print("saved", name, P.shape)
    zs = [np.load(f, allow_pickle=True) for f in sorted(ce.glob("out/M2e2_s*.npz"))]
    if zs:
        np.savez_compressed(out / "M2_e2.npz", probs=np.mean([z["probs"] for z in zs], 0), eid=zs[0]["eid"])
        print("saved M2_e2", len(zs), "seeds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["folds", "run", "convert"])
    ap.add_argument("--host", choices=list(HOSTS))
    ap.add_argument("--phase", choices=["cnn", "cnn_e2", "train", "ext"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hours", type=float, default=24)
    a = ap.parse_args()
    if a.cmd == "folds":  # design notes: 5 random folds over the ForestNet training events, seed 20261027
        fn = fn_events()
        tr = fn[fn.part == "train"].eid.to_numpy()
        f = np.random.default_rng(20261027).permutation(len(tr)) % 5
        V4.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"eid": tr, "fold": f}).to_csv(V4 / "folds.csv", index=False)
        print("folds:", np.bincount(f).tolist())
    elif a.cmd == "run":
        run(a)
    else:
        convert()


if __name__ == "__main__":
    main()
