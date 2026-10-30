"""Driver attribution, fact-sheet country ablation (26/10/2026): every evidence card names Indonesia in its first line, also for
the cells on other continents. This batch runs the unchanged v1 M3c adapter on the E3 cards with the cell's country
named instead (nothing else in the card changes) and compares the result with the published E3 predictions.

    LAB_PW=... .venv/Scripts/python paper/rs_forestnet/scripts/fn_country_ablation.py run [--gpus 1]
    .venv/Scripts/python paper/rs_forestnet/scripts/fn_country_ablation.py evaluate

run: package (E3 chips, card jobs, prompts, adapter; no label) -> upload -> pipeline_country.py through the GPU queue ->
pull out/ -> delete the server copy (lab-server batch rule). Results in runs/rs_forestnet_robust/raw/country/.
evaluate: BA at k = 4 with and without the country (paired bootstrap of cells and of countries, 2000 resamples),
by region, and the share of cells whose binary call changes. Writes runs/rs_forestnet_robust/country_ablation.json.
"""
import argparse
import json
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
from fn_v4 import M3C_ADAPTER, ORDERS, e2_facts  # noqa: E402
from fn_v5 import E3, HOSTS, OFF3, k4_probs, queued, sims_job  # noqa: E402
from rsdrivers.card import question  # noqa: E402
from rsdrivers.features import series_wide  # noqa: E402
from rsdrivers.lab import Lab  # noqa: E402

RAW = ROOT / "runs/rs_forestnet_robust/raw/country"
OUT = ROOT / "runs/rs_forestnet_robust/country_ablation.json"
CODE = ["server/vlm_infer.py", "server/gpu_pipeline.py", "server/render_cards.py", "rsdrivers/render.py",
        "server/pipeline_country.py"]
HEADER = "Forest loss event in Indonesia (Hansen GFC)."
NAMES = {"Burma": "Myanmar", "Central African Rep": "the Central African Republic", "Cote d'Ivoire": "Côte d'Ivoire",
         "Dem Rep of the Congo": "the Democratic Republic of the Congo", "Gambia, The": "the Gambia",
         "Rep of the Congo": "the Republic of the Congo", "Solomon Is": "the Solomon Islands",
         "Trinidad & Tobago": "Trinidad and Tobago", "Reunion": "Réunion", "Philippines": "the Philippines",
         "Dominican Republic": "the Dominican Republic"}


def build(pkg: Path):
    (pkg / "code").mkdir(parents=True)
    for d in ("chips/sims3", "out", "adapters"):
        (pkg / d).mkdir(parents=True)
    for f in CODE:
        shutil.copy(ROOT / "paper/rs_forestnet" / f, pkg / "code" / Path(f).name)
    ev = pd.read_csv(E3 / "events.csv", usecols=["sid", "year", "Latitude", "Longitude", "country"])  # no label column
    geoms = {f["properties"]["sid"]: shape(f["geometry"]) for f in json.loads((E3 / "polygons.geojson").read_text())["features"]}
    swide = series_wide(pd.read_csv(E3 / "series_gee.csv"))
    sstat = pd.read_csv(E3 / "static.csv").set_index("sid")
    jobs = []
    with open(pkg / "m3c_e3_country.jsonl", "w", encoding="utf-8") as fh:
        for r in ev.itertuples():
            src = E3 / "chips" / str(r.sid)
            if src.exists():
                shutil.copytree(src, pkg / "chips/sims3" / str(r.sid), dirs_exist_ok=True)
            j = sims_job(E3, "sims3", r, 4, geoms[r.sid])
            jobs.append(j)
            facts = e2_facts(swide.loc[r.sid].to_dict() if r.sid in swide.index else {}, sstat.loc[r.sid], 4)
            assert facts.startswith(HEADER), facts[:80]
            facts = facts.replace(HEADER, f"Forest loss event in {NAMES.get(r.country, r.country)} (Hansen GFC).", 1)
            for o, order in enumerate(ORDERS):
                q, _ = question(order)
                fh.write(json.dumps({"id": f"{OFF3 + int(r.sid)}|4|{o}", "text": facts + "\n\n" + q, "image": j["name"],
                                     "shots": []}) + "\n")
    (pkg / "jobs.json").write_text(json.dumps(jobs))
    shutil.copytree(M3C_ADAPTER, pkg / "adapters/m3c")
    print(f"package: E3 {len(ev)} cells, {4 * len(ev)} prompts, adapter {M3C_ADAPTER.parent.parent.name}", flush=True)


def run(a):
    lab, H = Lab("lab"), HOSTS["server1"]
    R, run_id, keep = H["remote"], f"country_{time.strftime('%Y%m%d%H%M%S')}", False
    try:
        lab.ssh(f"mkdir -p {R} && chmod 700 {R}")
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / run_id
            build(pkg)
            tar = Path(td) / f"{run_id}.tar"
            with tarfile.open(tar, "w") as t:
                t.add(pkg, arcname=run_id)
            print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
            lab.put_chunked(tar, f"{R}/{run_id}.tar", chunk_mb=4, workers=6)
        if lab.run_detached(R, f"tar xf {run_id}.tar && rm {run_id}.tar", f"{R}/{run_id}.unpack", poll_s=15) != 0:
            raise RuntimeError("unpack failed")
        pipe = queued(lab, a.gpus, f"{R}/{run_id}", f"{H['py']} {R}/{run_id}/code/pipeline_country.py --gpus {a.gpus}")
        code = lab.run_detached(R, f"{pipe} >> {R}/{run_id}/pipeline_log.txt 2>&1", f"{R}/{run_id}.exit", poll_s=60,
                                timeout_h=6)
        keep = True
        print("pipeline exit", code, "|", lab.retry(lambda: lab.ssh(f"tail -3 {R}/{run_id}/pipeline_log.txt"), "log"), flush=True)
        lab.retry(lambda: lab.ssh(f"cd {R}/{run_id} && tar cf ../{run_id}_out.tar out logs state pipeline_log.txt"), "pack")
        if RAW.exists():
            shutil.rmtree(RAW)
        RAW.mkdir(parents=True)
        lab.get_chunked(f"{R}/{run_id}_out.tar", RAW / "out.tar")
        with tarfile.open(RAW / "out.tar") as t:
            t.extractall(RAW, filter="data")
        (RAW / "out.tar").unlink()
        (RAW / "RUN_ID").write_text(run_id)
        if code != 0:
            raise RuntimeError(f"pipeline failed (exit {code}); data kept on server1 in {R}/{run_id}")
        keep = False
    finally:
        if keep:
            print(f"server data kept on server1 in {R}/{run_id}", flush=True)
        else:
            left = lab.retry(lambda: lab.ssh(f"rm -rf {R}/{run_id} {R}/{run_id}.tar* {R}/{run_id}_out.tar {R}/{run_id}.exit* "
                                             f"{R}/{run_id}.unpack*; ls -A {R} | wc -l"), "clean", tries=40)
            print("server data left on server1:", left.strip(), flush=True)
        lab.close()


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def evaluate():
    ev = pd.read_csv(E3 / "events.csv")
    ids = (OFF3 + ev.sid).to_numpy()
    y = ev.label_agri.to_numpy()
    z = np.load(ROOT / "runs/rs_forestnet_v5/probs/M3c_E3.npz", allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    base = np.stack([probs[pos[int(e)]] for e in ids])
    eids, P = k4_probs(RAW / "out/m3c_e3_country.jsonl")
    cpos = {int(e): i for i, e in enumerate(eids)}
    ctry = np.stack([np.asarray(P)[cpos[int(e)]] for e in ids])
    a0 = ((base[:, 0] + base[:, 1]) >= 0.5).astype(int)
    a1 = ((ctry[:, 0] + ctry[:, 1]) >= 0.5).astype(int)
    rng = np.random.default_rng(20261021)
    n = len(y)
    d_cell = [(lambda i: ba(y[i], a1[i]) - ba(y[i], a0[i]))(rng.integers(0, n, n)) for _ in range(2000)]
    countries = ev.country.to_numpy()
    uc = np.unique(countries)
    members = {c: np.flatnonzero(countries == c) for c in uc}
    d_ctry = []
    for _ in range(2000):
        i = np.concatenate([members[c] for c in rng.choice(uc, len(uc), replace=True)])
        d_ctry.append(ba(y[i], a1[i]) - ba(y[i], a0[i]))
    out = {"n": int(n), "BA_card_names_Indonesia": round(ba(y, a0), 4), "BA_card_names_country": round(ba(y, a1), 4),
           "diff": round(ba(y, a1) - ba(y, a0), 4), "ci_cell": [round(float(v), 4) for v in np.percentile(d_cell, [2.5, 97.5])],
           "ci_country": [round(float(v), 4) for v in np.percentile(d_ctry, [2.5, 97.5])],
           "share_binary_call_changed": round(float((a0 != a1).mean()), 4),
           "share_4class_changed": round(float((base.argmax(1) != ctry.argmax(1)).mean()), 4),
           "by_group": {g: {"Indonesia": round(ba(y[m], a0[m]), 4), "country": round(ba(y[m], a1[m]), 4), "n": int(m.sum())}
                        for g in ("Africa", "Asia-Pacific", "Latin America") for m in [(ev.group == g).to_numpy()]}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "evaluate"])
    ap.add_argument("--gpus", default="1")
    a = ap.parse_args()
    run(a) if a.what == "run" else evaluate()


if __name__ == "__main__":
    main()
