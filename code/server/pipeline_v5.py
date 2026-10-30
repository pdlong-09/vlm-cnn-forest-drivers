"""design notes GPU pipeline, resumable stage by stage (server1 for the VLM phases, server2 for the CNN phase).

    python pipeline_v5.py --phase train|test|cnn_e3 [--gpus 0,1]

train (server1): render -> two chains side by side, one per GPU when both are free:
          p0:  m6_p0_train  (fn_m6.py, p = 0)   -> m6_p0_dev_ep1  -> m6_p0_dev_ep2   (E2, k = 4, 4 option orders)
          p05: m6_p05_train (fn_m6.py, p = 0.5) -> m6_p05_dev_ep1 -> m6_p05_dev_ep2
        -> select: binary BA at k = 4 on E2 (P(agriculture) = P(Plantation) + P(Smallholder), mean over the 4 orders,
           decision at 0.5); ties: p = 0.5 first, then epoch 1 (design notes). Writes out/m6_choice.json.
seeds (server1, design notes, post hoc): M3c v1 recipe with --seeds; per seed train -> val_sel epoch choice -> E2, E3 (k = 4).
test (server1): render -> chains m6_e3 (the chosen candidate) | m3c_e3 (v1 vlm_infer.py, the v1 M3c adapter); E3, k = 4.
cnn_e3 (server2): M2 seeds 0-2 (v3 weights) on the E3 chips.
GPU pool, waits and OOM retries come from gpu_pipeline.py; only E2 labels (meta_e2.csv) are on the server.
"""
import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import gpu_pipeline as cp

RUN = Path(__file__).resolve().parent.parent
ROLE = {"server1": {"py": "/home/user/llm4deforest/paperA/venv/bin/python",
                   "pyr": "/home/user/llm4deforest/paperA/venv_render/bin/python",
                   "model": "/home/user/llm4deforest/models/Qwen3.5-9B"},
        "server2": {"py": "/home/user/llm4deforest_v4/venv/bin/python"}}
CANDIDATES = [("p05", 1), ("p05", 2), ("p0", 1), ("p0", 2)]  # tie order of design notes


def dev_ba(path, y):
    """Binary BA at k = 4 of letter probabilities averaged over the 4 option orders."""
    acc = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        e, k, o = map(int, r["id"].split("|"))
        lp = [r["logprobs"][L] for L in "ABCD"]
        m = max(lp)
        p = [math.exp(v - m) if math.isfinite(v) else 0.0 for v in lp] if math.isfinite(m) else [0.25] * 4
        s = sum(p)
        a = acc.setdefault(e, [0.0, 0])
        a[0] += sum(p[i] / s for i in range(4) if cp.ORDERS[o][i] in ("Plantation", "Smallholder agriculture"))
        a[1] += 1
    if any(n != 4 for _, n in acc.values()) or set(acc) != set(y):
        raise RuntimeError(f"{path}: incomplete answers")
    pos = [e for e in y if y[e] == 1]
    neg = [e for e in y if y[e] == 0]
    sens = sum(acc[e][0] / 4 >= 0.5 for e in pos) / len(pos)
    spec = sum(acc[e][0] / 4 < 0.5 for e in neg) / len(neg)
    return (sens + spec) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "test", "cnn_e3", "seeds"], required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    cp.POOL = cp.Pool([int(g) for g in a.gpus.split(",")])
    P = ROLE["server2" if a.phase == "cnn_e3" else "server1"]
    PY = P["py"]
    state, out, logs = RUN / "state", RUN / "out", RUN / "logs"
    for d in (state, out, logs, RUN / "adapters"):
        d.mkdir(exist_ok=True)

    def stage(name, fn):
        if (state / name).exists():
            print(f"[{name}] already done", flush=True)
            return
        t0 = time.time()
        fn()
        (state / name).write_text(f"{time.time() - t0:.0f} s\n")
        print(f"[{name}] done in {time.time() - t0:.0f} s", flush=True)

    def run(name, cmd, gpu=True):
        cp.run(name, cmd, gpu, logs / f"{name}.txt")

    def chains_run(chains):
        failed = {}

        def go(name, steps):
            try:
                for n, fn in steps:
                    stage(n, fn)
            except BaseException as e:
                failed[name] = repr(e)
                print(f"chain {name} stopped: {e!r}", flush=True)
        threads = [threading.Thread(target=go, args=(n, s)) for n, s in chains.items()]
        for t in threads:
            t.start()
            time.sleep(60)
        for t in threads:
            t.join()
        return failed

    failed = {}
    if a.phase == "cnn_e3":
        def cnn():
            for s in (0, 1, 2):
                run(f"M2_e3_s{s}", f"{PY} code/cnn_v3.py predict --data chips_e3.npz --weights weights/M2_s{s}.pt "
                                   f"--out out/M2e3_s{s}.npz")
        stage("cnn_e3", cnn)
    else:
        MODEL = P["model"]
        m6 = lambda ck, inp, o: (f"{PY} code/fn_m6.py infer --model {MODEL} --ckpt {ck} --input {inp} --images images "  # noqa: E731
                                 f"--feats aef.npz --output {o} --batch 8")
        stage("render", lambda: run("render", f"{P['pyr']} code/render_cards.py --jobs jobs.json --chips chips --out images", None))
        if a.phase == "train":
            def chain(tag, p):
                return [(f"m6_{tag}_train", lambda: run(f"m6_{tag}_train", f"{PY} code/fn_m6.py train --model {MODEL} --train "
                                                                           f"m6_train.jsonl --images images --feats aef.npz "
                                                                           f"--out adapters/m6_{tag} --p {p}"))] + \
                       [(f"m6_{tag}_dev_ep{e}", lambda e=e: run(f"m6_{tag}_dev_ep{e}", m6(f"adapters/m6_{tag}/epoch{e}",
                                                                                         "m6_dev.jsonl", f"out/m6_{tag}_dev_ep{e}.jsonl")))
                        for e in (1, 2)]
            failed = chains_run({"p0": chain("p0", 0.0), "p05": chain("p05", 0.5)})
            if not failed:
                y = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_e2.csv"))}
                scores = {f"{t}_ep{e}": dev_ba(out / f"m6_{t}_dev_ep{e}.jsonl", y) for t, e in CANDIDATES}
                best = max(CANDIDATES, key=lambda c: (scores[f"{c[0]}_ep{c[1]}"], -CANDIDATES.index(c)))
                (out / "m6_choice.json").write_text(json.dumps({"tag": best[0], "epoch": best[1], "e2_ba_k4": scores}))
                (out / "m6_choice_path.txt").write_text(f"adapters/m6_{best[0]}/epoch{best[1]}")
                print(f"[select] {best} chosen: {scores}", flush=True)
        elif a.phase == "seeds":  # design notes (post hoc): v1 M3c recipe, other seeds; epoch chosen on val_sel as v1
            def vl(lora, inp, o):
                return f"{PY} code/vlm_infer.py --model {MODEL} --images images --input {inp} --lora {lora} --output {o}"
            y = {int(r["eid"]): int(r["y"]) for r in csv.DictReader(open(RUN / "meta_valsel.csv"))}
            seeds = [int(x) for x in a.seeds.split(",")]

            def select(sd):
                sc = {e: cp.holdout_f1(out / f"m3c_s{sd}_valsel_ep{e}.jsonl", y) for e in (1, 2)}
                e = 1 if sc[1][0] >= sc[2][0] else 2
                (out / f"m3c_s{sd}_choice.json").write_text(json.dumps({"seed": sd, "epoch": e,
                                                                         "val_sel_mean_f1": {k: v[0] for k, v in sc.items()}}))
                print(f"[m3c_s{sd}_select] epoch {e}: {sc}", flush=True)

            def ep(sd):
                return json.loads((out / f"m3c_s{sd}_choice.json").read_text())["epoch"]
            steps = []
            for sd in seeds:
                steps.append((f"m3c_s{sd}_train", lambda sd=sd: run(f"m3c_s{sd}_train", (
                    f"{PY} code/qlora_train.py --model {MODEL} --train m3c_train.jsonl --images images "
                    f"--out adapters/m3c_s{sd} --epochs 2 --seed {sd}"))))
                for e in (1, 2):
                    steps.append((f"m3c_s{sd}_valsel_ep{e}", lambda sd=sd, e=e: run(f"m3c_s{sd}_valsel_ep{e}", vl(
                        f"adapters/m3c_s{sd}/epoch{e}", "m3c_valsel.jsonl", f"out/m3c_s{sd}_valsel_ep{e}.jsonl"))))
                steps.append((f"m3c_s{sd}_select", lambda sd=sd: select(sd)))
                for t in ("e2", "e3"):
                    steps.append((f"m3c_s{sd}_{t}", lambda sd=sd, t=t: run(f"m3c_s{sd}_{t}", vl(
                        f"adapters/m3c_s{sd}/epoch{ep(sd)}", f"m3c_{t}.jsonl", f"out/m3c_s{sd}_{t}.jsonl"))))
            failed = chains_run({"seeds": steps})
            (out / "m3c_seed_paths.txt").write_text(" ".join(f"adapters/m3c_s{sd}/epoch{ep(sd)}" for sd in seeds
                                                             if (out / f"m3c_s{sd}_choice.json").exists()))
        else:
            vl = (f"{PY} code/vlm_infer.py --model {MODEL} --images images --input m3c_e3.jsonl --lora adapters/m3c "
                  f"--output out/m3c_e3.jsonl")
            failed = chains_run({"m6": [("m6_e3", lambda: run("m6_e3", m6("adapters/m6", "m6_e3.jsonl", "out/m6_e3.jsonl")))],
                                 "m3c": [("m3c_e3", lambda: run("m3c_e3", vl))]})
    print("pipeline finished" + (f" with failures: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
