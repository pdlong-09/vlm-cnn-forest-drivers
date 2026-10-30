"""POST-HOC exploration on the ForestNet test (run 23/10/2026, after the single pre-specified test scoring).
Nothing here changes a pre-specified outcome; every number is labelled exploratory.

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/explore_forestnet_posthoc.py

A. The v2 / v2.1 recipe on ForestNet: M4 = mean(M2, M3c), with and without temperature scaling (fitted per k on
   val_sel, the v1 selection set), beta 0 / 0.15, alpha 0.05 / 0.10, lambda by learn-then-test on val_cal.
B. How much the excluded future features (Hansen gain, OSM distances) add: M1 vs M1future (scripts/m1_lightgbm.py
   --with-future), macro-F1 per step on test.
Output: runs/rs_forestnet/posthoc_explore.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load  # noqa: E402
from rsdrivers import earlystop_v2 as es  # noqa: E402
from rsdrivers.io import RUNS  # noqa: E402


def mf1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def boot(fn, n, B=2000, seed=20261021):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


def main():
    data = {m: load(m) for m in ("M1", "M1future", "M2", "M3c")}
    common = sorted(set.intersection(*[set(d[0]) for d in data.values()]))
    arr = {}
    for m, (eids, P, part, y) in data.items():
        pos = {e: i for i, e in enumerate(eids)}
        idx = [pos[e] for e in common]
        arr[m] = P[idx]
        part_c, y_c = part[idx], y[idx]
    part, y = part_c, y_c
    sel, cal, te = part == "val_sel", part == "val_cal", part == "test"
    temps = {m: es.fit_temperature(arr[m][sel], y[sel]) for m in ("M2", "M3c")}
    ts = {m: es.apply_temperature(arr[m], temps[m]) for m in temps}
    models = {"M2": arr["M2"], "M3c": arr["M3c"], "M4": (arr["M2"] + arr["M3c"]) / 2,
              "M2+TS": ts["M2"], "M3c+TS": ts["M3c"], "M4+TS": (ts["M2"] + ts["M3c"]) / 2}
    out = {"note": "POST-HOC, after the pre-specified test scoring; exploratory only", "n_test": int(te.sum()),
           "temperatures": {m: t.tolist() for m, t in temps.items()}, "A": {}, "B": {}}
    stops = {}
    for name, P in models.items():
        for alpha in (0.05, 0.10):
            for beta in (0.0, 0.15):
                lam = es.calibrate(P[cal], y[cal], alpha, 0.10, beta)
                e, t, pred = es.evaluate(P[te], y[te], lam, beta)
                e["macro_f1_by_k"] = [round(mf1(y[te], P[te, k].argmax(1)), 3) for k in range(5)]
                e["macro_f1_at_stop"] = round(mf1(y[te], pred), 3)
                out["A"][f"{name}|a{alpha}|b{beta}"] = e
                stops[(name, alpha, beta)] = t
    yt = y[te]
    for ts_ in ("", "+TS"):
        for alpha in (0.05, 0.10):
            for beta in (0.0, 0.15):
                d = stops[(f"M4{ts_}", alpha, beta)] - stops[(f"M2{ts_}", alpha, beta)]
                out["A"][f"stop M4{ts_} - M2{ts_}|a{alpha}|b{beta}"] = {"diff": round(float(d.mean()), 3),
                                                                       "ci": boot(lambda i: d[i].mean(), len(d))}
    f = lambda m, i: mf1(yt[i], models[m][te][i, 4].argmax(1))  # noqa: E731
    allidx = np.arange(len(yt))
    best = max(["M2", "M3c"], key=lambda m: f(m, allidx))
    out["A"]["F1 k4 M4 - best(M2, M3c)"] = {"best": best, "diff": round(f("M4", allidx) - f(best, allidx), 3),
                                            "ci": boot(lambda i: f("M4", i) - f(best, i), len(yt))}
    for m in ("M1", "M1future"):
        out["B"][m] = [round(mf1(yt, arr[m][te, k].argmax(1)), 3) for k in range(5)]
    g = lambda m, i: mf1(yt[i], arr[m][te][i, 4].argmax(1))  # noqa: E731
    out["B"]["F1 k4 M1future - M1"] = {"diff": round(g("M1future", allidx) - g("M1", allidx), 3),
                                       "ci": boot(lambda i: g("M1future", i) - g("M1", i), len(yt))}
    path = RUNS / "posthoc_explore.json"
    path.write_text(json.dumps(out, indent=1))
    for k, v in out["A"].items():
        if "|a0.05|" in k or k.startswith("F1"):
            if "lambda" in v:
                print(f"{k:22s} lambda {v['lambda']:.2f} gap {v['gap_loss']:.3f} stop {v['mean_stop']:.2f} acc@stop "
                      f"{v['acc_early']:.3f} F1@stop {v['macro_f1_at_stop']:.3f} F1 k0..4 {v['macro_f1_by_k']}")
            else:
                print(k, v)
    for k, v in out["A"].items():
        if "|a0.1|" in k and "lambda" in v:
            print(f"{k:22s} lambda {v['lambda']:.2f} gap {v['gap_loss']:.3f} stop {v['mean_stop']:.2f} acc@stop {v['acc_early']:.3f}")
    print("B:", json.dumps(out["B"]))
    print("temperatures:", out["temperatures"])
    print("saved", path)


if __name__ == "__main__":
    main()
