"""Driver attribution robustness analyses on CPU (25/10/2026), run on server2 from an anonymous bundle (fn_robust_a.py cpu).

    python robust_a.py --bundle bundle --out out/robust_a.json

1. Spatial block bootstrap of the VLM - CNN differences in balanced accuracy (BA) on E2 and E3: clusters = countries and
   1-degree cells, besides the cell-level bootstrap of the article; plus the sign of the difference per country.
2. Early stop with learn-then-test (alpha 0.05, delta 0.10, beta 0): the gap on each target set with lambda calibrated on
   ForestNet val_cal (source) versus lambda calibrated on n_cal labelled cells of the target set itself, over 1000 random
   splits per n_cal (violation rate = share of splits whose gap on the rest exceeds alpha).
3. Tabular baseline M1x: one LightGBM per step k on the yearly index series (no clear-observation counts) and the four
   static layers shared by every set (area, parts, tree cover 2000, elevation), trained on ForestNet train with selection
   on val_sel exactly as M1; scored as binary agricultural conversion at k = 4.
Binary: P(agriculture) = P(Plantation) + P(Smallholder agriculture), decision at 0.5.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

import earlystop_v2 as es

B, SEED, ALPHA, DELTA = 2000, 20261021, 0.05, 0.10
INDICES = ["ndvi", "nbr", "ndmi"]
STATIC = ["area_ha", "n_parts", "tc2000_in", "srtm0_mean"]
GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (15, 31) for mc in (10, 25)]


def ba(y, p):
    pos, neg = y == 1, y == 0
    return 0.5 * ((p[pos] == 1).mean() + (p[neg] == 0).mean())


def agri(P):
    return P[..., 0] + P[..., 1]


def to_bin(P):
    a = agri(P)
    return np.stack([1 - a, a], -1)


def load(bundle, name, eids, k4=False):
    z = np.load(bundle / f"{name}.npz", allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    P = np.stack([probs[pos[int(e)]] for e in eids])
    return P[:, 4] if (k4 and P.ndim == 3) else P


def q(x):
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


# ---------- 1. block bootstrap ----------
def block_boot(y, pa, pb, clusters, rng):
    uc = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uc}
    out = []
    for _ in range(B):
        idx = np.concatenate([members[c] for c in rng.choice(uc, len(uc))])
        if y[idx].min() == y[idx].max():
            continue
        out.append(ba(y[idx], pa[idx]) - ba(y[idx], pb[idx]))
    return q(out)


def cell_boot(y, pa, pb, rng):
    n = len(y)
    return q([(lambda i: ba(y[i], pa[i]) - ba(y[i], pb[i]))(rng.integers(0, n, n)) for _ in range(B)])


def part1(bundle):
    res = {}
    for t in ("e2", "e3"):
        ev = pd.read_csv(bundle / f"{t}_events.csv")
        y = ev.label_agri.to_numpy()
        ids = ev.eid.to_numpy()
        P = {"M2": load(bundle, f"{t}_M2", ids, True), "M3c": load(bundle, f"{t}_M3c", ids, True)}
        for s in (1, 2):
            P[f"M3c_s{s}"] = load(bundle, f"{t}_M3c_s{s}", ids, True)
        if t == "e3":
            P["M6"] = load(bundle, "e3_M6", ids, True)
        P["M3c_mean3"] = (P["M3c"] + P["M3c_s1"] + P["M3c_s2"]) / 3
        P["M4"] = (P["M2"] + P["M3c"]) / 2
        pred = {m: (agri(p) >= 0.5).astype(int) for m, p in P.items()}
        country = ev.country.to_numpy()
        grid = (np.floor(ev.Latitude).astype(int).astype(str) + "_" + np.floor(ev.Longitude).astype(int).astype(str)).to_numpy()
        r = {"n": int(len(y)), "n_countries": int(len(np.unique(country))), "n_grid_cells": int(len(np.unique(grid)))}
        for a_ in [m for m in ("M3c", "M3c_s1", "M3c_s2", "M3c_mean3", "M4", "M6") if m in pred]:
            rng = np.random.default_rng(SEED)
            r[f"{a_} - M2"] = {"diff": round(ba(y, pred[a_]) - ba(y, pred["M2"]), 4),
                               "ci_cell": cell_boot(y, pred[a_], pred["M2"], rng),
                               "ci_country": block_boot(y, pred[a_], pred["M2"], country, rng),
                               "ci_1deg": block_boot(y, pred[a_], pred["M2"], grid, rng)}
        per = []
        for c in np.unique(country):
            i = np.flatnonzero(country == c)
            if len(i) >= 30 and 0 < y[i].sum() < len(i):
                per.append({"country": c, "n": int(len(i)), "M3c": round(ba(y[i], pred["M3c"][i]), 3),
                            "M2": round(ba(y[i], pred["M2"][i]), 3)})
        r["per_country_n>=30"] = per
        r["countries_M3c_better"] = f"{sum(p['M3c'] > p['M2'] for p in per)} of {len(per)}"
        res[t] = r
    return res


# ---------- 2. repeated calibration ----------
def part2(bundle):
    fn = pd.read_csv(bundle / "fn_events.csv")
    cal = fn[fn.part == "val_cal"]
    test = fn[fn.part == "test"]
    targets = {}
    PF = {"M2": load(bundle, "fn_M2", fn.eid), "M3c": None}
    m3 = np.load(bundle / "fn_M3c.npz", allow_pickle=True)
    pos = {int(e): i for i, e in enumerate(m3["eid"])}
    m3p = m3["probs"]

    def fnP(m, ids, cnn="fn_M2"):
        """ForestNet probabilities; cnn = the CNN whose probabilities the target set uses (v1 on ForestNet test, the
        retrained v3 CNN, fn3_M2, on the Sims sets), so that lambda is calibrated on the very model it is applied to."""
        if m == "M2":
            return load(bundle, cnn, ids)
        if m == "M3c":
            return np.stack([m3p[pos[int(e)]] for e in ids])
        return (fnP("M2", ids, cnn) + fnP("M3c", ids)) / 2
    targets["ForestNet test (4 classes)"] = {"y": test.y.to_numpy(), "P": {m: fnP(m, test.eid) for m in ("M2", "M3c", "M4")},
                                             "src": {m: (fnP(m, cal.eid), cal.y.to_numpy()) for m in ("M2", "M3c", "M4")}}
    ybin_cal = cal.y.isin([0, 1]).astype(int).to_numpy()
    for t, name in (("sidn", "Sims Indonesia"), ("e2", "E2")):
        ev = pd.read_csv(bundle / f"{t}_events.csv")
        P = {"M2": load(bundle, f"{t}_M2", ev.eid), "M3c": load(bundle, f"{t}_M3c", ev.eid)}
        P["M4"] = (P["M2"] + P["M3c"]) / 2
        targets[f"{name} (binary)"] = {"y": ev.label_agri.to_numpy(), "P": {m: to_bin(p) for m, p in P.items()},
                                        "src": {m: (to_bin(fnP(m, cal.eid, "fn3_M2")), ybin_cal) for m in P}}
        if t == "e2":
            for g in sorted(ev.group.unique()):
                i = np.flatnonzero((ev.group == g).to_numpy())
                targets[f"E2 {g} (binary)"] = {"y": ev.label_agri.to_numpy()[i], "P": {m: to_bin(p[i]) for m, p in P.items()},
                                                "src": {m: (to_bin(fnP(m, cal.eid, "fn3_M2")), ybin_cal) for m in P}}
    res = {}
    for tname, T in targets.items():
        y, n = T["y"], len(T["y"])
        rt = {"n": int(n)}
        for m, P in T["P"].items():
            lam_src = es.calibrate(*T["src"][m], ALPHA, DELTA, 0.0)
            ev_src = es.evaluate(P, y, lam_src, 0.0)[0]
            rm = {"source_lambda": round(lam_src, 3), "source_gap": round(ev_src["gap_loss"], 4),
                  "source_mean_stop": round(ev_src["mean_stop"], 3), "local": {}}
            rng = np.random.default_rng(SEED)
            for n_cal in (50, 100, 200, 400):
                if n_cal > 0.6 * n:
                    continue
                gaps, full, stops, never = [], [], [], 0
                for _ in range(1000):
                    perm = rng.permutation(n)
                    c, r_ = perm[:n_cal], perm[n_cal:]
                    lam = es.calibrate(P[c], y[c], ALPHA, DELTA, 0.0)
                    never += lam > 1.0
                    l, st, _ = es.losses(P[r_], y[r_], lam, 0.0)
                    gaps.append(l.mean())
                    full.append(es.losses(P, y, lam, 0.0)[0].mean())  # risk on the whole target set, a proxy of R(lambda)
                    stops.append(st.mean())
                gaps, full = np.array(gaps), np.array(full)
                rm["local"][str(n_cal)] = {"share_gap_rest_above_alpha": round(float((gaps > ALPHA).mean()), 4),
                                           "share_risk_fullset_above_alpha": round(float((full > ALPHA).mean()), 4),
                                           "mean_gap": round(float(gaps.mean()), 4), "gap_p95": round(float(np.percentile(gaps, 95)), 4),
                                           "mean_stop": round(float(np.mean(stops)), 3), "share_never_early": round(never / 1000, 3)}
            rt[m] = rm
        res[tname] = rt
    return res


# ---------- 3. tabular baseline ----------
def series_wide(s):
    s = s.rename(columns={f"{i}_median": i for i in INDICES})
    w = s.pivot_table(index="eid", columns=["zone", "k"], values=INDICES)
    w.columns = [f"{v}_{z}_y{k}" for v, z, k in w.columns]
    return w


def step_features(wide, k):
    out = wide.copy()
    for col in out.columns:
        if int(col.rsplit("_y", 1)[1]) > k:
            out[col] = np.nan
    for i in INDICES:
        for z in ("in", "ring"):
            pre = out[[f"{i}_{z}_y{t}" for t in (-2, -1) if f"{i}_{z}_y{t}" in out]].mean(1)
            post = out[[f"{i}_{z}_y{t}" for t in range(0, k + 1) if f"{i}_{z}_y{t}" in out]]
            out[f"{i}_{z}_drop"] = pre - post.min(1)
            out[f"{i}_{z}_last_minus_pre"] = post.iloc[:, -1] - pre
            out[f"{i}_{z}_recovery"] = post.iloc[:, -1] - post.min(1)
    return out


def features(bundle, t, ids, k):
    wide = series_wide(pd.read_csv(bundle / f"{t}_series.csv"))
    st = pd.read_csv(bundle / f"{t}_static.csv").set_index("eid")[STATIC]
    return step_features(wide, k).reindex(ids).reset_index(drop=True).join(st.reindex(ids).reset_index(drop=True))


def part3(bundle):
    fn = pd.read_csv(bundle / "fn_events.csv")
    y = fn.y.to_numpy()
    tr, sel = (fn.part == "train").to_numpy(), (fn.part == "val_sel").to_numpy()
    sets = {t: pd.read_csv(bundle / f"{t}_events.csv") for t in ("sidn", "e2", "e3")}
    probs = {"fn": np.zeros((len(fn), 5, 4)), **{t: np.zeros((len(e), 5, 4)) for t, e in sets.items()}}
    sel_f1 = []
    for k in range(5):
        X = features(bundle, "fn", fn.eid.to_numpy(), k)
        best = None
        for g in GRID:
            m = lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=k, **g)
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]), eval_set=[(X[sel], y[sel])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            f1 = f1_score(y[sel], m.predict(X[sel]), average="macro")
            if best is None or f1 > best[0]:
                best = (f1, m)
        sel_f1.append(round(best[0], 4))
        probs["fn"][:, k] = best[1].predict_proba(X)
        for t, e in sets.items():
            probs[t][:, k] = best[1].predict_proba(features(bundle, t, e.eid.to_numpy(), k))
    res = {"val_sel_macro_f1_by_k": sel_f1}
    te = (fn.part == "test").to_numpy()
    yb = fn.y.isin([0, 1]).astype(int).to_numpy()
    m2 = load(bundle, "fn_M2", fn.eid)
    res["ForestNet test"] = {"M1x macro-F1 k4 (4 classes)": round(f1_score(y[te], probs["fn"][te, 4].argmax(1), average="macro"), 4),
                             "M1x BA k4 (binary)": round(ba(yb[te], (agri(probs["fn"][te, 4]) >= 0.5).astype(int)), 4),
                             "M2 BA k4 (binary)": round(ba(yb[te], (agri(m2[te, 4]) >= 0.5).astype(int)), 4)}
    for t, e in sets.items():
        yt = e.label_agri.to_numpy()
        P = {"M1x": probs[t][:, 4], "M2": load(bundle, f"{t}_M2", e.eid, True), "M3c": load(bundle, f"{t}_M3c", e.eid, True)}
        P["M1x+M2"] = (P["M1x"] + P["M2"]) / 2
        P["M1x+M3c"] = (P["M1x"] + P["M3c"]) / 2
        pred = {m: (agri(p) >= 0.5).astype(int) for m, p in P.items()}
        r = {m: round(ba(yt, p), 4) for m, p in pred.items()}
        if "group" in e:
            r["by_group"] = {g: {m: round(ba(yt[i], pred[m][i]), 4) for m in ("M1x", "M2", "M3c")}
                             for g in sorted(e.group.unique()) for i in [np.flatnonzero((e.group == g).to_numpy())]}
        rng = np.random.default_rng(SEED)
        r["M3c - M1x"] = {"diff": round(r["M3c"] - r["M1x"], 4), "ci_cell": cell_boot(yt, pred["M3c"], pred["M1x"], rng)}
        r["M1x - M2"] = {"diff": round(r["M1x"] - r["M2"], 4), "ci_cell": cell_boot(yt, pred["M1x"], pred["M2"], rng)}
        r["M1x BA by k"] = [round(ba(yt, (agri(probs[t][:, k]) >= 0.5).astype(int)), 4) for k in range(5)]
        res[t] = r
    np.savez_compressed("out/M1x_probs.npz", **{f"{t}_probs": p for t, p in probs.items()},
                        **{f"{t}_eid": (fn.eid if t == "fn" else sets[t].eid).to_numpy() for t in probs})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="1,3,2")
    a = ap.parse_args()
    bundle = Path(a.bundle)
    res = {}
    todo = {"1": ("1_block_bootstrap", part1), "3": ("3_tabular_M1x", part3), "2": ("2_repeated_calibration", part2)}
    for key, fn in (todo[p] for p in a.parts.split(",")):
        res[key] = fn(bundle)
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))  # kept after every part
        print(key, "done", flush=True)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
