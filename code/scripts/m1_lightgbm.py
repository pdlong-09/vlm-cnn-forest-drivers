"""M1: one LightGBM per step k (k = 0..4) on series features up to year k plus static layers (design notes).

    .venv_rs/Scripts/python paper/rs_forestnet/scripts/m1_lightgbm.py

Trained on train, hyperparameters and early stopping on val_sel only. Writes class probabilities for every event and
step to runs/rs_forestnet/probs/M1.npz (probs [n, 5, 4], eid, part, y). Prints val_sel macro-F1 per step; the test
part is written but not scored here (scoring happens once, in evaluate.py, after every model is frozen).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsdrivers.features import LABELS4, series_wide, step_features, usable  # noqa: E402
from rsdrivers.io import INTERIM, RUNS, events  # noqa: E402

STEPS = range(0, 5)
GRID = [{"num_leaves": nl, "min_child_samples": mc} for nl in (15, 31) for mc in (10, 25)]


def main():
    # --no-nobs (post-test sensitivity, 22/10/2026): drop the clear-observation counts, which reveal the sensor era
    no_nobs = "--no-nobs" in sys.argv[1:]
    # --with-future (post-test exploration, 23/10/2026): keep Hansen gain and the OSM distances that v1 excludes as
    # future information, to measure how much of ForestNet's published score such features can carry
    with_future = "--with-future" in sys.argv[1:]
    name = "M1nonobs" if no_nobs else "M1future" if with_future else "M1"
    ev = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    y = ev.merged_label.map({l: i for i, l in enumerate(LABELS4)}).to_numpy()
    static = pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid")
    static = static if with_future else usable(static)
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    probs = np.zeros((len(ev), len(STEPS), len(LABELS4)))
    tr, sel = (ev.part == "train").to_numpy(), (ev.part == "val_sel").to_numpy()
    for k in STEPS:
        ids = ev.eid.to_numpy()
        X = (step_features(wide, k).reindex(ids).reset_index(drop=True)
             .join(static.reindex(ids).reset_index(drop=True)))
        if no_nobs:
            X = X.drop(columns=[c for c in X.columns if c.startswith("nobs_")])
        best = None
        for g in GRID:
            m = lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.03, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, random_state=k, **g)
            m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]), eval_set=[(X[sel], y[sel])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            f1 = f1_score(y[sel], m.predict(X[sel]), average="macro")
            if best is None or f1 > best[0]:
                best = (f1, g, m)
        probs[:, k] = best[2].predict_proba(X)
        print(f"k={k}: val_sel macro-F1 {best[0]:.3f} with {best[1]}, {best[2].best_iteration_} trees", flush=True)
    out = RUNS / "probs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", probs=probs, eid=ev.eid.to_numpy(), part=ev.part.to_numpy(), y=y)
    print("saved", out / f"{name}.npz")


if __name__ == "__main__":
    main()
