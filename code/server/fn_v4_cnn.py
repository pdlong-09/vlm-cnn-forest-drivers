"""design notes: the remote-sensing expert C = the v1/v3 M2 CNN (seed 0) — features (512-d after pooling) and
4-class probabilities at every step k. On the training events they are out-of-fold: 5 CNNs, each trained with the
unchanged recipe (server/cnn_v3.py train, selection on val_sel) on 4/5 of train, predict the left-out fifth.
Everywhere else (val_sel, val_cal, test, E2) the v3 seed-0 CNN trained on all of train is used.

    python fn_v4_cnn.py folds   --data chips.npz --folds folds.csv --outdir cnnv4      (5 trainings)
    python fn_v4_cnn.py extract --data chips.npz --weights w.pt --out feats.npz [--only-eids eids.txt]
"""
import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_v3  # noqa: E402
from cnn_train import model, to_input  # noqa: E402


def extract(X, weights, dev="cuda"):
    net, _ = model()
    net.load_state_dict(torch.load(weights, map_location=dev))
    fc = net.fc
    net.fc = nn.Identity()
    net = net.to(dev).eval()
    fc = fc.to(dev)
    feats, probs = np.zeros((len(X), 5, 512), np.float32), np.zeros((len(X), 5, 4), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 128):
            xb = X[s:s + 128].to(dev)
            for k in range(5):
                f = net(to_input(xb, torch.full((len(xb),), k, device=dev)))
                feats[s:s + len(xb), k] = f.cpu().numpy()
                probs[s:s + len(xb), k] = torch.softmax(fc(f), 1).cpu().numpy()
    return feats, probs


def folds(a):
    z = np.load(a.data, allow_pickle=True)
    fold = {int(r["eid"]): int(r["fold"]) for r in csv.DictReader(open(a.folds))}
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    feats, probs, eids = [], [], []
    for f in range(5):
        part = z["part"].copy()
        held = np.array([fold.get(int(e), -1) == f for e in z["eid"]])
        part[held] = "oof"  # left out of training; cnn_v3.train trains on "train" and selects on "val_sel"
        tmp = out / f"fold{f}.npz"
        np.savez(tmp, X=z["X"], y=z["y"], part=part, eid=z["eid"])
        w = out / f"fold{f}.pt"
        cnn_v3.train(SimpleNamespace(data=str(tmp), out=str(out / f"fold{f}_probs.npz"), weights=str(w), epochs=40, seed=0))
        tmp.unlink()
        is_train = np.array([int(e) in fold for e in z["eid"]])
        fe_all, pr_all = extract(torch.from_numpy(z["X"][is_train]), w)  # every training event through fold model f
        tr_eid = z["eid"][is_train]
        np.savez_compressed(out / f"fold{f}_trainfeats.npz", eid=tr_eid, feat=fe_all)  # retrieval space of fold f (M5tr)
        sel = np.isin(tr_eid, z["eid"][held])
        feats.append(fe_all[sel]); probs.append(pr_all[sel]); eids.append(tr_eid[sel])
        print(f"fold {f}: {held.sum()} out-of-fold events", flush=True)
    np.savez_compressed(out / "oof_feats.npz", eid=np.concatenate(eids), feat=np.concatenate(feats), prob=np.concatenate(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["folds", "extract"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds")
    ap.add_argument("--outdir")
    ap.add_argument("--weights")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode == "folds":
        folds(a)
    else:
        z = np.load(a.data, allow_pickle=True)
        fe, pr = extract(torch.from_numpy(z["X"]), a.weights)
        np.savez_compressed(a.out, eid=z["eid"], part=z["part"], feat=fe, prob=pr)
        print("extracted", len(fe))


if __name__ == "__main__":
    main()
