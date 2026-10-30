"""Driver attribution v3: the v1 M2 CNN (server/cnn_train.py, imported unchanged) with the weights kept, so the external set can be
predicted in a later batch (design notes; v1 saved only probabilities).

    python cnn_v3.py train   --data chips.npz --out probs.npz --weights w.pt [--epochs 40] [--seed 0]
    python cnn_v3.py predict --data ext.npz --weights w.pt --out probs.npz

train: exactly the loop of cnn_train.main (same model, loss, optimiser, schedule, augmentation, val_sel selection);
the chosen state is also written to --weights. predict: probabilities for every event and step of --data.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_train import macro_f1, model, to_input  # noqa: E402


def predict_all(net, X, dev="cuda"):
    net.eval()
    out = np.zeros((len(X), 5, 4))
    with torch.no_grad():
        for s in range(0, len(X), 128):
            xb = X[s:s + 128].to(dev)
            for k in range(5):
                out[s:s + len(xb), k] = torch.softmax(net(to_input(xb, torch.full((len(xb),), k, device=dev))), 1).cpu().numpy()
    return out


def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    z = np.load(a.data, allow_pickle=True)
    X, y, part = torch.from_numpy(z["X"]), z["y"].astype(int), z["part"]
    dev = "cuda"
    tr, sel = np.flatnonzero(part == "train"), np.flatnonzero(part == "val_sel")
    net, pre = model()
    net = net.to(dev)
    w = torch.tensor(len(tr) / (4 * np.bincount(y[tr], minlength=4)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    yt = torch.from_numpy(y)
    best, state, t0 = -1, None, time.time()
    for ep in range(a.epochs):
        net.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            xb = X[b].to(dev)
            if np.random.rand() < 0.5:
                xb = xb.flip(3)
            if np.random.rand() < 0.5:
                xb = xb.flip(2)
            xb = torch.rot90(xb, int(np.random.randint(4)), dims=(2, 3))
            k = torch.randint(0, 5, (len(b),), device=dev)
            loss = crit(net(to_input(xb, k)), yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = predict_all(net, X[sel])
        f1 = np.mean([macro_f1(y[sel], p[:, k].argmax(1)) for k in range(5)])
        if f1 > best:
            best, state = f1, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep}: loss {loss.item():.3f} val_sel mean macro-F1 {f1:.3f} (best {best:.3f}) {time.time() - t0:.0f} s",
              flush=True)
    net.load_state_dict(state)
    torch.save(state, a.weights)
    np.savez_compressed(a.out, probs=predict_all(net, X), eid=z["eid"], part=part, y=y, pretrained=pre, best_val_sel=best)
    print("done; imagenet init:", pre)


def predict(a):
    z = np.load(a.data, allow_pickle=True)
    net, _ = model()
    net.load_state_dict(torch.load(a.weights, map_location="cuda"))
    net = net.to("cuda")
    np.savez_compressed(a.out, probs=predict_all(net, torch.from_numpy(z["X"])), eid=z["eid"], part=z["part"])
    print("predicted", len(z["eid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (train if a.mode == "train" else predict)(a)


if __name__ == "__main__":
    main()
