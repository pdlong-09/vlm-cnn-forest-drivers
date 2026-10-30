"""Driver attribution fair-comparison CNNs (26/10/2026, server2): the M2 recipe of cnn_train.py (ResNet-18 from ImageNet, years
-1..4 stacked as 18 channels with the years after k zeroed, class-balanced loss, AdamW 3e-4, cosine schedule, 40 epochs,
batch 64, flips and rot90, a random k per sample, the epoch with the best mean macro-F1 over k on val_sel) with
  img3km: the full 3 km chips (128 px) that the VLM sees, instead of the central 2.25 km;
  facts:  the full chips plus the numbers of the VLM fact sheet at step k (tab_*.npz: T [n, 5, F], NaN where missing),
          standardised with the training events, missing values set to 0 with a missingness indicator, through an MLP
          (2F -> 128 -> 128, GELU) concatenated with the 512-d image feature before the classifier.

    python cnn_fair.py --variant img3km|facts --seed 0 --data chips_fn.npz --tab tab_fn.npz \
        --ext sidn=chips_sidn.npz:tab_sidn.npz e2=chips_e2.npz:tab_e2.npz e3=chips_e3.npz:tab_e3.npz --outdir out

Writes <outdir>/<variant>_s<seed>_{fn,sidn,e2,e3}.npz with probabilities for every event and step k = 0..4.
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


class Fused(nn.Module):
    def __init__(self, n_tab):
        super().__init__()
        self.cnn, pre = model()
        assert pre, "ImageNet weights not found (TORCH_HOME)"
        self.cnn.fc = nn.Identity()
        self.tab = nn.Sequential(nn.Linear(2 * n_tab, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.GELU())
        self.head = nn.Linear(512 + 128, 4)

    def forward(self, x, t):
        return self.head(torch.cat([self.cnn(x), self.tab(t)], 1))


def tab_input(T, k, mean, std):
    """T [b, 5, F] numpy (NaN = missing), k [b] -> float tensor [b, 2F]: standardised values (0 if missing) and mask."""
    v = T[np.arange(len(k)), k]
    m = ~np.isnan(v)
    x = np.where(m, (v - mean) / std, 0.0)
    return torch.from_numpy(np.concatenate([x, m], 1).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["img3km", "facts"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tab", required=True)
    ap.add_argument("--ext", nargs="*", default=[])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    z = np.load(a.data, allow_pickle=True)
    X, y, part, eid = torch.from_numpy(z["X"]), z["y"].astype(int), z["part"], z["eid"]  # read each key once
    zt = np.load(a.tab, allow_pickle=True)
    T, teid = zt["T"], zt["eid"]
    assert (teid == eid).all()
    tr, sel = np.flatnonzero(part == "train"), np.flatnonzero(part == "val_sel")
    ref = T[tr, 4]
    mean, std = np.nanmean(ref, 0), np.nanstd(ref, 0)
    mean, std = np.nan_to_num(mean), np.where(np.nan_to_num(std) > 0, np.nan_to_num(std), 1.0)
    facts = a.variant == "facts"
    if facts:
        net = Fused(T.shape[2]).to(dev)
    else:
        net, pre = model()
        assert pre, "ImageNet weights not found (TORCH_HOME)"
        net = net.to(dev)

    def fwd(xb, k, Tb):
        x = to_input(xb, k)
        return net(x, tab_input(Tb, k.cpu().numpy(), mean, std).to(dev)) if facts else net(x)

    def predict(Xs, Ts):
        net.eval()
        out = np.zeros((len(Xs), 5, 4))
        with torch.no_grad():
            for s in range(0, len(Xs), 128):
                xb = Xs[s:s + 128].to(dev)
                for k in range(5):
                    kk = torch.full((len(xb),), k, device=dev)
                    out[s:s + len(xb), k] = torch.softmax(fwd(xb, kk, Ts[s:s + 128]), 1).cpu().numpy()
        return out

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
            loss = crit(fwd(xb, k, T[b]), yt[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = predict(X[sel], T[sel])
        f1 = np.mean([macro_f1(y[sel], p[:, k].argmax(1)) for k in range(5)])
        if f1 > best:
            best, state = f1, {k_: v.detach().clone() for k_, v in net.state_dict().items()}
        print(f"epoch {ep}: loss {loss.item():.3f} val_sel mean macro-F1 {f1:.3f} (best {best:.3f}) {time.time() - t0:.0f} s",
              flush=True)
    net.load_state_dict(state)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.variant}_s{a.seed}"
    np.savez_compressed(out / f"{tag}_fn.npz", probs=predict(X, T), eid=eid, part=part, y=y, best_val_sel=best)
    for item in a.ext:
        name, paths = item.split("=", 1)
        pc, pt = paths.split(":")
        ze, zte = np.load(pc, allow_pickle=True), np.load(pt, allow_pickle=True)
        Xe, ee, Te = torch.from_numpy(ze["X"]), ze["eid"], zte["T"]
        assert (zte["eid"] == ee).all()
        np.savez_compressed(out / f"{tag}_{name}.npz", probs=predict(Xe, Te), eid=ee)
    print(f"done {tag}: best val_sel {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
