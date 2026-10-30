"""Driver attribution, M2 on server1 GPU 1: ResNet-18 on stacked yearly chips (years -1..4, 18 channels), later years zeroed.

    python cnn_train.py --data chips.npz --out probs.npz [--epochs 40] [--seed 0]

chips.npz: X uint8 [n, 6, H, W, 3] (years -1..4), y [n], part [n] (train / val_sel / val_cal / test), eid [n].
Training uses a random step k in 0..4 per sample (years after k set to 0), class-balanced loss, flips/rot90.
The checkpoint with the best val_sel macro-F1 (mean over k) is kept; probabilities for every event and step are saved.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision


def macro_f1(y, p, c=4):
    f = []
    for i in range(c):
        tp = ((p == i) & (y == i)).sum(); fp = ((p == i) & (y != i)).sum(); fn = ((p != i) & (y == i)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def model(in_ch=18, n_cls=4):
    try:
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        pre = True
    except Exception:
        m = torchvision.models.resnet18(weights=None)
        pre = False
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w.repeat(1, in_ch // 3, 1, 1) / (in_ch // 3)
    m.fc = nn.Linear(512, n_cls)
    return m, pre


def to_input(xb, k):
    """xb uint8 [b, 6, H, W, 3], k [b] -> float [b, 18, H, W] with years after k zeroed."""
    x = xb.float().div(255.0)
    keep = (torch.arange(6, device=x.device)[None, :] <= (k[:, None] + 1)).float()
    x = x * keep[:, :, None, None, None]
    return x.permute(0, 1, 4, 2, 3).reshape(x.shape[0], 18, x.shape[2], x.shape[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
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

    def predict(idx):
        net.eval()
        out = np.zeros((len(idx), 5, 4))
        with torch.no_grad():
            for s in range(0, len(idx), 128):
                b = idx[s:s + 128]
                xb = X[b].to(dev)
                for k in range(5):
                    kk = torch.full((len(b),), k, device=dev)
                    out[s:s + len(b), k] = torch.softmax(net(to_input(xb, kk)), 1).cpu().numpy()
        return out

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
        p = predict(sel)
        f1 = np.mean([macro_f1(y[sel], p[:, k].argmax(1)) for k in range(5)])
        if f1 > best:
            best, state = f1, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"epoch {ep}: loss {loss.item():.3f} val_sel mean macro-F1 {f1:.3f} (best {best:.3f}) {time.time() - t0:.0f} s",
              flush=True)
    net.load_state_dict(state)
    probs = predict(np.arange(len(y)))
    np.savez_compressed(a.out, probs=probs, eid=z["eid"], part=part, y=y, pretrained=pre, best_val_sel=best)
    print("done; imagenet init:", pre)


if __name__ == "__main__":
    main()
