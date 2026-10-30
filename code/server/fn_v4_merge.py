"""design notes: one feature table for M5e: out-of-fold CNN features/probabilities for the training events, seed-0
full-train CNN for every other event.

    python fn_v4_merge.py oof_feats.npz full_fn.npz feats_v4.npz       (ForestNet)
    python fn_v4_merge.py --append feats_v4.npz full_e2.npz feats_v4.npz  (add E2)
"""
import sys

import numpy as np

args = sys.argv[1:]
if args[0] == "--append":
    a, b, out = np.load(args[1], allow_pickle=True), np.load(args[2], allow_pickle=True), args[3]
    eid = np.concatenate([a["eid"], b["eid"]])
    assert len(set(eid.tolist())) == len(eid), "eid clash"
    np.savez_compressed(out, eid=eid, feat=np.concatenate([a["feat"], b["feat"]]), prob=np.concatenate([a["prob"], b["prob"]]),
                        source=np.concatenate([a["source"], np.array(["full"] * len(b["eid"]))]))
    print("appended", len(b["eid"]), "->", len(eid))
else:
    oof, full, out = np.load(args[0], allow_pickle=True), np.load(args[1], allow_pickle=True), args[2]
    o = set(oof["eid"].tolist())
    keep = np.array([e not in o for e in full["eid"].tolist()])
    eid = np.concatenate([oof["eid"], full["eid"][keep]])
    np.savez_compressed(out, eid=eid, feat=np.concatenate([oof["feat"], full["feat"][keep]]),
                        prob=np.concatenate([oof["prob"], full["prob"][keep]]),
                        source=np.array(["oof"] * len(oof["eid"]) + ["full"] * int(keep.sum())))
    print(f"merged: {len(oof['eid'])} out-of-fold + {int(keep.sum())} full = {len(eid)}")
