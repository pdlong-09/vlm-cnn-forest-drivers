"""Driver attribution figures for the manuscript (Figure 1 is TikZ in peerjcs/figsrc/fig1_pipeline.tex).

    .venv/Scripts/python paper/rs_forestnet/scripts/make_figures_a.py [names...]

fig2_study_area   ForestNet events and the Sims sets (Indonesia, E2, E3) on a tropical land outline (Natural Earth 1:110m)
fig3_card         one evidence card (image grid and fact sheet) of a ForestNet training event
fig4_transfer     in-domain versus cross-continent: binary balanced accuracy (BA) of the CNN and the VLM by set and region,
                  and the VLM - CNN difference per set and seed (paired bootstrap, 2,000 resamples)
fig6_by_k         performance by year of accumulated evidence k (ForestNet test macro-F1; E2 and Sims Indonesia BA)
Reads only scored outputs (runs/rs_forestnet*); writes peerjcs/figures/<name>.pdf and .png (300 dpi).
Palette: validated categorical slots (dataviz reference palette, light mode), fixed order.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
sys.path.insert(0, str(ROOT / "paper/rs_forestnet/scripts"))

matplotlib.use("Agg")
OUT = ROOT / "paper/rs_forestnet/peerjcs/figures"
RUNS = ROOT / "runs"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF, LAND = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#efeee9"
CM = 1 / 2.54
C_CNN, C_VLM, C_ENS = SLOTS[0], SLOTS[1], SLOTS[2]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})
OFF = {"sidn": 100000, "e2": 200000, "e3": 300000}
SIMS = {t: ROOT / f"data/interim/{d}" for t, d in (("sidn", "rs_forestnet_sims"), ("e2", "rs_forestnet_sims2"),
                                                   ("e3", "rs_forestnet_sims3"))}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def agri(P):
    return P[..., 0] + P[..., 1]


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def npz(path, eids, k=None):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    P = np.stack([probs[pos[int(e)]] for e in eids])
    return P[:, k] if (k is not None and P.ndim == 3) else P


def boot_diff(y, pa, pb, seed=20261021, B=2000):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = [(lambda i: ba(y[i], pa[i]) - ba(y[i], pb[i]))(rng.integers(0, n, n)) for _ in range(B)]
    return ba(y, pa) - ba(y, pb), np.percentile(d, [2.5, 97.5])


# ---------------------------------------------------------------- figure 2
def fig2_study_area():
    import geopandas as gpd
    from rsdrivers.io import events
    land = gpd.read_file("zip://" + str(ROOT / "data/raw/basemap_ne/ne_110m_land.zip"))
    fn = events()
    fig, ax = plt.subplots(figsize=(17.5 * CM, 6.6 * CM))
    land.plot(ax=ax, color=LAND, edgecolor="#b9b8b3", linewidth=0.3, zorder=1)
    ax.axhspan(-23.44, 23.44, color="#f7f6f2", zorder=0)
    sets = [("e3", "Held-out test E3 (n = 2306)", SLOTS[1], 4), ("e2", "Regional validation E2 (n = 1500)", SLOTS[0], 4),
            ("sidn", "Indonesia outside ForestNet (n = 548)", SLOTS[2], 5)]
    for t, lab, c, s in sets:
        ev = pd.read_csv(SIMS[t] / "events.csv", usecols=["Latitude", "Longitude"])
        ax.scatter(ev.Longitude, ev.Latitude, s=s, color=c, lw=0.25, edgecolor="white", label=lab, zorder=3)
    ax.scatter(fn.longitude, fn.latitude, s=5, marker="^", color=INK, lw=0, label=f"ForestNet events (n = {len(fn)})", zorder=4)
    for y_ in (-23.44, 23.44):
        ax.axhline(y_, color=INK2, lw=0.4, ls=(0, (3, 3)), zorder=2)
    ax.set_xlim(-100, 160)
    ax.set_ylim(-32, 32)
    ax.set_aspect("equal")
    ax.set_xticks(range(-90, 161, 30))
    ax.set_xticklabels([f"{abs(v)}°{'W' if v < 0 else 'E' if v > 0 else ''}" for v in range(-90, 161, 30)])
    ax.set_yticks([-20, 0, 20])
    ax.set_yticklabels(["20°S", "0°", "20°N"])
    for s in ax.spines.values():
        s.set_color("#b9b8b3")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, handletextpad=0.3, columnspacing=2.0,
              markerscale=2.2)
    save(fig, "fig2_study_area")


# ---------------------------------------------------------------- figure 3
def fig3_card(eid=77):
    from rsdrivers.card import fact_sheet, grid_image
    from rsdrivers.features import series_wide, usable
    from rsdrivers.io import INTERIM, events
    ev = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    if eid is None:  # a training plantation event with a cloud-free chip in every year -1..4
        for r in ev[(ev.part == "train") & (ev.merged_label == "Plantation")].itertuples():
            ok = all((INTERIM / "chips" / str(r.eid) / f"k{t}_{int(r.year) + t}.png").exists() for t in range(-1, 5))
            if ok and int(r.year) >= 2008 and r.eid in wide.index:
                eid = r.eid
                break
    r = ev.set_index("eid").loc[eid]
    r = r.copy()
    r["eid"] = eid
    row = pd.Series(r)
    img = grid_image(row, 4, scale=2)
    facts = fact_sheet(row, wide.loc[eid].to_dict(), static.loc[eid], 4)
    fig = plt.figure(figsize=(17.5 * CM, 6.6 * CM))
    ax = fig.add_axes([0.0, 0.0, 0.47, 0.90])
    ax.imshow(np.asarray(img))
    ax.axis("off")
    tx = fig.add_axes([0.50, 0.0, 0.50, 0.90])
    tx.axis("off")
    fig.text(0.0, 0.95, "(a) Image grid: yearly composites, years −1 to +4", fontsize=8, va="bottom")
    fig.text(0.50, 0.95, "(b) Fact sheet and question (options shortened)", fontsize=8, va="bottom")
    lines = []
    for l in facts.split("\n"):
        lines += textwrap.wrap(l, 74) if len(l) > 74 else [l]
    lines += ["", "What is the direct driver of this forest loss?", "A) large-scale plantation ...  B) smallholder agriculture ...",
              "C) grassland or shrubland ...  D) other ...", "Answer with the letter only."]
    tx.text(0.0, 1.0,"\n".join(lines), family="DejaVu Sans Mono", fontsize=5.6, va="top", ha="left", color=INK,
            linespacing=1.25, transform=tx.transAxes)
    save(fig, "fig3_card")
    print("card event", eid, "label", r.merged_label, "year", int(r.year))


# ---------------------------------------------------------------- figure 4
def transfer_data():
    from rsdrivers.io import INTERIM, events
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    te = fn[fn.part == "test"]
    yfn = te.merged_label.isin(["Plantation", "Smallholder agriculture"]).astype(int).to_numpy()
    m2 = (agri(npz(RUNS / "rs_forestnet_v3/probs/M2.npz", te.eid, 4)) >= 0.5).astype(int)  # the CNN used on every set
    m3 = (agri(npz(RUNS / "rs_forestnet/probs/M3c_val_cal+test.npz", te.eid, 4)) >= 0.5).astype(int)
    rows = [("ForestNet test\n(in domain)", "all", yfn, m2, m3)]
    sidn = pd.read_csv(SIMS["sidn"] / "events.csv")
    ids = OFF["sidn"] + sidn.sid
    rows.append(("Indonesia outside\nForestNet", "all", sidn.label_agri.to_numpy(),
                 (agri(npz(RUNS / "rs_forestnet_v3/probs/M2.npz", ids, 4)) >= 0.5).astype(int),
                 (agri(npz(RUNS / "rs_forestnet_v3/probs/M3c.npz", ids, 4)) >= 0.5).astype(int)))
    for t, lab, p2, p3 in (("e2", "E2, other continents\n(regional validation)", RUNS / "rs_forestnet_v4/probs/M2_e2.npz",
                            RUNS / "rs_forestnet_v4/probs/M3c.npz"),
                           ("e3", "E3, other continents\n(held-out test)", RUNS / "rs_forestnet_v5/probs/M2_E3.npz",
                            RUNS / "rs_forestnet_v5/probs/M3c_E3.npz")):
        ev = pd.read_csv(SIMS[t] / "events.csv")
        ids = OFF[t] + ev.sid
        y = ev.label_agri.to_numpy()
        a2 = (agri(npz(p2, ids, 4)) >= 0.5).astype(int)
        a3 = (agri(npz(p3, ids, 4)) >= 0.5).astype(int)
        rows.append((lab, "all", y, a2, a3))
        for g in ("Africa", "Asia-Pacific", "Latin America"):
            i = (ev.group == g).to_numpy()
            rows.append((lab, g, y[i], a2[i], a3[i]))
    return rows


def fig4_transfer():
    rows = transfer_data()
    seeds = json.loads((RUNS / "rs_forestnet_v5/eval_v5_seeds.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(17.5 * CM, 7.6 * CM), gridspec_kw={"width_ratios": [1.15, 1]})
    # (a) BA of CNN and VLM: overall per set, and by region for E2 and E3
    labels, yy = [], []
    y0 = 0
    for lab, g, y, p2, p3 in rows:
        name = lab.replace("\n", " ") if g == "all" else f"   {g}"
        labels.append(name if g == "all" else g)
        yy.append(y0)
        b2, b3 = ba(y, p2), ba(y, p3)
        a.plot([b2, b3], [y0, y0], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
        a.scatter([b2], [y0], s=26 if g == "all" else 14, color=C_CNN, zorder=3, edgecolor="white", lw=0.6)
        a.scatter([b3], [y0], s=26 if g == "all" else 14, color=C_VLM, zorder=3, edgecolor="white", lw=0.6)
        y0 += 1 if g != "Latin America" else 1.6
        if g == "all" and lab.startswith(("ForestNet", "Indonesia")):
            y0 += 0.6
    a.set_yticks(yy)
    a.set_yticklabels([l if not l.startswith(("Africa", "Asia", "Latin")) else "    " + l for l in labels], fontsize=7)
    for tick, (lab, g, *_rest) in zip(a.get_yticklabels(), rows):
        tick.set_color(INK if g == "all" else INK2)
    a.invert_yaxis()
    a.set_xlim(0.58, 0.88)
    a.set_xlabel("Balanced accuracy, agricultural conversion (k = 4)")
    a.set_title("(a) CNN and fine-tuned VLM by evaluation set", loc="left")
    a.scatter([], [], s=26, color=C_CNN, label="CNN (ResNet-18)")
    a.scatter([], [], s=26, color=C_VLM, label="VLM (Qwen3.5-9B, QLoRA)")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False, handletextpad=0.2)
    clean(a)
    # (b) VLM - CNN with 95% CI: in domain, Indonesia, E2 and E3 (three seeds of the VLM on E2 and E3)
    items = []
    for lab, g, y, p2, p3 in rows:
        if g == "all" and lab.startswith(("ForestNet", "Indonesia")):
            d, ci = boot_diff(y, p3, p2)
            items.append((lab.replace("\n", " "), d, ci, INK2, "o"))
    for t, lab in (("E2", "E2 (regional validation)"), ("E3", "E3 (held-out test)")):
        for s in (0, 1, 2):
            r = seeds[t][f"M3c seed {s}"]
            items.append((f"{lab}, seed {s}", r["minus M2"], r["ci"], C_VLM, "o"))
    yb = np.arange(len(items))[::-1]
    for (lab, d, ci, c, m), y_ in zip(items, yb):
        b.plot(ci, [y_, y_], color=c, lw=1.6, solid_capstyle="round")
        b.scatter([d], [y_], s=24, color=c, zorder=3, edgecolor="white", lw=0.6, marker=m)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_yticks(yb)
    b.set_yticklabels([i[0] for i in items], fontsize=7)
    b.set_xlabel("VLM − CNN, balanced accuracy (95% CI)")
    b.set_title("(b) Difference by set and VLM seed", loc="left")
    b.set_xlim(-0.08, 0.11)
    clean(b)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "fig4_transfer")
    for lab, g, y, p2, p3 in rows:
        print(f"{lab.replace(chr(10), ' '):45s} {g:14s} n={len(y):5d} CNN {ba(y, p2):.3f} VLM {ba(y, p3):.3f}")
    for it in items:
        print(it[0], round(it[1], 4), np.round(it[2], 4))


# ---------------------------------------------------------------- figure 6
def fig6_by_k():
    from rsdrivers.features import LABELS4
    from rsdrivers.io import INTERIM, events
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    te = fn[fn.part == "test"]
    yfn = te.merged_label.map(LABELS4.index).to_numpy()

    def mf1_by_k(P):
        out = []
        for k in range(5):
            p = P[:, k].argmax(1)
            f = [0.0 if ((p == c) & (yfn == c)).sum() == 0 else
                 2 * ((p == c) & (yfn == c)).sum() / (2 * ((p == c) & (yfn == c)).sum() + ((p == c) & (yfn != c)).sum()
                                                      + ((p != c) & (yfn == c)).sum()) for c in range(4)]
            out.append(float(np.mean(f)))
        return out
    fP = {m: npz(RUNS / f, te.eid) for m, f in (("M1", "rs_forestnet/probs/M1.npz"), ("M2", "rs_forestnet_v3/probs/M2.npz"),
                                               ("M3c", "rs_forestnet/probs/M3c_val_cal+test.npz"),
                                               ("M3a", "rs_forestnet/probs/M3a_val_cal+test.npz"))}
    fP["M4"] = (fP["M2"] + fP["M3c"]) / 2
    v4 = json.loads((RUNS / "rs_forestnet_v4/eval_v4.json").read_text(encoding="utf-8"))
    v3 = json.loads((RUNS / "rs_forestnet_v3/eval_v3.json").read_text(encoding="utf-8"))
    k = np.arange(5)
    fig, axs = plt.subplots(1, 3, figsize=(17.5 * CM, 5.6 * CM), sharex=True)
    panels = [
        (axs[0], "(a) ForestNet test, 4 classes", "Macro-F1",
         [("Tabular (LightGBM)", mf1_by_k(fP["M1"]), SLOTS[3]), ("CNN", mf1_by_k(fP["M2"]), C_CNN),
          ("VLM", mf1_by_k(fP["M3c"]), C_VLM), ("CNN + VLM", mf1_by_k(fP["M4"]), C_ENS),
          ("VLM zero-shot", mf1_by_k(fP["M3a"]), "#9a9893")]),
        (axs[1], "(b) Indonesia outside ForestNet", "Balanced accuracy",
         [("CNN", v3["by_model"]["M2"]["balanced_acc_by_k"], C_CNN), ("VLM", v3["by_model"]["M3c"]["balanced_acc_by_k"], C_VLM),
          ("CNN + VLM", v3["by_model"]["M4"]["balanced_acc_by_k"], C_ENS)]),
        (axs[2], "(c) E2, other continents", "Balanced accuracy",
         [("CNN", v4["by_model"]["M2"]["BA_by_k"], C_CNN), ("VLM", v4["by_model"]["M3c"]["BA_by_k"], C_VLM),
          ("CNN + VLM", v4["by_model"]["M4"]["BA_by_k"], C_ENS)]),
    ]
    for ax, title, ylab, series in panels:
        for lab, vals, c in series:
            ls = (0, (2, 2)) if "zero" in lab else "-"
            ax.plot(k, vals, color=c, lw=1.8, ls=ls, marker="o", ms=3.4, mec="white", mew=0.5, label=lab)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylab)
        ax.set_xticks(k)
        clean(ax, "y")
    axs[0].set_ylim(0.1, 0.82)
    for ax in axs[1:]:
        ax.set_ylim(0.62, 0.78)
    axs[2].set_ylabel("")
    fig.supxlabel("Years of evidence accumulated after the loss (k)", fontsize=8, y=0.1)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, frameon=False)
    fig.subplots_adjust(wspace=0.32, bottom=0.3)
    save(fig, "fig6_by_k")


# ---------------------------------------------------------------- figure 5
def fig5_early_stop():
    """Learn-then-test early stop (alpha 0.05, delta 0.10). (a) Source threshold, evaluated on whole target sets
    (early_stop_source.json, independent of the calibration data). (b, c) Local threshold calibrated on n cells of some
    clusters and evaluated on the other clusters (early_stop_split.json: countries in E2, 1-degree cells in Indonesia)."""
    c = json.loads((RUNS / "rs_forestnet_robust/early_stop_source.json").read_text(encoding="utf-8"))
    sp = json.loads((RUNS / "rs_forestnet_robust/early_stop_split.json").read_text(encoding="utf-8"))
    sets = [("ForestNet test (4 classes)", "ForestNet test"), ("Indonesia outside ForestNet", "Indonesia"),
            ("E2 Africa", "E2 Africa"), ("E2 Asia-Pacific", "E2 Asia-Pacific"), ("E2 Latin America", "E2 Latin America")]
    models = [("CNN", "CNN", "CNN", C_CNN), ("VLM", "VLM", "VLM", C_VLM), ("CNN + VLM", "CNN + VLM", "CNN + VLM", C_ENS)]
    fig, (a, b, d) = plt.subplots(1, 3, figsize=(17.5 * CM, 6.4 * CM), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    y = np.arange(len(sets))[::-1] * 1.0
    off = {"CNN": 0.22, "VLM": 0.0, "CNN + VLM": -0.22}
    for m, lab, _, col in models:
        a.scatter([c[s][m]["gap"] for s, _ in sets], y + off[m], s=24, color=col, edgecolor="white", lw=0.5, zorder=3,
                  label=lab)
    a.axvline(0.05, color=INK2, lw=0.8, ls=(0, (3, 2)))
    a.text(0.0505, y[0] + 0.55, "α = 0.05", fontsize=6.5, color=INK2, va="bottom")
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in sets])
    a.set_xlim(0, 0.105)
    a.set_ylim(-0.6, y[0] + 0.85)
    a.set_xlabel("Gap: share losing a correct attribution")
    a.set_title("(a) Threshold from ForestNet", loc="left")
    clean(a)
    ns = [50, 100, 200, 400]
    pos = {n: i for i, n in enumerate(ns)}
    for tset, ls in (("Indonesia", (0, (3, 1.5))), ("E2", "-")):
        for _, _, mk, col in models:
            r = sp[tset]["models"][mk]
            xs = [n for n in ns if str(n) in r]
            px = [pos[n] for n in xs]
            b.plot(px, [r[str(n)]["share_test_gap_above_alpha"] for n in xs], color=col, lw=1.4, ls=ls, marker="o", ms=2.8,
                   mec="white", mew=0.4)
            d.plot(px, [r[str(n)]["mean_stop_year"] for n in xs], color=col, lw=1.4, ls=ls, marker="o", ms=2.8, mec="white", mew=0.4)
    b.axhline(0.10, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.text(-0.25, 0.103, "δ = 0.10", fontsize=6.5, color=INK2, ha="left", va="bottom")
    b.set_ylim(0, 0.23)
    b.set_ylabel("Repetitions with held-out gap > α")
    b.set_title("(b) Local threshold: held-out gap", loc="left")
    d.set_ylabel("Mean stopping year")
    d.set_ylim(0, 2.0)
    d.set_title("(c) Local threshold: stop year", loc="left")
    for ax in (b, d):
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlim(-0.3, len(ns) - 0.7)
        ax.set_xlabel("Local calibration cells")
        clean(ax, "y")
    h1, l1 = a.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h2 = [Line2D([], [], color=INK2, lw=1.4, ls="-"), Line2D([], [], color=INK2, lw=1.4, ls=(0, (3, 1.5)))]
    fig.legend(h1 + h2, l1 + ["E2 (held-out countries)", "Indonesia (held-out 1° cells)"], loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.48, bottom=0.27)
    save(fig, "fig5_early_stop")


def main():
    names = sys.argv[1:] or ["fig2_study_area", "fig3_card", "fig4_transfer", "fig5_early_stop", "fig6_by_k"]
    OUT.mkdir(parents=True, exist_ok=True)
    for n in names:
        globals()[n]()


if __name__ == "__main__":
    main()
