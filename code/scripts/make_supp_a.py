"""Driver attribution supplementary tables (S1-S3) from the scored outputs; writes peerjcs/supplementary/tables/*.tex and the
example card text (tables/card_example.tex).

    .venv/Scripts/python paper/rs_forestnet/scripts/make_supp_a.py

Reads only runs/rs_forestnet* and data/interim; no label is used for any choice here.
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper/rs_forestnet"))
from rsdrivers.card import fact_sheet, question  # noqa: E402
from rsdrivers.features import LABELS4, series_wide, usable  # noqa: E402
from rsdrivers.io import INTERIM, events  # noqa: E402

RUNS = ROOT / "runs"
OUT = Path(os.environ.get("SUPP_OUT", ROOT / "paper/rs_forestnet/peerjcs/supplementary/tables"))
RELEASE = ROOT / "release/rs_forestnet_release"
SIMS = {"Indonesia": "rs_forestnet_sims", "E2": "rs_forestnet_sims2", "E3": "rs_forestnet_sims3"}
GROUPS = ["Africa", "Asia-Pacific", "Latin America"]


def j(path):
    return json.loads((RUNS / path).read_text(encoding="utf-8"))


def r3(x):
    # half-up, as in the manuscript (stored values are often rounded to 4 decimals already, e.g. 0.0465)
    return Decimal(str(float(x))).quantize(Decimal("0.001"), ROUND_HALF_UP)


def minus(s):
    return s.replace("-", "$-$")


def f3(x):
    return "--" if x is None else minus(f"{r3(x)}")


def ci(c):
    return f"{minus(str(r3(c[0])))}--{minus(str(r3(c[1])))}"


def signed(x):
    v = r3(x)
    return f"+{v}" if v >= 0 else minus(f"{v}")


MAPPING = {  # Sims et al. driver -> (closest ForestNet classes, agreement of the binary classes)
    "Permanent agriculture": ("plantation; smallholder agriculture; pasture would be grassland or shrubland",
                              "yes, except pasture"),
    "Shifting cultivation": ("smallholder agriculture; regrowth would be other (secondary forest)", "mostly"),
    "Logging": ("other (logging); timber plantations would be plantation", "no for timber plantations"),
    "Hard commodities": ("other (mining)", "yes"),
    "Wildfire": ("other; grassland or shrubland where it persists", "yes"),
    "Settlements & infrastructure": ("other", "yes"),
    "Other natural disturbances": ("other", "yes"),
}

COUNTRY = {"Dem Rep of the Congo": "Democratic Republic of the Congo", "Burma": "Myanmar",
           "Cote d'Ivoire": r"C\^{o}te d'Ivoire"}


def write(name, caption, label, colspec, header, rows, note=None, size=r"\small"):
    lines = [r"\begin{table}[H]", r"\centering", size, rf"\caption{{{caption}\label{{{label}}}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r if r.startswith(("\\midrule", "\\cmidrule")) else r + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def npz(path, eids):
    z = np.load(path, allow_pickle=True)
    probs, eid = z["probs"], z["eid"]  # read each key once
    pos = {int(e): i for i, e in enumerate(eid)}
    return np.stack([probs[pos[int(e)]] for e in eids])


def macro_f1(y, p, c=4):
    out = []
    for i in range(c):
        tp = ((p == i) & (y == i)).sum()
        fp = ((p == i) & (y != i)).sum()
        fn = ((p != i) & (y == i)).sum()
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out)), out


def ba(y, p):
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


# ---------------------------------------------------------------- S1
def s1_tables():
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    parts = ["train", "val_sel", "val_cal", "test"]
    rows = []
    for lab in LABELS4:
        rows.append(f"{lab} & " + " & ".join(str(int(((fn.merged_label == lab) & (fn.part == p)).sum())) for p in parts))
    rows.append(r"\midrule")
    rows.append("Total & " + " & ".join(str(int((fn.part == p).sum())) for p in parts))
    write("s1_forestnet", "ForestNet events by class and part. The validation set of the release was split at random, "
          "stratified by class, into a selection half and a calibration half.", "tab:s_fn",
          "lcccc", r"\textbf{Class} & \textbf{Training} & \textbf{Selection} & \textbf{Calibration} & \textbf{Test}", rows)

    rows = []
    for name, d in SIMS.items():
        ev = pd.read_csv(INTERIM.parent / d / "events.csv")
        grp = "Indonesia" if "group" not in ev else ", ".join(f"{g} {int((ev.group == g).sum())}" for g in GROUPS)
        nc = 1 if "country" not in ev else ev.country.nunique()
        rows.append(f"{name} & {len(ev)} & {int(ev.label_agri.sum())} ({ev.label_agri.mean() * 100:.0f}\\%) & {grp} & {nc} & "
                    f"{(ev.src == 'validation').sum()} & {(ev.Confidence_primary == 'High').sum()} & "
                    f"{np.median(ev.dist_fn_train_km):.0f} & {int(ev.year.min())}--{int(ev.year.max())}")
    write("s1_sets", "External evaluation sets drawn from the reference cells of Sims et al. Agriculture: permanent "
          "agriculture or shifting cultivation as the primary driver. Validation file: cells from the validation file of "
          "Sims et al.\\ rather than their training file. Distance: median distance to the nearest ForestNet training event.",
          "tab:s_sets", "lccp{3.4cm}ccccc",
          r"\textbf{Set} & \textbf{Cells} & \textbf{Agriculture} & \textbf{Regions} & \textbf{Countries} & "
          r"\textbf{Valid.\ file} & \textbf{High conf.} & \textbf{Dist.\ (km)} & \textbf{Years}", rows, size=r"\footnotesize")

    rows = []
    ev = pd.concat([pd.read_csv(INTERIM.parent / d / "events.csv").assign(set=n) for n, d in SIMS.items()])
    for drv, g in ev.groupby("Driver_primary"):
        fnc, match = MAPPING[drv]
        rows.append(f"{drv.replace('&', chr(92) + '&')} & {'agriculture' if g.label_agri.iloc[0] == 1 else 'other'} & {fnc} & "
                    f"{match} & " + " & ".join(str(int((g.set == s).sum())) for s in SIMS))
    write("s1_drivers", "Primary drivers of Sims et al.\\ in the three sets, their binary class, the closest ForestNet classes "
          "and whether the binary classes of the two schemes agree. ForestNet counts plantation (oil palm, timber and other "
          "large-scale plantations) and smallholder agriculture as conversion to agriculture, and grassland or shrubland and "
          "other (mining, fish ponds, logging, secondary forest and other) as other drivers.", "tab:s_drivers",
          "llp{4.3cm}p{3.6cm}ccc", r"\textbf{Driver of Sims et al.} & \textbf{Binary} & \textbf{Closest ForestNet classes} & "
          r"\textbf{Binary agreement} & \textbf{Indon.} & \textbf{E2} & \textbf{E3}", rows, size=r"\footnotesize")


def card_example(eid=77, k=4):
    ev = events().set_index("eid")
    wide = series_wide(pd.read_csv(INTERIM / "series_gee.csv"))
    static = usable(pd.read_parquet(INTERIM / "static_features.parquet").set_index("eid"))
    row = ev.loc[eid].copy()
    row["eid"] = eid
    facts = fact_sheet(pd.Series(row), wide.loc[eid].to_dict(), static.loc[eid], k)
    q, _ = question(LABELS4)
    text = facts + "\n\n" + q
    esc = text.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("_", r"\_")
    body = "\n".join(ln + r"\\" if ln else r"\par\smallskip" for ln in esc.split("\n"))
    (OUT / "card_example.tex").write_text(r"{\ttfamily\footnotesize\raggedright " + body + "}\n", encoding="utf-8")
    print("wrote card_example")


# ---------------------------------------------------------------- S3
def s3_forestnet():
    fn = events().merge(pd.read_csv(INTERIM / "parts.csv"), on="eid")
    te = fn[fn.part == "test"]
    y = te.merged_label.map(LABELS4.index).to_numpy()
    yb = (y <= 1).astype(int)
    P = {"Tabular (LightGBM)": npz(RUNS / "rs_forestnet/probs/M1.npz", te.eid),
         "CNN": npz(RUNS / "rs_forestnet_v3/probs/M2.npz", te.eid),  # the CNN used on every set
         "VLM zero-shot": npz(RUNS / "rs_forestnet/probs/M3a_val_cal+test.npz", te.eid),
         "VLM eight examples": npz(RUNS / "rs_forestnet/probs/M3b_val_cal+test.npz", te.eid),
         "VLM (QLoRA)": npz(RUNS / "rs_forestnet/probs/M3c_val_cal+test.npz", te.eid)}
    P["CNN + VLM"] = (P["CNN"] + P["VLM (QLoRA)"]) / 2
    rows_k, rows_c = [], []
    for m, p in P.items():
        f_by_k = [macro_f1(y, p[:, k].argmax(-1))[0] for k in range(5)]
        f4, per = macro_f1(y, p[:, 4].argmax(-1))
        acc = (p[:, 4].argmax(-1) == y).mean()
        b = ba(yb, ((p[:, 4, 0] + p[:, 4, 1]) >= 0.5).astype(int))
        rows_k.append(f"{m} & " + " & ".join(f3(v) for v in f_by_k) + f" & {f3(acc)} & {f3(b)}")
        rows_c.append(f"{m} & " + " & ".join(f3(v) for v in per) + f" & {f3(f4)}")
    write("s3_fn_by_k", f"ForestNet test set ({len(te)} events): macro-F1 on the four classes by step $k$, accuracy at "
          "$k = 4$ and balanced accuracy (BA) of conversion to agriculture at $k = 4$.", "tab:s_fn_k", "lccccccc",
          r"\textbf{Model} & $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ & \textbf{Acc.} & \textbf{BA}", rows_k)
    write("s3_fn_class", "ForestNet test set: F1 by class at $k = 4$.", "tab:s_fn_class", "lccccc",
          r"\textbf{Model} & \textbf{Plantation} & \textbf{Smallholder} & \textbf{Grassland} & \textbf{Other} & \textbf{Macro-F1}",
          rows_c)


def s3_indonesia():
    v3 = j("rs_forestnet_v3/eval_v3.json")
    p13 = j("rs_forestnet_robust/parts13.json".replace("parts13", "robust_a_parts13"))["3_tabular_M1x"]["sidn"]
    bb = j("rs_forestnet_robust/backbones_eval.json")
    names = [("CNN (ResNet-18)", "M2"), ("VLM", "M3c"), ("CNN + VLM", "M4"), ("VLM, vision encoder tuned", "M3v"),
             ("CNN + VLM with tuned vision encoder", "M4v")]
    sub = [(">=10 km from ForestNet train", r"$\geq$10 km"), ("validation file only", "Valid.\\ file"),
           ("High confidence only", "High conf.")]
    rows = []
    for lab, m in names:
        r = v3["by_model"][m]
        rows.append(f"{lab} & " + " & ".join(f3(v) for v in r["balanced_acc_by_k"]) + " & "
                    + " & ".join(f3(v3["subsets"][s][m]) for s, _ in sub))
    rows.append(r"\midrule")
    for lab, key in (("CNN (ResNet-50)", "resnet50"), ("CNN (ConvNeXt-Tiny)", "convnext_tiny")):
        rows.append(f"{lab} & -- & -- & -- & -- & {f3(bb[key]['sidn'])} & -- & -- & --")
    rows.append(f"Tabular (LightGBM) & " + " & ".join(f3(v) for v in p13["M1x BA by k"]) + " & -- & -- & --")
    rows.append(f"Coordinates only (1-NN) & -- & -- & -- & -- & {f3(v3['by_model']['location 1-NN']['balanced_acc_k4'])} & "
                + " & ".join(f3(v3["subsets"][s]["location 1-NN"]) for s, _ in sub))
    n_sub = ", ".join(f"{t} {v3['subsets'][s]['n']}" for s, t in sub)
    write("s3_indonesia", f"Indonesia outside ForestNet ({v3['n']} cells): BA of conversion to agriculture by step $k$ and, "
          f"at $k = 4$, on subsets (cells: {n_sub}).", "tab:s_idn", "lcccccccc",
          r"\textbf{Model} & $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ & " + " & ".join(rf"\textbf{{{t}}}" for _, t in sub), rows,
          size=r"\footnotesize")


def s3_continents():
    v4, d5, f5 = j("rs_forestnet_v4/eval_v4.json"), j("rs_forestnet_v5/eval_v5_dev.json"), j("rs_forestnet_v5/eval_v5_final.json")
    p13 = j("rs_forestnet_robust/robust_a_parts13.json")["3_tabular_M1x"]
    bb = j("rs_forestnet_robust/backbones_eval.json")
    seeds = j("rs_forestnet_v5/eval_v5_seeds.json")
    rows = []

    def line(lab, e2, e3):
        def cell(r):
            if r is None:
                return "-- & -- & -- & --"
            return f"{f3(r[0])} & " + " & ".join(f3(r[1].get(g)) for g in GROUPS)
        rows.append(f"{lab} & {cell(e2)} & {cell(e3)}")

    def from_by(d, m):
        r = d["by_model"].get(m)
        return None if r is None else (r["BA_k4"], r["BA_by_group"])

    line("CNN (ResNet-18)", from_by(v4, "M2"), from_by(f5, "M2"))
    line("VLM", from_by(v4, "M3c"), from_by(f5, "M3c"))
    line("CNN + VLM", from_by(v4, "M4"), from_by(f5, "M4"))
    rows.append(r"\midrule")
    line("Tabular (LightGBM)", (p13["e2"]["M1x"], {g: p13["e2"]["by_group"][g]["M1x"] for g in GROUPS}),
         (p13["e3"]["M1x"], {g: p13["e3"]["by_group"][g]["M1x"] for g in GROUPS}))
    line("Coordinates only (1-NN)", (v4["by_model"]["location 1-NN"]["BA_k4"], v4["by_model"]["location 1-NN"]["BA_by_group"]), None)
    for lab, key in (("CNN (ResNet-50)", "resnet50"), ("CNN (ConvNeXt-Tiny)", "convnext_tiny")):
        rows.append(f"{lab} & {f3(bb[key]['e2'])} & -- & -- & -- & {f3(bb[key]['e3'])} & -- & -- & --")
    rows.append(r"\midrule")
    line("VLM + CNN features as tokens", from_by(v4, "M5e"), None)
    line("VLM + CNN probabilities as text", from_by(v4, "M5t"), None)
    line("\\quad + labels of 4 retrieved neighbours", from_by(v4, "M5tr"), None)
    line("AlphaEarth probe", from_by(d5, "A-probe"), from_by(f5, "A-probe"))
    line("VLM + AlphaEarth probe (mean)", from_by(d5, "M4A"), from_by(f5, "M4A"))
    line("CNN + VLM + AlphaEarth probe (mean)", from_by(d5, "M2+M3c+A"), from_by(f5, "M2+M3c+A"))
    for key, lab in (("M6_p0_ep1", "no dropout, epoch 1"), ("M6_p0_ep2", "no dropout, epoch 2"),
                     ("M6_p05_ep1", "dropout 0.5, epoch 1"), ("M6_p05_ep2", "dropout 0.5, epoch 2$^{a}$")):
        line(f"VLM + AlphaEarth tokens, {lab}", from_by(d5, key), from_by(f5, "M6") if key == "M6_p05_ep2" else None)
    n2 = ", ".join(f"{g} {v4['by_group_n'][g]}" for g in GROUPS)
    write("s3_continents", f"Other continents: BA of conversion to agriculture at $k = 4$ on E2 (1500 cells; {n2}) and "
          "E3 (2306 cells), overall and by region (Afr.: Africa; A-P: Asia-Pacific; LA: Latin America). "
          "Fusion variants were evaluated on E2 only, except the configuration chosen on E2 and the AlphaEarth probes.",
          "tab:s_cont", "lcccccccc",
          r"& \multicolumn{4}{c}{\textbf{E2}} & \multicolumn{4}{c}{\textbf{E3}} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}"
          r"\textbf{Model} & \textbf{All} & \textbf{Afr.} & \textbf{A-P} & \textbf{LA} & \textbf{All} & \textbf{Afr.} & \textbf{A-P} & \textbf{LA}",
          rows, note="$^{a}$ Configuration chosen on E2 and then scored on E3.", size=r"\footnotesize")

    rows = []
    for lab, m in (("CNN (ResNet-18)", "M2"), ("VLM", "M3c"), ("CNN + VLM", "M4"), ("VLM + CNN features as tokens", "M5e")):
        rows.append(f"{lab} & " + " & ".join(f3(v) for v in v4["by_model"][m]["BA_by_k"]))
    write("s3_e2_by_k", "E2: BA of conversion to agriculture by step $k$.", "tab:s_e2k", "lccccc",
          r"\textbf{Model} & $k=0$ & $k=1$ & $k=2$ & $k=3$ & $k=4$", rows)

    rows = []
    sub = [("validation file only", "Valid.\\ file"), ("High confidence only", "High conf."),
           (">=100 km from ForestNet train", r"$\geq$100 km")]
    for lab, m in (("CNN (ResNet-18)", "M2"), ("VLM", "M3c"), ("CNN + VLM", "M4")):
        rows.append(f"{lab} & " + " & ".join(f3(v4["secondary"][s][m]) for s, _ in sub) + " & "
                    + " & ".join(f3(f5["secondary"][s][m]) for s, _ in sub))
    n = " & ".join(str(v4["secondary"][s]["n"]) for s, _ in sub) + " & " + " & ".join(str(f5["secondary"][s]["n"]) for s, _ in sub)
    rows.insert(0, f"Cells & {n}")
    rows.insert(1, r"\midrule")
    write("s3_subsets", "BA at $k = 4$ on subsets of E2 and E3: cells from the validation file of Sims et al., cells "
          "labelled with high confidence, and cells at least 100 km from the nearest ForestNet training event.", "tab:s_sub",
          "lcccccc", r"& \multicolumn{3}{c}{\textbf{E2}} & \multicolumn{3}{c}{\textbf{E3}} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}"
          r"\textbf{Model} & " + " & ".join(rf"\textbf{{{t}}}" for _, t in sub) * 1 + " & " + " & ".join(rf"\textbf{{{t}}}" for _, t in sub),
          rows, size=r"\footnotesize")
    return seeds


def s3_uncertainty():
    b = j("rs_forestnet_robust/robust_a_parts13.json")["1_block_bootstrap"]
    bbb = j("rs_forestnet_robust/backbones_block_bootstrap.json")
    rows = []
    labels = [("M3c - M2", "VLM (seed 0) $-$ CNN"), ("M3c_s1 - M2", "VLM seed 1 $-$ CNN"), ("M3c_s2 - M2", "VLM seed 2 $-$ CNN"),
              ("M3c_mean3 - M2", "VLM mean of three seeds $-$ CNN"), ("M4 - M2", "CNN + VLM $-$ CNN"),
              ("M6 - M2", "VLM + AlphaEarth tokens $-$ CNN")]
    for s in ("e2", "e3"):
        for key, lab in labels:
            r = b[s].get(key)
            if r is None:
                continue
            rows.append(f"{s.upper()} & {lab} & {signed(r['diff'])} & {ci(r['ci_cell'])} & {ci(r['ci_country'])} & {ci(r['ci_1deg'])}")
        for arch, lab in (("resnet50", "ResNet-50"), ("convnext_tiny", "ConvNeXt-Tiny")):
            r = bbb[f"{s} VLM-{arch}"]
            rows.append(f"{s.upper()} & VLM $-$ CNN ({lab}) & {signed(r['diff'])} & {ci(r['cell'])} & {ci(r['country'])} & {ci(r['1deg'])}")
        if s == "e2":
            rows.append(r"\midrule")
    write("s3_bootstrap", f"Differences in BA at $k = 4$ with 95\\% intervals from 2000 bootstrap resamples of cells, "
          f"countries ({b['e2']['n_countries']} in E2, {b['e3']['n_countries']} in E3) and 1$^\\circ$ cells "
          f"({b['e2']['n_grid_cells']} and {b['e3']['n_grid_cells']}). CNN: ResNet-18 unless stated otherwise.", "tab:s_boot",
          "llcccc", r"\textbf{Set} & \textbf{Difference} & \textbf{Estimate} & \textbf{Cells} & \textbf{Countries} & \textbf{1$^\circ$ cells}",
          rows, size=r"\footnotesize")

    rows = []
    for s in ("e2", "e3"):
        for r in sorted(b[s]["per_country_n>=30"], key=lambda r: COUNTRY.get(r["country"], r["country"]).replace("\\^{o}", "o")):
            rows.append(f"{s.upper()} & {COUNTRY.get(r['country'], r['country'])} & {r['n']} & {f3(r['M2'])} & {f3(r['M3c'])} & {signed(r['M3c'] - r['M2'])}")
        if s == "e2":
            rows.append(r"\midrule")
    write("s3_countries", "BA of the CNN and the VLM at $k = 4$ in countries with at least 30 cells and both classes "
          f"present (VLM better in {b['e2']['countries_M3c_better']} countries of E2 and {b['e3']['countries_M3c_better']} of E3).",
          "tab:s_country", "llcccc", r"\textbf{Set} & \textbf{Country} & \textbf{Cells} & \textbf{CNN} & \textbf{VLM} & \textbf{VLM $-$ CNN}",
          rows, size=r"\footnotesize")

    a = j("rs_forestnet_robust/ablation_eval.json")
    rel = released_ablation()
    rows = []
    for s in ("e2", "e3"):
        for key, lab in (("full", "Full card"), ("img", "Fact sheet emptied (images only)"), ("txt", "Blank image grid (fact sheet only)")):
            v = rel[s]["ba"][key] if key == "full" else [a[s][key]] + [a[s]["by_group"][g][key] for g in GROUPS]
            rows.append(f"{s.upper()} & {lab} & " + " & ".join(f3(x) for x in v))
        if s == "e3":
            c = j("rs_forestnet_robust/country_ablation.json")
            rows.append(f"E3 & Card naming the cell's country & {f3(c['BA_card_names_country'])} & "
                        + " & ".join(f3(c["by_group"][g]["country"]) for g in GROUPS))
        rows.append(f"{s.upper()} & CNN (ResNet-18) & {f3(a[s]['CNN'])} & -- & -- & --")
        if s == "e2":
            rows.append(r"\midrule")
    d = a["e2"]
    r2, r3 = rel["e2"], rel["e3"]
    note = (f"Images only $-$ full card: E2 {signed(r2['diff'])} (cells {ci(r2['ci_cell'])}; countries {ci(r2['ci_country'])}), "
            f"E3 {signed(r3['diff'])} (cells {ci(r3['ci_cell'])}; countries {ci(r3['ci_country'])}), with the same binary call "
            f"for {r2['same'] * 100:.1f}\\% and {r3['same'] * 100:.1f}\\% of the cells; images only $-$ CNN: "
            f"E2 {signed(d['img-CNN'][0])} ({ci(d['img-CNN'][1])}), E3 {signed(a['e3']['img-CNN'][0])} ({ci(a['e3']['img-CNN'][1])}). "
            f"Country named $-$ Indonesia named (E3): {signed(c['diff'])} (cells {ci(c['ci_cell'])}; countries "
            f"{ci(c['ci_country'])}); binary call changed for {c['share_binary_call_changed'] * 100:.1f}\\% of the cells.")
    write("s3_ablation", "Input ablation of the fine-tuned VLM at $k = 4$ (BA), with the trained model unchanged.",
          "tab:s_abl", "llcccc", r"\textbf{Set} & \textbf{Input} & \textbf{All} & \textbf{Africa} & \textbf{Asia-Pacific} & \textbf{Latin America}",
          rows, note=note, size=r"\footnotesize")

    e = j("rs_forestnet_robust/event_dominance_sensitivity.json")
    rows = []
    for s in ("e2", "e3"):
        for key, lab in (("dominant year >=50% of 2001-2022 loss", r"Dominant year $\geq$50\% of the 2001--2022 loss"),
                         ("dominant year >=75%", r"Dominant year $\geq$75\%"),
                         ("high confidence and >=50%", r"High confidence and $\geq$50\%")):
            r = e[s][key]
            rows.append(f"{s.upper()} & {lab} & {r['n']} & {f3(r['CNN'])} & {f3(r['VLM'])} & {signed(r['diff'])} ({ci(r['ci'])})")
        if s == "e2":
            rows.append(r"\midrule")
    write("s3_dominance", "BA at $k = 4$ on cells in which the event year carried a large share of the cell's loss, so that "
          "the cell label most likely describes the evaluated event.", "tab:s_dom", "llcccc",
          r"\textbf{Set} & \textbf{Cells} & \textbf{n} & \textbf{CNN} & \textbf{VLM} & \textbf{VLM $-$ CNN (95\% CI)}", rows,
          size=r"\footnotesize")


def released_ablation(seed=20261030, B=2000):
    """Full-card BA from the released seed-0 VLM (overall and by region) and images-only minus full card, with 2000 paired
    resamples of cells and then of countries."""
    def pk4(s, m):
        d = pd.read_csv(RELEASE / "predictions" / f"{s}_{m}.csv")
        d = d[d.k == 4].set_index("eid")
        return (d.p_plantation + d.p_smallholder) >= 0.5

    def ba(y, yhat):
        return 0.5 * (yhat[y == 1].mean() + (~yhat[y == 0]).mean())

    rng = np.random.default_rng(seed)
    out = {}
    for s in ("e2", "e3"):
        c = pd.read_csv(RELEASE / "data" / f"{s}_cells.csv").set_index("eid")
        y = c.label_agri.to_numpy() == 1
        full, img = (pk4(s, m).reindex(c.index).to_numpy() for m in ("vlm", "vlm_image_only"))
        g = c.group.to_numpy()
        n = len(y)
        cell = [ba(y[i], img[i]) - ba(y[i], full[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
        cc = c.country.to_numpy()
        uc = np.unique(cc)
        idx = {u: np.where(cc == u)[0] for u in uc}
        ctry = []
        for _ in range(B):
            i = np.concatenate([idx[u] for u in rng.choice(uc, len(uc))])
            ctry.append(ba(y[i], img[i]) - ba(y[i], full[i]))
        out[s] = {"ba": {"full": [ba(y, full)] + [ba(y[g == r], full[g == r]) for r in GROUPS]},
                  "diff": ba(y, img) - ba(y, full), "ci_cell": list(np.percentile(cell, [2.5, 97.5])),
                  "ci_country": list(np.percentile(ctry, [2.5, 97.5])), "same": float((img == full).mean())}
    return out


def s3_labels():
    d = j("rs_forestnet_robust/label_sensitivity.json")
    b = j("rs_forestnet_robust/robust_a_parts13.json")["1_block_bootstrap"]
    rows = []
    for s in ("Indonesia", "E2", "E3"):
        for v, r in d[s].items():
            if s != "Indonesia" and v == "main mapping":
                r = dict(r, ci_cell=b[s.lower()]["M3c - M2"]["ci_cell"], ci_cluster=b[s.lower()]["M3c - M2"]["ci_country"])
            rows.append(f"{s} & {v[0].upper() + v[1:]} & {r['n']} & {f3(r['CNN'])} & {f3(r['VLM'])} & {signed(r['diff'])} & "
                        f"{ci(r['ci_cell'])} & {ci(r['ci_cluster'])}")
        if s != "E3":
            rows.append(r"\midrule")
    write("s3_labels", "Sensitivity of BA at $k = 4$ to the mapping of the external labels (Table~\\ref{tab:s_drivers}): logging "
          "cells excluded or counted as agriculture, shifting-cultivation cells excluded, both excluded, or only cells labelled "
          "with high confidence. 95\\% intervals of VLM $-$ CNN from 2000 bootstrap resamples of cells and of clusters "
          "(1$^\\circ$ cells in Indonesia, countries in E2 and E3). The intervals of the main mapping on E2 and E3 are those of "
          "Table~\\ref{tab:s_boot}.", "tab:s_labels", "llcccccc",
          r"\textbf{Set} & \textbf{Labels} & \textbf{n} & \textbf{CNN} & \textbf{VLM} & \textbf{Diff.} & \textbf{Cells} & \textbf{Clusters}",
          rows, size=r"\footnotesize")


def s3_fair():
    d = j("rs_forestnet_robust/fair_comparison.json")
    names = ["CNN (2.25 km)", "CNN (3 km)", "CNN (3 km) + fact-sheet numbers", "VLM", "VLM, images only"]
    shown = {"CNN (2.25 km)": "CNN, 2.25 km images", "CNN (3 km)": "CNN, 3 km images",
             "CNN (3 km) + fact-sheet numbers": "CNN, 3 km images and fact sheet", "VLM": "VLM, images and full card",
             "VLM, images only": "VLM, images only (3 seeds)"}
    rows = []
    for m in names:
        r = d["models"][m]
        rows.append(f"{shown[m]} & {f3(r['fn_macro_f1'])} & {f3(r['sidn_BA'])} & "
                    + " & ".join(f3(r[f'{t}_BA_by_group'][g]) for t in ("e2", "e3") for g in GROUPS))
    sd = j("rs_forestnet_robust/fair_vlmimg_seeds.json")
    srows = []
    for t, lab in (("fn", "ForestNet test"), ("sidn", "Indonesia"), ("e2", "E2"), ("e3", "E3")):
        im = sd["image_only"][t]
        fc = sd["full_card"].get(t)
        srows.append(f"{lab} & " + " / ".join(f3(v) for v in im["BA_seeds"]) + f" & {f3(im['BA_three_seed_model'])} & "
                    + ((" / ".join(f3(v) for v in fc["BA_seeds"]) + f" & {f3(fc['BA_three_seed_model'])}") if fc else "-- & --"))
    dd = sd["differences"]
    note = "; ".join(f"{t.upper()} {signed(dd[f'{t}: image-only (3 seeds) - full card (3 seeds)']['diff'])} "
                     f"(countries {ci(dd[f'{t}: image-only (3 seeds) - full card (3 seeds)']['ci_country'])})" for t in ("e2", "e3"))
    write("s3_seeds", "VLM with images only and VLM with the full card over three seeds: BA of conversion to agriculture at $k = 4$ "
          "for seeds 0, 1 and 2 and for the three-seed model (mean of the probabilities). The full-card seeds 1 and 2 were run on "
          "E2 and E3 only.", "tab:s_seeds", "lcccc",
          r"\textbf{Set} & \textbf{Images only, seeds 0/1/2} & \textbf{3 seeds} & \textbf{Full card, seeds 0/1/2} & \textbf{3 seeds}",
          srows, note=f"Images only $-$ full card, three-seed models: {note}.", size=r"\footnotesize")
    write("s3_fair", "Controls for the inputs: ForestNet test macro-F1, BA in Indonesia outside ForestNet, and BA on E2 and E3 by "
          "region (Afr.: Africa; A-P: Asia-Pacific; LA: Latin America), at $k = 4$.", "tab:s_fair", "lcccccccc",
          r"& & & \multicolumn{3}{c}{\textbf{E2}} & \multicolumn{3}{c}{\textbf{E3}} \\ \cmidrule(lr){4-6}\cmidrule(lr){7-9}"
          r"\textbf{Model and input} & \textbf{FN F1} & \textbf{Indon.} & \textbf{Afr.} & \textbf{A-P} & \textbf{LA} & "
          r"\textbf{Afr.} & \textbf{A-P} & \textbf{LA}", rows, size=r"\footnotesize")
    rows = []
    for key, r in d["differences"].items():
        t, comp = key.split(": ", 1)
        comp = comp.replace(" (3 km) + fact-sheet numbers", ", 3 km and fact sheet").replace(" (3 km)", ", 3 km").replace(
            " (2.25 km)", ", 2.25 km").replace(" - ", " $-$ ")
        sets = {"sidn": "Indonesia", "e2": "E2", "e3": "E3"}
        rows.append(f"{sets[t]} & {comp} & {signed(r['diff'])} & {ci(r['ci_cell'])} & "
                    f"{ci(r['ci_country']) if 'ci_country' in r else '--'}")
    write("s3_fair_diff", "Controls for the inputs: differences in BA at $k = 4$ with 95\\% intervals from 2000 paired bootstrap "
          "resamples of cells and of countries.", "tab:s_fair_diff", "llccc",
          r"\textbf{Set} & \textbf{Difference} & \textbf{Estimate} & \textbf{Cells} & \textbf{Countries}", rows, size=r"\footnotesize")


def s3_early():
    src = j("rs_forestnet_robust/early_stop_source.json")
    names = ["CNN", "VLM", "CNN + VLM"]
    rows = []
    for s, v in src.items():
        lab = s.replace(" (4 classes)", ", 4 classes")
        rows.append(f"{lab} ({v['CNN']['n']}) & " + " & ".join(f"{v[m]['lambda']:.2f} / {f3(v[m]['gap'])} / {v[m]['mean_stop']:.2f}"
                                                               for m in names))
    write("s3_early_source", "Early attribution with the threshold calibrated on the ForestNet calibration half (237 events) "
          "and applied to each whole set: threshold $\\hat\\lambda$ / gap (share of events that lose a correct attribution by "
          "stopping early) / mean stopping year. "
          "The binary sets use the binary rule calibrated on the same events.", "tab:s_es_src", "lccc",
          r"\textbf{Set (events)} & \textbf{CNN} & \textbf{VLM} & \textbf{CNN + VLM}", rows, size=r"\footnotesize")

    sp = j("rs_forestnet_robust/early_stop_split.json")
    rows = []
    for s in ("Indonesia", "E2"):
        for m in ("CNN", "VLM", "CNN + VLM"):
            for n, r in sp[s]["models"][m].items():
                rows.append(f"{s} & {m} & {n} & {r['share_test_gap_above_alpha']:.3f} & {f3(r['mean_test_gap'])} & "
                            f"{f3(r['p95_test_gap'])} & {r['share_attributed_before_k4']:.2f} & {r['share_attributed_at_k0']:.2f} & "
                            f"{r['mean_stop_year']:.2f} & {r['mean_test_cells']:.0f}")
        if s == "Indonesia":
            rows.append(r"\midrule")
    write("s3_early_local", "Local calibration on $n$ cells of some clusters (1$^\\circ$ cells in Indonesia, countries in E2) "
          "and evaluation on the other clusters (1000 repetitions; $\\alpha = 0.05$, $\\delta = 0.10$): share of repetitions "
          "with a held-out gap above $\\alpha$, mean and 95th percentile of the held-out gap (share of held-out cells that lose a "
          "correct attribution), share of cells attributed before $k = 4$ and at $k = 0$, mean stopping year and mean number "
          "of held-out cells. Every repetition certified a threshold. "
          "Indonesia has too few cells for $n = 400$.", "tab:s_es_loc", "lllccccccc",
          r"\textbf{Set} & \textbf{Model} & $n$ & $>\alpha$ & \textbf{Mean gap} & \textbf{P95} & $k<4$ & $k=0$ & \textbf{Stop} & \textbf{Held-out}",
          rows, size=r"\footnotesize")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s1_tables()
    card_example()
    s3_forestnet()
    s3_indonesia()
    s3_continents()
    s3_uncertainty()
    s3_labels()
    s3_fair()
    s3_early()


if __name__ == "__main__":
    main()
