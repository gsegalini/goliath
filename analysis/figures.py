"""Camera-ready figures: one per experiment that appears in the paper.

Replaces results/plot_rebuttal.py, which read only 5 experiment dirs and pointed at the superseded
`dp-defense` (batch 256, C=1.0) rather than the published `dp-defense-b1-c`.

Styling matches results/plot.py (seaborn colorblind, darkgrid, font_scale 1.3, pdf.fonttype 42 so
fonts embed as Type 42 rather than Type 3). Every figure is written as BOTH .pdf (for
\\includegraphics) and .svg (for quick viewing).

Figures whose source data is missing are skipped with a message, so this can be run while
experiments are still finishing.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from analysis import loaders as L  # noqa: E402
from analysis import tables as T  # noqa: E402

OUT_DIR = os.path.join(L.RESULTS_DIR, "camera_ready", "figures")
FILE_TYPES = ("pdf", "svg")

# same look as results/plot.py
sns.set_style("darkgrid")
sns.set_theme(font_scale=1.3, style="darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
PALETTE = "colorblind"


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in FILE_TYPES:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.{{{','.join(FILE_TYPES)}}}")


def _skip(name: str, why: str) -> None:
    print(f"  skipped {name}: {why}")


# ── FedAvg: does accumulating K local steps cost the attacker anything? ───────
def fig_fedavg() -> None:
    try:
        df = L.load_experiment("fedavg")
        dm = L.load_experiment("fedavg-masked")
    except FileNotFoundError as e:
        return _skip("fedavg_k", str(e))
    import pandas as pd

    d = pd.concat([df, dm], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for (ds, dt), cell in d.groupby(["dataname", "diffusion_type"]):
        g = cell.groupby("fedavg_local_steps").tableak_score.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3,
                    label=f"{ds}, {'mixed' if dt=='masked' else dt}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("local SGD steps $K$ (K=1 is the raw gradient)")
    ax.set_ylabel("reconstruction accuracy")
    ax.legend(fontsize=9)
    _save(fig, "fedavg_k")


# ── Reshuffling: cross-epoch gain per variant ────────────────────────────────
def fig_reshuffle() -> None:
    try:
        df = L.load_experiment("reshuffle")
    except FileNotFoundError as e:
        return _skip("reshuffle", str(e))
    for c in ("reshuffle", "global_ensemble", "fast_align"):
        df[c] = df[c].astype(bool)
    labels, series = [], []
    for name, pred in T.RESHUFFLE_VARIANTS:
        s = df[pred(df)]
        vals = []
        for ds in ("adult", "abalone"):
            for dt in ("gaussian", "masked"):
                c = s[(s.dataname == ds) & (s.diffusion_type == dt)]
                if len(c):
                    vals.append(c[c.epoch == 4].tableak_score.mean() - c[c.epoch == 0].tableak_score.mean())
        labels.append(name.replace("reshuffled ", "resh. ").replace(" (global pool)", ""))
        series.append(np.array(vals) * 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x, [v.mean() for v in series], yerr=[v.std() for v in series],
           capsize=4, color=sns.color_palette(PALETTE)[0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("cross-epoch gain (pp)")
    ax.axhline(0, color="k", lw=0.8)
    _save(fig, "reshuffle_gain")


# ── DP: the privacy/utility trade-off on one axis ───────────────────────────
def fig_dp() -> None:
    try:
        rec = L.load_experiment("dp-defense-b1-c")
        util = L.load_experiment("dp-utility-c")
    except FileNotFoundError as e:
        return _skip("dp_tradeoff", str(e))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = sns.color_palette(PALETTE)
    for i, ds in enumerate(("adult", "credit")):
        s = rec[rec.dataname == ds]
        g = s[s.dp_noise_multiplier > 0].groupby("dp_noise_multiplier").tableak_score.mean()
        eps = [L.epsilon_from_z(z) for z in g.index]
        ax.plot(eps, g.values, marker="o", color=colors[i], label=f"{ds}: reconstruction")
        ax.axhline(L.latent_ablation_reference(ds, 1), color=colors[i], ls=":", lw=1)
    ax2 = ax.twinx()
    ub = util[util.batch_size == 1]
    last = ub.sort_values("epoch").groupby(["dataname", "dp_noise_multiplier"]).train_loss.last().reset_index()
    for i, ds in enumerate(("adult", "credit")):
        s = last[(last.dataname == ds) & (last.dp_noise_multiplier > 0)]
        eps = [L.epsilon_from_z(z) for z in s.dp_noise_multiplier]
        ax2.plot(eps, s.train_loss, marker="s", ls="--", color=colors[i + 2],
                 label=f"{ds}: client train loss")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"privacy budget $\varepsilon$ (stronger privacy $\rightarrow$)")
    ax.set_ylabel("reconstruction accuracy")
    ax2.set_ylabel("client train loss (1.0 = untrained)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    _save(fig, "dp_tradeoff")


# ── Feasible-set contraction over observed epochs ───────────────────────────
def fig_contraction() -> None:
    try:
        df = L.load_experiment("interepoch")
    except FileNotFoundError as e:
        return _skip("contraction", str(e))
    d = df[df.dataname == "adult"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for dt, lab in (("gaussian", "gaussian"), ("masked", "mixed")):
        g = d[d.diffusion_type == dt].groupby("epoch").tableak_score.mean()
        ax.plot(g.index + 1, g.values, marker="o", label=lab)
    g = d.groupby("epoch").tableak_score.mean()
    ax.plot(g.index + 1, g.values, marker="", ls="--", color="k", lw=1, label="pooled")
    ax.set_xlabel("epochs observed")
    ax.set_ylabel("reconstruction accuracy (adult)")
    ax.legend()
    _save(fig, "contraction")


# ── Cardinality: accuracy by bin, and how it degrades with batch ────────────
def fig_cardinality() -> None:
    try:
        df = L.load_experiment("rebuttal-metrics")
    except FileNotFoundError as e:
        return _skip("cardinality", str(e))
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, name in T.CARD_COLS[:3]:
        ys = [np.mean(T._leakage_runs(df, b, "gaussian", col, 3)) * 100 for b in (1, 8, 32)]
        ax.plot([1, 8, 32], ys, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32])
    ax.set_xticklabels(["1", "8", "32"])
    ax.set_xlabel("batch size")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=9)
    _save(fig, "cardinality_vs_batch")


# ── Census vs adult: many-category generalisation ──────────────────────────
def fig_census() -> None:
    try:
        t = T.tab_census()
    except FileNotFoundError as e:
        return _skip("census", str(e))
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(t))
    w = 0.2
    for i, col in enumerate(["census gaussian", "adult gaussian", "census mixed", "adult mixed"]):
        vals = [float(v) if v != "-" else np.nan for v in t[col]]
        ax.bar(x + (i - 1.5) * w, vals, w, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"batch {b}" for b in t["batch"]])
    ax.set_ylabel("reconstruction accuracy")
    ax.legend(fontsize=8)
    _save(fig, "census_vs_adult")


# ── FT-Transformer vs MLP per dataset ──────────────────────────────────────
def fig_ft() -> None:
    try:
        d = T._ft_cells()
    except FileNotFoundError as e:
        return _skip("ft_transformer", str(e))
    g = d[d.diffusion == "gaussian"].copy()
    g["gap"] = g.mlp - g.ft
    g = g.sort_values("gap")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g.ft, 0.4, label="FT-Transformer")
    ax.bar(x + 0.2, g.mlp, 0.4, label="MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(g.dataset, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("reconstruction accuracy (gaussian, batch 32)")
    ax.legend()
    _save(fig, "ft_vs_mlp")


# ── Alignment: greedy vs Hungarian ─────────────────────────────────────────
def fig_alignment() -> None:
    try:
        df = L.load_experiment("align-diagnostic")
    except FileNotFoundError as e:
        return _skip("alignment", str(e))
    g = df.groupby("batch_size").agg(hun=("tableak_score", "mean"),
                                     gre=("tableak_score_greedy", "mean"),
                                     dis=("align_disagreement", "mean"))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g.index, g.hun, marker="o", label="Hungarian")
    ax.plot(g.index, g.gre, marker="s", label="greedy")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size $N$")
    ax.set_ylabel("reconstruction accuracy")
    ax2 = ax.twinx()
    ax2.plot(g.index, g.dis * 100, marker="^", ls=":", color="grey", label="rows reassigned")
    ax2.set_ylabel("rows assigned differently (%)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center right")
    _save(fig, "alignment_bias")


FIGURES = [fig_fedavg, fig_reshuffle, fig_dp, fig_contraction,
           fig_cardinality, fig_census, fig_ft, fig_alignment]


def build_all() -> None:
    print(f"figures -> {os.path.relpath(OUT_DIR, os.path.dirname(L.RESULTS_DIR))}")
    for fn in FIGURES:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - one bad figure must not stop the rest
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    build_all()
