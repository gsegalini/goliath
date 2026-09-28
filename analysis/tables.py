"""One function per published table, each returning a tidy DataFrame.

Tidy-first: the CSV is the source of truth and the LaTeX is a rendering of it, so any number in
the paper can be traced without re-running an experiment.

Adding an appendix table = one function + one TABLES entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from analysis import loaders as L

DATASETS_9 = ["abalone", "adult", "bean", "credit", "default", "gesture", "letter", "magic", "shoppers"]
LEAKAGE_DATASETS = ["adult", "letter", "gesture", "magic", "abalone"]


@dataclass
class Spec:
    fn: Callable[[], pd.DataFrame]
    caption: str
    label: str
    where: str = "appendix"          # "main" or "appendix"
    percent: list[str] = field(default_factory=list)   # columns already in percent
    note: str = ""


# ── 1. FedAvg: accumulated K-step updates vs a single gradient ────────────────
def tab_fedavg() -> pd.DataFrame:
    g = L.load_experiment("fedavg")
    m = L.load_experiment("fedavg-masked")
    df = pd.concat([g, m], ignore_index=True)
    rows = []
    for (ds, dt, b), cell in df.groupby(["dataname", "diffusion_type", "batch_size"]):
        rec = {"dataset": ds, "diffusion": dt, "subset": b}
        for k in (1, 2, 4, 8):
            sub = cell[cell.fedavg_local_steps == k]["tableak_score"]
            # ddof=1 (sample std) to match the published table, which was produced with
            # pandas .std() defaults. The leakage/cardinality/sensitive tables use ddof=0
            # (numpy default) instead -- the two conventions differ by sqrt(n/(n-1)) = 1.22 at
            # n=3, which is visible in the third decimal. Do not "harmonise" them without
            # re-blessing the fixtures and updating the paper.
            rec[f"K={k}"] = f"{sub.mean():.3f} +- {sub.std(ddof=1):.3f}" if len(sub) else "-"
        rows.append(rec)
    out = pd.DataFrame(rows)
    order = {"gaussian": 0, "masked": 1}
    return out.sort_values(["diffusion", "dataset", "subset"], key=lambda c: c.map(order).fillna(c)).reset_index(drop=True)


# ── 2. Leakage: whole-row, >=90/80% cells, re-identification ─────────────────
def _leakage_runs(df: pd.DataFrame, batch: int, diffusion: str, metric: str, n_runs: int):
    per_run = []
    for r in range(n_runs):
        vals = []
        for ds in LEAKAGE_DATASETS:
            cell = df[(df.dataname == ds) & (df.batch_size == batch) & (df.diffusion_type == diffusion)]
            runs = L.split_seed_runs(cell, L.ROWS_PER_RUN_REBUTTAL_METRICS[batch], n_runs=n_runs)
            if len(runs) > r and metric in runs[r]:
                v = runs[r][metric].dropna()
                if len(v):
                    vals.append(v.mean())
        if vals:
            per_run.append(np.mean(vals))
    return per_run


def tab_leakage(n_runs: int = 3) -> pd.DataFrame:
    df = L.load_experiment("rebuttal-metrics")
    rows = []
    for dt in ("gaussian", "masked"):
        for b in (1, 8, 32):
            rec = {"batch": b, "diffusion": dt}
            for metric, name in [
                ("whole_row_acc", "whole-row exact"),
                ("near_row_90", ">=90% cells"),
                ("near_row_80", ">=80% cells"),
                ("reid_rate", "re-identification"),
            ]:
                pr = _leakage_runs(df, b, dt, metric, n_runs)
                rec[name] = f"{np.mean(pr)*100:.1f} +- {np.std(pr)*100:.1f}" if pr else "-"
            rows.append(rec)
    return pd.DataFrame(rows)


# ── 3. Census vs adult (many-category scalability) ───────────────────────────
def tab_census() -> pd.DataFrame:
    b18 = L.load_experiment("accuracy-epochX-census-b18")
    b16 = L.load_experiment("accuracy-epochX-census-b16")
    b32 = L.load_experiment("accuracy-epochX-census")
    adult = L.load_experiment("accuracy-epochX")
    adult = adult[adult.dataname == "adult"]
    rows = []
    for b in (1, 8, 16, 32):
        src = b32 if b == 32 else b16 if b == 16 else b18
        rec = {"batch": b}
        for dt, lab in (("gaussian", "gaussian"), ("masked", "mixed")):
            c = src[(src.batch_size == b) & (src.diffusion_type == dt) & (src.epoch == 4)]
            a = adult[(adult.batch_size == b) & (adult.diffusion_type == dt) & (adult.epoch == 4)]
            rec[f"census {lab}"] = f"{c.tableak_score.mean():.3f}" if len(c) else "-"
            rec[f"adult {lab}"] = f"{a.tableak_score.mean():.3f}" if len(a) else "-"
        rows.append(rec)
    return pd.DataFrame(rows)[["batch", "census gaussian", "adult gaussian", "census mixed", "adult mixed"]]


# ── 4/5. Adaptive TabLeak ────────────────────────────────────────────────────
def _tableak_cells() -> pd.DataFrame:
    adaptive = L.load_experiment("tableak-adaptive")
    goliath = L.load_experiment("accuracy-epochX")
    rows = []
    for ds in DATASETS_9:
        for b in (1, 8, 16, 32):
            for dt in ("gaussian", "masked"):
                a = adaptive[(adaptive.dataname == ds) & (adaptive.batch_size == b) & (adaptive.diffusion_type == dt)]
                gm = goliath[(goliath.dataname == ds) & (goliath.batch_size == b) & (goliath.diffusion_type == dt)]
                if not len(a) or not len(gm):
                    continue
                rows.append({
                    "dataset": ds, "batch": b, "diffusion": dt,
                    "adaptive": a.tableak_score.mean(),
                    "multi": gm[gm.epoch == gm.epoch.max()].tableak_score.mean(),
                    "single": gm[gm.epoch == 0].tableak_score.mean(),
                })
    d = pd.DataFrame(rows)
    d["gap_multi"] = d["multi"] - d["adaptive"]
    d["gap_single"] = d["single"] - d["adaptive"]
    return d


def tab_tableak_summary() -> pd.DataFrame:
    d = _tableak_cells()
    rows = []
    for dt in ("gaussian", "masked", "all"):
        s = d if dt == "all" else d[d.diffusion == dt]
        rows.append({
            "diffusion": dt, "cells": len(s),
            "GOLIATH-adaptive (multi)": f"+{s.gap_multi.mean()*100:.1f}pp",
            "wins (multi)": f"{int((s.gap_multi > 0).sum())}/{len(s)}",
            "GOLIATH-adaptive (single)": f"+{s.gap_single.mean()*100:.1f}pp",
            "wins (single)": f"{int((s.gap_single > 0).sum())}/{len(s)}",
        })
    return pd.DataFrame(rows)


def tab_tableak_perdataset() -> pd.DataFrame:
    d = _tableak_cells()
    g = d.groupby("dataset").agg(GOLIATH=("multi", "mean"), adaptive=("adaptive", "mean"),
                                 gap=("gap_multi", "mean"), worst=("gap_multi", "min"))
    g = g.sort_values("gap", ascending=False).reset_index()
    g["GOLIATH"] = g.GOLIATH.map("{:.3f}".format)
    g["adaptive TabLeak"] = g.adaptive.map("{:.3f}".format)
    g["gap"] = g.gap.map(lambda v: f"+{v*100:.1f}pp")
    g["worst cell"] = g.worst.map(lambda v: f"{v*100:+.1f}pp")
    return g[["dataset", "GOLIATH", "adaptive TabLeak", "gap", "worst cell"]]


# ── 6/6b. Per-cardinality accuracy ───────────────────────────────────────────
CARD_COLS = [("tableak_cat_low_score", "categorical low (<=4)"),
             ("tableak_cat_medium_score", "categorical medium (5-9)"),
             ("tableak_cat_high_score", "categorical high (>=10)"),
             ("tableak_cont_score", "continuous")]


def tab_cardinality(n_runs: int = 3) -> pd.DataFrame:
    df = L.load_experiment("rebuttal-metrics")
    rows = []
    for ds in ["adult", "magic", "gesture", "letter", "abalone"]:
        cell = df[(df.dataname == ds) & (df.batch_size == 1) & (df.diffusion_type == "gaussian")]
        runs = L.split_seed_runs(cell, L.ROWS_PER_RUN_REBUTTAL_METRICS[1], n_runs=n_runs)
        rec = {"dataset": ds}
        for col, name in CARD_COLS:
            vals = [r[col].dropna().mean() for r in runs if col in r and r[col].notna().any()]
            rec[name] = f"{np.mean(vals)*100:.1f} +- {np.std(vals)*100:.1f}" if vals else "-"
        rows.append(rec)
    # census: one run at subset 50 (epoch 0 of the b18 sweep) + two at subset 32
    cc = L.load_experiment("census-cardinality")
    b18 = L.load_experiment("accuracy-epochX-census-b18")
    first = b18[(b18.batch_size == 1) & (b18.diffusion_type == "gaussian") & (b18.epoch == 0)]
    census_runs = [first] + L.split_seed_runs(cc, 32, n_runs=2)
    rec = {"dataset": "census (10 cols with >=10 cats)"}
    for col, name in CARD_COLS:
        vals = [r[col].dropna().mean() for r in census_runs if col in r and r[col].notna().any()]
        rec[name] = f"{np.mean(vals)*100:.1f} +- {np.std(vals)*100:.1f}" if vals else "-"
    rows.insert(1, rec)
    return pd.DataFrame(rows)


def tab_cardinality_batch(n_runs: int = 3) -> pd.DataFrame:
    df = L.load_experiment("rebuttal-metrics")
    rows = []
    for b in (1, 8, 32):
        rec = {"batch": b}
        for col, name in CARD_COLS[:3]:
            pr = _leakage_runs(df, b, "gaussian", col, n_runs)
            rec[name] = f"{np.mean(pr)*100:.1f}" if pr else "-"
        rows.append(rec)
    return pd.DataFrame(rows)


# ── 7. Reshuffling: cross-epoch gain ────────────────────────────────────────
RESHUFFLE_VARIANTS = [
    ("fixed batch order", lambda d: ~d.reshuffle),
    ("reshuffled each epoch (naive)", lambda d: d.reshuffle & ~d.global_ensemble),
    ("reshuffled + order-invariant (global pool)", lambda d: d.reshuffle & d.global_ensemble & d.fast_align),
    ("reshuffled + order-invariant + Hungarian", lambda d: d.reshuffle & d.global_ensemble & ~d.fast_align),
]


def tab_reshuffle() -> pd.DataFrame:
    df = L.load_experiment("reshuffle")
    for c in ("reshuffle", "global_ensemble", "fast_align"):
        df[c] = df[c].astype(bool)
    rows = []
    for name, pred in RESHUFFLE_VARIANTS:
        rec = {"setting": name}
        for ds in ("adult", "abalone"):
            for dt, lab in (("gaussian", "gaussian"), ("masked", "masked")):
                s = df[pred(df) & (df.dataname == ds) & (df.diffusion_type == dt)]
                if not len(s):
                    rec[f"{ds} {lab}"] = "-"
                    continue
                gain = s[s.epoch == 4].tableak_score.mean() - s[s.epoch == 0].tableak_score.mean()
                rec[f"{ds} {lab}"] = f"{gain*100:+.1f}pp"
        rows.append(rec)
    return pd.DataFrame(rows)[["setting", "adult gaussian", "adult masked", "abalone gaussian", "abalone masked"]]


# ── 8. DP trade-off: privacy and client utility ─────────────────────────────
def tab_dp() -> pd.DataFrame:
    rec_df = L.load_experiment("dp-defense-b1-c")
    eps_df = L.load_experiment("dp-defense-eps-c")
    util = L.load_experiment("dp-utility-c")
    util_b1 = util[util.batch_size == 1]

    def recon(df, ds, z):
        s = df[(df.dataname == ds) & (np.isclose(df.dp_noise_multiplier, z))]
        return s.tableak_score.mean() if len(s) else np.nan

    def loss_at(z_min=None):
        """Final-epoch train loss at batch 1, over the z values in the certified band."""
        sel = util_b1[util_b1.dp_noise_multiplier > 0] if z_min else util_b1[util_b1.dp_noise_multiplier == 0]
        last = sel.sort_values("epoch").groupby(["dataname", "dp_noise_multiplier"]).train_loss.last()
        return f"{last.min():.2f}-{last.max():.2f}"

    rows = [{
        "epsilon": "inf (undefended)",
        "adult": f"{recon(rec_df,'adult',0.0):.3f}",
        "credit": f"{recon(rec_df,'credit',0.0):.3f}",
        "client train loss (batch 1)": loss_at(),
    }]
    for z in (0.01, 0.03):
        rows.append({
            "epsilon": f"~{L.epsilon_from_z(z):.0f} (z={z})",
            "adult": f"{recon(rec_df,'adult',z):.3f}",
            "credit": f"{recon(rec_df,'credit',z):.3f}",
            "client train loss (batch 1)": loss_at(True),
        })
    cert = sorted(eps_df.dp_noise_multiplier.unique())
    a = [recon(eps_df, "adult", z) for z in cert]
    c = [recon(eps_df, "credit", z) for z in cert]
    rows.append({
        "epsilon": ", ".join(f"{L.epsilon_from_z(z):.3g}" for z in sorted(cert, key=lambda z: -z))[::1],
        "adult": f"{np.nanmin(a):.3f}-{np.nanmax(a):.3f}",
        "credit": f"{np.nanmin(c):.3f}-{np.nanmax(c):.3f}",
        "client train loss (batch 1)": loss_at(True),
    })
    return pd.DataFrame(rows)


# ── 9. Accuracy versus observed epochs ──────────────────────────────────────
def tab_epochs() -> pd.DataFrame:
    df = L.load_experiment("interepoch")
    df = df[df.dataname == "adult"]
    rows = []
    for k in (1, 2, 4, 6, 10):
        e = L.epoch_slice(df, k)
        rows.append({
            "epochs observed": k,
            "gaussian": f"{e[e.diffusion_type=='gaussian'].tableak_score.mean():.3f}",
            "mixed": f"{e[e.diffusion_type=='masked'].tableak_score.mean():.3f}",
            "both (pooled)": f"{e.tableak_score.mean():.3f}",
        })
    return pd.DataFrame(rows)


# ── 10. Per-column sensitive leakage ────────────────────────────────────────
SENSITIVE = ["pcol_race", "pcol_sex", "pcol_marital-status", "pcol_native-country",
             "pcol_relationship", "pcol_age"]


def tab_sensitive(n_runs: int = 3) -> pd.DataFrame:
    df = L.load_experiment("rebuttal-metrics")
    cell = df[(df.dataname == "adult") & (df.batch_size == 32) & (df.diffusion_type == "gaussian")]
    runs = L.split_seed_runs(cell, L.ROWS_PER_RUN_REBUTTAL_METRICS[32], n_runs=n_runs)
    rows = []
    for col in SENSITIVE + ["tableak_score"]:
        vals = [r[col].dropna().mean() for r in runs if col in r and r[col].notna().any()]
        if not vals:
            continue
        name = "overall (all columns)" if col == "tableak_score" else col.replace("pcol_", "")
        if name == "age":
            name = "age (continuous)"
        rows.append({"sensitive column (adult)": name,
                     "reconstruction accuracy": f"{np.mean(vals)*100:.1f} +- {np.std(vals)*100:.1f}"})
    return pd.DataFrame(rows)


# ── 11. Greedy versus Hungarian alignment ───────────────────────────────────
def tab_alignment() -> pd.DataFrame:
    df = L.load_experiment("align-diagnostic")
    rows = []
    for b in sorted(df.batch_size.unique()):
        s = df[df.batch_size == b]
        hun, gre = s.tableak_score.mean(), s.tableak_score_greedy.mean()
        rows.append({"batch size N": b, "Hungarian": f"{hun:.3f}", "greedy": f"{gre:.3f}",
                     "gap": f"{(hun-gre)*100:+.1f}pp",
                     "rows reassigned": f"{s.align_disagreement.mean()*100:.1f}%"})
    return pd.DataFrame(rows)


# ── 12/13. FT-Transformer versus MLP ────────────────────────────────────────
def _ft_cells() -> pd.DataFrame:
    # The published FT numbers come from the loss_fn="auto" configs only -- that is all the
    # committed sweep contains. The registered ft-transformer-v2 experiment ALSO defines
    # loss_fn="both" configs, and because this function aggregates over whatever rows exist at
    # batch 32, running the full experiment would silently fold a DIFFERENT loss configuration
    # into the published mean rather than adding a seed. Filtering here makes that impossible.
    ft = L.load_experiment("ft-transformer-v2")
    ft = ft[ft.loss_fn == "auto"]
    mlp = L.load_experiment("accuracy-epochX")
    rows = []
    for ds in DATASETS_9:
        for dt in ("gaussian", "masked"):
            f = ft[(ft.dataname == ds) & (ft.batch_size == 32) & (ft.diffusion_type == dt)]
            m = mlp[(mlp.dataname == ds) & (mlp.batch_size == 32) & (mlp.diffusion_type == dt)]
            if not len(f) or not len(m):
                continue
            f4, m4 = f[f.epoch == f.epoch.max()], m[m.epoch == m.epoch.max()]
            rows.append({"dataset": ds, "diffusion": dt,
                         "ft": f4.tableak_score.mean(), "mlp": m4.tableak_score.mean(),
                         "ft_cat": f4.tableak_cat_score.mean(), "mlp_cat": m4.tableak_cat_score.mean(),
                         "ft_cont": f4.tableak_cont_score.mean(), "mlp_cont": m4.tableak_cont_score.mean()})
    return pd.DataFrame(rows)


def tab_ft_summary() -> pd.DataFrame:
    d = _ft_cells()
    rows = []
    for dt, lab in (("gaussian", "gaussian"), ("masked", "mixed")):
        s = d[d.diffusion == dt]
        rows.append({"diffusion": lab,
                     "FT-Transformer": f"{s.ft.mean():.3f}", "MLP": f"{s.mlp.mean():.3f}",
                     "gap": f"{(s.mlp.mean()-s.ft.mean())*100:.1f}pp",
                     "FT categorical": f"{s.ft_cat.mean():.3f}", "MLP categorical": f"{s.mlp_cat.mean():.3f}",
                     "FT continuous": f"{s.ft_cont.mean():.3f}", "MLP continuous": f"{s.mlp_cont.mean():.3f}"})
    return pd.DataFrame(rows)


def tab_ft_perdataset() -> pd.DataFrame:
    d = _ft_cells()
    g = d[d.diffusion == "gaussian"].set_index("dataset")
    m = d[d.diffusion == "masked"].set_index("dataset")
    rows = []
    for ds in g.index:
        rows.append({"dataset": ds,
                     "FT gaussian": f"{g.loc[ds,'ft']:.3f}", "MLP gaussian": f"{g.loc[ds,'mlp']:.3f}",
                     "gap (gaussian)": f"{(g.loc[ds,'mlp']-g.loc[ds,'ft'])*100:.1f}pp",
                     "FT mixed": f"{m.loc[ds,'ft']:.3f}" if ds in m.index else "-",
                     "MLP mixed": f"{m.loc[ds,'mlp']:.3f}" if ds in m.index else "-",
                     "_sort": g.loc[ds, "mlp"] - g.loc[ds, "ft"]})
    return pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


# ── registry ────────────────────────────────────────────────────────────────
TABLES: dict[str, Spec] = {
    "fedavg": Spec(tab_fedavg, "Accumulated FedAvg updates ($K$ local steps) versus single-gradient inversion. Mean $\\pm$ std over 3 seeds.", "tab:fedavg", "main"),
    "leakage": Spec(tab_leakage, "Whole-row exact reconstruction, near-row thresholds and nearest-neighbour re-identification, averaged over five datasets. Percentages, mean $\\pm$ std over 3 seeds.", "tab:leakage", "main"),
    "census": Spec(tab_census, "Census-Income (KDD) versus adult after 5 epochs; Census carries 34 categorical columns against adult's 9.", "tab:census", "main"),
    "tableak_summary": Spec(tab_tableak_summary, "GOLIATH versus the adaptive TabLeak baseline over 9 datasets $\\times$ 4 batch sizes $\\times$ 2 diffusion types.", "tab:tableak-summary", "main"),
    "tableak_perdataset": Spec(tab_tableak_perdataset, "GOLIATH versus adaptive TabLeak per dataset, averaged over batch sizes and diffusion types. ``Worst cell'' is the smallest gap over that dataset's eight cells.", "tab:tableak-perdataset"),
    "cardinality": Spec(tab_cardinality, "Reconstruction accuracy by feature type and categorical cardinality (batch 1, gaussian). Percentages, mean $\\pm$ std over 3 seeds.", "tab:cardinality", "main"),
    "cardinality_batch": Spec(tab_cardinality_batch, "Cardinality-bin accuracy as the batch grows, pooled over the same five datasets.", "tab:cardinality-batch"),
    "reshuffle": Spec(tab_reshuffle, "Cross-epoch gain: accuracy after 5 epochs minus accuracy at epoch 0. Larger is better; fixed batch order is the no-reshuffling reference.", "tab:reshuffle", "main"),
    "dp": Spec(tab_dp, "Trade-off between reconstruction accuracy and client training utility under certified DP-SGD (batch 1). A train loss near 1.0 is an untrained model emitting the zero vector.", "tab:dp", "main"),
    "epochs": Spec(tab_epochs, "Reconstruction accuracy against the number of observed epochs (adult).", "tab:epochs", "main"),
    "sensitive": Spec(tab_sensitive, "Per-column leakage for the sensitive columns of adult (gaussian, batch 32).", "tab:sensitive", "main"),
    "alignment": Spec(tab_alignment, "Greedy versus optimal Hungarian row alignment scored on the same reconstruction.", "tab:alignment"),
    "ft_summary": Spec(tab_ft_summary, "FT-Transformer versus MLP backbone, averaged over 9 datasets (batch 32, cross-epoch ensembling).", "tab:ft-summary", "main"),
    "ft_perdataset": Spec(tab_ft_perdataset, "FT-Transformer versus MLP per dataset, ordered by the gaussian gap.", "tab:ft-perdataset"),
}
