import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ── Experiment selection ──────────────────────────────────────────────────────
# All experiments with a plotter. Previously this list was maintained by commenting lines in and
# out, which is why results/plot_rebuttal.py silently went stale -- whatever was commented out
# simply stopped being regenerated. Select at run time instead:
#   uv run plot.py                       # every experiment below
#   uv run plot.py --experiments accuracy defense
ALL_EXPERIMENTS = [
    "accuracy",
    "interepoch",
    "interepoch-8",
    "interepoch-8-2",
    "interepoch-simple",
    "ablation-tn",
    "ablation-loss",
    "ablation-prior",
    "known-t",
    "known-noise",
    "known-tn",
    "known-mask",
    "known-tm",
    "known-tnm",
    "known-tm-prior",
    "known-tm-prior-1",
    "pool-size",
    "pool-size-8",
    "pool-size-perfect",
    "defense",
    "init-strategy",
    "latent-reconstruction",
]
EXPERIMENTS = ALL_EXPERIMENTS  # overridden by --experiments
DATASETS = ["abalone", "adult", "bean", "default", "gesture", "letter", "magic", "shoppers", "credit"]  # "australia"
ACCURACY_DATASETS = DATASETS + ["census"]
METRICS_TO_PLOT = ["tableak_score", "dcr", "reconstruction_time"]
BATCH_SIZES = [1, 8, 16, 32]

# per-experiment metric override; experiments not listed use METRICS_TO_PLOT
EXPERIMENT_METRICS = {
    "latent-reconstruction": ["t_mae", "noise_mse", "mask_accuracy"],
    "known-mask": ["tableak_score", "mask_accuracy", "t_mae", "dcr"],
    "known-tm": ["tableak_score", "mask_accuracy", "t_mae", "dcr"],
    "known-tm-prior": ["tableak_score", "mask_accuracy", "mask_accuracy_not_aligned", "t_mae", "dcr"],
    "known-tm-prior-1": ["tableak_score", "mask_accuracy", "t_mae", "dcr"],
    "known-tnm": ["tableak_score", "mask_accuracy", "t_mae", "dcr"],
}

# ── Per-experiment settings ───────────────────────────────────────────────────
ACCURACY_POOL_SIZE = 1  # pool size used for accuracy experiment (set None to use all)
INTEREPOCH_BATCH_SIZE = 32  # batch size used for interepoch / defense / pool-size / init-strategy
MULTI_EPOCH_N = 5  # epoch index (0-based) used as "multi-epoch" result in tables
MULTI_EPOCH_ACC = "accuracy-epochX"
CENSUS_EPOCH_ACCURACY = {
    1: "accuracy-epochX-census-b18",
    8: "accuracy-epochX-census-b18",
    16: "accuracy-epochX-census-b16",
    32: "accuracy-epochX-census",
}

# ── Method names ──────────────────────────────────────────────────────────────
NAME = "GOLIATH (This work)"
BASELINE_METHODS = ["TabLeak", "Adaptive TabLeak", "GIDM"]
# Random baselines for masked diffusion: mask frozen to a random sample, or all latents frozen (equivalent to tableak in gaussian)
RANDOM_VARIANTS = [("mask", "Random Mask"), ("all", "TabLeak")]
RANDOM_METHODS = [label for _, label in RANDOM_VARIANTS]
OUR_METHODS = [NAME, f"{NAME} {MULTI_EPOCH_N}-epochs"]
METHODS_BY_DIFFUSION = {
    "gaussian": BASELINE_METHODS + OUR_METHODS,
    "masked": RANDOM_METHODS + ["Adaptive TabLeak"] + OUR_METHODS,
}
DIFFUSION_TYPES = ["gaussian", "masked"]
LOWER_IS_BETTER = {"dcr", "nndr"}

# ── Metric metadata ───────────────────────────────────────────────────────────
PERCENT_METRICS = [
    "tableak_score",
    "tableak_cat_score",
    "tableak_cont_score",
    "mask_accuracy",
    "mask_accuracy_not_aligned",
]
PRIVACY_METRICS = ["dcr"]
TIME_METRICS = ["reconstruction_time", "time_optimization_s", "time_pooling_s", "time_alignment_s"]
MEMORY_METRICS = ["estimated_vram_mb"]
LATENT_METRICS = ["t_mae", "noise_mse", "mask_accuracy"]

METRIC_NAME_MAP = {
    "tableak_score": "Reconstruction Accuracy (%)",
    "tableak_cat_score": "Categorical Accuracy (%)",
    "tableak_cont_score": "Continuous Accuracy (%)",
    "dcr": "Distance to Closest Record (DCR) ⬇",
    "nndr": "Nearest Neighbor Distance Ratio (NNDR) ⬇",
    "reconstruction_time": "Reconstruction Time (seconds)",
    "time_optimization_s": "Optimization Time (s)",
    "time_pooling_s": "Pooling Time (s)",
    "time_alignment_s": "Alignment Time (s)",
    "estimated_vram_mb": "Estimated VRAM (MB)",
    "t_mae": "Timestep MAE",
    "noise_mse": "Noise MSE",
    "mask_accuracy": "Mask Accuracy (%)",
    "mask_accuracy_not_aligned": "Mask Accuracy, Not Aligned (%)",
}

# ── Label maps ────────────────────────────────────────────────────────────────
DEFENSES_MAP = {
    "clipping": "Gradient Clipping",
    "compression": "Gradient Compression",
    "noise": "Additive Noise",
    "censor": "CENSOR",
    None: "No Defense",
}
LOSS_FN_MAP = {
    "mse": "MSE",
    "cossim": "1 - Cossim",
    "both": "Weighted",
    "double-cossim": "Double CosSim",
    "auto": "Dynamic",
}
INIT_STRATEGY_MAP = {"uniform": "Uniform", "previous": "Previous"}
ANCHOR_MAP = {"first": "First Epoch", "latest": "Latest Epoch"}

FAST_ALIGN = "Fast Align"
CUBIC_ALIGN = "Cubic Align"
NO_ALIGN = "No Align"
NORMAL_POOLING = "To Previous Reconstruction"
PERFECT_POOLING = "To Ground Truth"

# ── Output ────────────────────────────────────────────────────────────────────
BASE_DIR = "plots"
DATA_BASE_DIR = "."
# Camera-ready needs vector PDF for \includegraphics; SVG is kept for quick viewing.
# pdf.fonttype=42 below ensures fonts embed as TrueType rather than Type 3.
FILE_TYPES = ["pdf", "svg"]
FILE_TYPE = FILE_TYPES[-1]  # back-compat for any caller still reading the singular name

# ── Visualization defaults ────────────────────────────────────────────────────
COLOR_PALETTE_LINES = "colorblind"
COLOR_MAP_HEATMAP = sns.color_palette("viridis", as_cmap=True)

sns.set_style("darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
sns.set_theme(font_scale=1.3)
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["legend.title_fontsize"] = 11

LEGEND_FONT_SIZE = 11
LEGEND_FONT_SIZE_SMALL = 9
LEGEND_FONT_SIZE_TINY = 7
LEGEND_TITLE_FONT_SIZE_SMALL = 9


# ── Helpers ───────────────────────────────────────────────────────────────────


def ensure_diffusion_type(df):
    if "diffusion_type" not in df.columns:
        df["diffusion_type"] = "gaussian"
    df["diffusion_type"] = df["diffusion_type"].fillna("gaussian")


def load_csv(path, scale_percent=True):
    """Load a CSV and optionally scale percent metrics to 0-100."""
    df = pd.read_csv(path)
    if scale_percent:
        for col in PERCENT_METRICS:
            if col in df.columns:
                df[col] = df[col] * 100
    return df


def load_baseline(path, scale_percent=True):
    """Load a CSV for a baseline method; returns empty DataFrame on missing file."""
    try:
        df = load_csv(path, scale_percent=scale_percent)
        df = df[df["simple_attack"] == False]
        ensure_diffusion_type(df)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def load_census_epochs():
    """Join the nonoverlapping Census sweeps without mixing their batch counts."""
    return pd.concat([
        load_baseline(f"{DATA_BASE_DIR}/{experiment}/census/performance_census.csv")
        .query("batch_size == @batch_size")
        for batch_size, experiment in CENSUS_EPOCH_ACCURACY.items()
    ], ignore_index=True)


def _refilter_align_categories(df):
    """Restrict fast_align / perfect_pooling Categoricals to values present in df."""
    if "fast_align" in df.columns:
        present_fa = [c for c in [CUBIC_ALIGN, FAST_ALIGN, NO_ALIGN] if c in df["fast_align"].values]
        df["fast_align"] = pd.Categorical(df["fast_align"], categories=present_fa, ordered=True)
    if "perfect_pooling" in df.columns:
        present_pp = [c for c in [NORMAL_POOLING, PERFECT_POOLING] if c in df["perfect_pooling"].values]
        df["perfect_pooling"] = pd.Categorical(df["perfect_pooling"], categories=present_pp, ordered=True)


def apply_column_transforms(df):
    """Add human-readable fast_align, perfect_pooling, and anchor columns in-place."""
    try:
        df["fast_align"] = df.apply(
            lambda row: FAST_ALIGN if row["fast_align"] else (CUBIC_ALIGN if row["align"] else NO_ALIGN),
            axis=1,
        )
    except Exception:
        pass
    try:
        df["perfect_pooling"] = df["perfect_pooling"].map({True: PERFECT_POOLING, False: NORMAL_POOLING})
    except Exception:
        df["perfect_pooling"] = NORMAL_POOLING
    _refilter_align_categories(df)
    if "anchor" in df.columns:
        df["anchor"] = df["anchor"].map(lambda v: ANCHOR_MAP.get(v, str(v)))


def _apply_metric_ylim(metric):
    if metric not in PERCENT_METRICS:
        return
    ax = plt.gca()
    ymin, _ = ax.get_ylim()
    new_bottom = max(0, (int(ymin) // 10) * 10)
    new_top = min(100, (int(ax.get_ylim()[1]) // 10 + 1) * 10)
    plt.ylim(bottom=new_bottom, top=new_top)


def save_fig(path):
    """Write the current figure once per FILE_TYPES entry (pdf for LaTeX, svg for preview)."""
    root = path[: -(len(FILE_TYPE) + 1)] if path.endswith("." + FILE_TYPE) else path
    for ext in FILE_TYPES:
        plt.savefig(f"{root}.{ext}", bbox_inches="tight")


def fig_path(out_dir, prefix, data):
    return os.path.join(out_dir, f"{prefix}_{data}.{FILE_TYPE}")


def metric_prefix(metric):
    """Return a filename-friendly prefix for a metric."""
    return "tableak" if metric in PERCENT_METRICS else metric


def metric_source(metric):
    """Map derived/alias metric names to their CSV columns."""
    if metric == "mask_accuracy_not_aligned":
        return "mask_accuracy"
    return metric


def _legend_overlaps_lines(ax, legend):
    """Return True when the rendered legend box intersects plotted line paths."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer).expanded(1.03, 1.08)

    for line in ax.lines:
        if not line.get_visible():
            continue
        path = line.get_path()
        if len(path.vertices) < 2:
            continue
        display_path = path.transformed(line.get_transform())
        if display_path.intersects_bbox(legend_bbox, filled=False):
            return True
    return False


def _legend_with_outside_fallback(ax, handles, labels, inside_kwargs, outside_kwargs):
    """Keep the legend inside unless it overlaps lines, then move it outside right."""
    legend = ax.legend(handles=handles, labels=labels, **inside_kwargs)
    if _legend_overlaps_lines(ax, legend):
        legend.remove()
        legend = ax.legend(handles=handles, labels=labels, **outside_kwargs)
    return legend


def _subscore_top_legend(ax, handles, labels, score_type_count):
    """Place subscore legends as two compact columns at the top of the plot."""
    if not handles or not labels:
        return

    if "Alignment Algorithm" in labels:
        align_idx = labels.index("Alignment Algorithm")
        score_handles = handles[1:align_idx]
        score_labels = labels[1:align_idx]
        align_handles = handles[align_idx + 1:]
        align_labels = labels[align_idx + 1:]
    else:
        score_handles = handles[:score_type_count]
        score_labels = labels[:score_type_count]
        align_handles = []
        align_labels = []

    score_legend = ax.legend(
        handles=score_handles,
        labels=score_labels,
        title="Score Type",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0,
        frameon=True,
        prop={"size": LEGEND_FONT_SIZE_SMALL},
        title_fontsize=LEGEND_TITLE_FONT_SIZE_SMALL,
    )
    ax.add_artist(score_legend)

    if align_handles:
        ax.legend(
            handles=align_handles,
            labels=align_labels,
            title="Alignment Algorithm",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            borderaxespad=0,
            frameon=True,
            prop={"size": LEGEND_FONT_SIZE_SMALL},
            title_fontsize=LEGEND_TITLE_FONT_SIZE_SMALL,
        )


# ── Per-experiment plotters ───────────────────────────────────────────────────


def _plot_tableak_subscores(df, x_col, style_col, out_dir, prefix, data):
    """Separate figure: cat/cont/total subscores, hue=score_type, style=fast_align."""
    sns.set_palette(COLOR_PALETTE_LINES)
    score_cols = ["tableak_score", "tableak_cat_score", "tableak_cont_score"]
    score_labels = ["Total", "Categorical", "Continuous"]
    id_vars = [c for c in [x_col, "fast_align"] if c in df.columns]
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=[c for c in score_cols if c in df.columns],
        var_name="score_type",
        value_name="score",
    )
    df_melted["score_type"] = pd.Categorical(
        df_melted["score_type"].map(dict(zip(score_cols, score_labels))),
        categories=score_labels,
        ordered=True,
    )
    style = "fast_align" if "fast_align" in df_melted.columns else None
    ax = sns.lineplot(data=df_melted, x=x_col, y="score", errorbar=("ci", 95), hue="score_type", style=style)
    plt.ylabel(METRIC_NAME_MAP["tableak_score"])
    plt.xticks(ticks=df_melted["pool_size"].unique(), labels=df_melted["pool_size"].unique().astype(str))
    h, l = ax.get_legend_handles_labels()
    if style is not None:
        l[0] = "Score Type"
        style_idx = df_melted["score_type"].nunique() + 1
        if style_idx < len(l):
            l[style_idx] = "Alignment Algorithm"
        _subscore_top_legend(ax, h, l, df_melted["score_type"].nunique())
    else:
        _subscore_top_legend(ax, h, l, df_melted["score_type"].nunique())
    _apply_metric_ylim("tableak_score")
    save_fig(fig_path(out_dir, prefix, data))


def plot_interepoch(df, data, metric, out_dir, experiment):
    sns.set_palette(COLOR_PALETTE_LINES)
    # get unique batch size, fallback in case
    batch_sizes = df["batch_size"].dropna().unique()
    batch_size = batch_sizes[0] if len(batch_sizes) > 0 else INTEREPOCH_BATCH_SIZE
    df = df[df["batch_size"] == batch_size].copy()
    df["epoch"] = df["epoch"] + 1

    for diff_type in sorted(df["diffusion_type"].unique()):
        df_diff = df[df["diffusion_type"] == diff_type].copy()
        if df_diff.empty:
            continue

        if "anchor" in df_diff.columns and df_diff["anchor"].nunique() > 1:
            style_col = "anchor"
            style_label = "Anchor"
        elif df_diff["perfect_pooling"].nunique() > 1:
            style_col = "perfect_pooling"
            style_label = "Alignment Basis"
        else:
            style_col = None
            style_label = None

        ax = sns.lineplot(
            data=df_diff, x="epoch", y=metric, errorbar=("ci", 95),
            hue="fast_align", style=style_col,
        )
        plt.xlabel("Epoch")
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        ax.xaxis.label.set_size(20)

        h, l = ax.get_legend_handles_labels()
        loc = "lower right" if experiment == "interepoch" else "upper right"
        if style_col is not None:
            # with hue+style seaborn prepends column-name headers; replace them
            l[0] = "Alignment Algorithm"
            style_idx = df_diff["fast_align"].nunique() + 1
            if style_idx < len(l):
                l[style_idx] = style_label
            plt.legend(title=None, labels=l, handles=h, loc=loc, prop={"size": LEGEND_FONT_SIZE})
        else:
            # hue-only: no header entries, use legend title instead
            plt.legend(title="Alignment Algorithm", handles=h, labels=l, loc=loc, prop={"size": LEGEND_FONT_SIZE})

        prefix = metric_prefix(metric)
        _apply_metric_ylim(metric)
        save_fig(fig_path(out_dir, f"{prefix}_epoch_{diff_type}", data))

        if metric == "tableak_score":
            _plot_tableak_subscores(df_diff, "epoch", style_col, out_dir, f"tableak_subscores_epoch_{diff_type}", data)

        if metric in TIME_METRICS and experiment == "interepoch" and data != "australia":
            latex_string = r"${:.2f}_{{{:.2f}}}$ & ${:.2f}_{{{:.2f}}}$ & ${:.2f}_{{{:.2f}}}$"
            values = []
            for align_method in df_diff["fast_align"].cat.categories:
                tmp = df_diff[(df_diff["fast_align"] == align_method) & (df_diff["epoch"] >= 2)]
                values += [tmp[metric].mean(), tmp[metric].std()]
            if len(values) == 6:
                print(f"[{diff_type}] " + latex_string.format(*values))


def plot_accuracy(df, data, metric, out_dir, table_metrics):
    sns.set_palette(COLOR_PALETTE_LINES)
    df = df[df["simple_attack"] == False].copy()
    if ACCURACY_POOL_SIZE is not None:
        df = df[df["pool_size"] == ACCURACY_POOL_SIZE]
    ensure_diffusion_type(df)

    tableak_dir = (
        f"{DATA_BASE_DIR}/tableak-baseline-{ACCURACY_POOL_SIZE}"
        if ACCURACY_POOL_SIZE
        else f"{DATA_BASE_DIR}/tableak-baseline"
    )
    df_gidm = load_baseline(f"{DATA_BASE_DIR}/gidm-baseline/{data}/performance_{data}.csv")
    df_tableak = load_baseline(f"{tableak_dir}/{data}/performance_{data}.csv")
    df_adaptive = load_baseline(f"{DATA_BASE_DIR}/tableak-adaptive/{data}/performance_{data}.csv")
    if data == "census":
        # The Census batch-1/8 and batch-32 sweeps used different attacked-batch counts.
        # Keep them separate until after filtering to the requested batch size.
        df_multi_all = load_census_epochs()
    else:
        df_multi_all = load_baseline(f"{DATA_BASE_DIR}/{MULTI_EPOCH_ACC}/{data}/performance_{data}.csv")
    df_multi = pd.DataFrame()
    if not df_multi_all.empty:
        df_multi = df_multi_all[df_multi_all["epoch"] == MULTI_EPOCH_N - 1]

    df_random = load_baseline(f"{DATA_BASE_DIR}/random-baseline/{data}/performance_{data}.csv")
    random_sources = {}
    if not df_random.empty and "random_latents" in df_random.columns:
        for variant_key, variant_label in RANDOM_VARIANTS:
            random_sources[variant_label] = df_random[df_random["random_latents"] == variant_key]

    sources_gaussian = {
        NAME: df,
        f"{NAME} {MULTI_EPOCH_N}-epochs": df_multi,
        "GIDM": df_gidm,
        "TabLeak": df_tableak,
        "Adaptive TabLeak": df_adaptive,
    }

    sources_masked = {
        NAME: df,
        f"{NAME} {MULTI_EPOCH_N}-epochs": df_multi,
        **random_sources,
        "Adaptive TabLeak": df_adaptive,
    }

    for diff_type in DIFFUSION_TYPES:
        sns.set_palette(COLOR_PALETTE_LINES)
        allowed_methods = METHODS_BY_DIFFUSION[diff_type]
        sources = sources_gaussian if diff_type == "gaussian" else sources_masked
        for method_name, method_df in sources.items():
            if method_name not in allowed_methods:
                continue
            if method_df.empty:
                continue
            df_diff = method_df[method_df["diffusion_type"] == diff_type]
            if df_diff.empty:
                # old style, gaussian only
                if diff_type != "gaussian":
                    continue
                else:
                    df_diff = method_df
            # filter out only wanted batch sizes
            df_diff = df_diff[df_diff["batch_size"].isin(BATCH_SIZES)]
            if metric not in df_diff.columns or df_diff[metric].notna().sum() == 0:
                continue
            sns.lineplot(data=df_diff, x="batch_size", y=metric, errorbar=("ci", 95), label=method_name)
            table_metrics[(metric, diff_type)][method_name][data] = (
                df_diff.groupby("batch_size")[metric].mean().to_dict(),
                df_diff.groupby("batch_size")[metric].std().to_dict(),
            )

        plt.ylabel("Reconstruction Accuracy" if metric in PERCENT_METRICS else f"Privacy Metric ({metric})")
        plt.xticks(ticks=BATCH_SIZES)
        plt.xlabel("Batch Size")
        _apply_metric_ylim(metric)
        save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_batchsize_{diff_type}", data))

    plot_accuracy_epoch(df_multi_all, data, metric, out_dir)


def plot_accuracy_epoch(df_multi, data, metric, out_dir):
    """Epoch-progression plot from accuracy-epochX: x=epoch, hue=batch_size."""
    sns.set_palette(COLOR_PALETTE_LINES)
    if df_multi.empty:
        plt.clf()
        return
    df = df_multi.copy()
    df["epoch"] = df["epoch"] + 1

    for diff_type in DIFFUSION_TYPES:
        sns.set_palette(COLOR_PALETTE_LINES)
        df_diff = df[df["diffusion_type"] == diff_type]
        if df_diff.empty:
            continue
        df_diff = df_diff[df_diff["batch_size"].isin(BATCH_SIZES)].copy()
        if df_diff.empty:
            continue
        df_diff["batch_size"] = df_diff["batch_size"].astype(str)
        sns.lineplot(data=df_diff, x="epoch", y=metric, hue="batch_size", style="batch_size", errorbar=("ci", 10))
        plt.xlabel("Epoch")
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        plt.xticks(ticks=sorted(df_diff["epoch"].unique()))
        _apply_metric_ylim(metric)
        save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_epoch_{diff_type}", data))


def plot_ablation_tn(df, data, metric, out_dir):
    df_avg = df.groupby(["optimize_t", "optimize_noise"])[metric].mean().reset_index()
    pivot = df_avg.pivot(index="optimize_t", columns="optimize_noise", values=metric)
    fmt = ".1f%" if metric in PERCENT_METRICS else ".2f"
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=COLOR_MAP_HEATMAP)
    plt.xticks(ticks=[0.5, 1.5], labels=["No", "Yes"])
    plt.yticks(ticks=[0.5, 1.5], labels=["No", "Yes"], rotation=0)
    save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_ablation_tn", data))


def plot_ablation_loss(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    df = df.copy()

    def make_variant_label(row):
        if row["loss_fn"] == "double-cossim":
            return f"Double CosSim\n(α={row['double_cossim_alpha']})"
        return LOSS_FN_MAP.get(row["loss_fn"], row["loss_fn"])

    df["variant_label"] = df.apply(make_variant_label, axis=1)

    base_order = [LOSS_FN_MAP[k] for k in ["mse", "cossim", "both", "auto"] if k in df["loss_fn"].values]
    alphas = sorted(df.loc[df["loss_fn"] == "double-cossim", "double_cossim_alpha"].dropna().unique())
    alpha_labels = [f"Double CosSim\n(α={a})" for a in alphas]
    ordered_categories = [c for c in base_order + alpha_labels if c in df["variant_label"].values]
    df["variant_label"] = pd.Categorical(df["variant_label"], categories=ordered_categories, ordered=True)

    score_cols = ["tableak_score", "tableak_cat_score", "tableak_cont_score"]
    score_labels = ["Total", "Categorical", "Continuous"]

    if metric != "tableak_score":
        plt.figure(figsize=(22, 8)) # <-- ADD IT HERE
        sns.barplot(data=df, x="variant_label", y=metric, hue="diffusion_type")
        plt.xlabel("Loss Function")
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_ablation_loss", data))
    else:
        for diff_type in DIFFUSION_TYPES:
            df_diff = df[df["diffusion_type"] == diff_type]
            if df_diff.empty:
                continue
            df_melted = df_diff.melt(
                id_vars=["variant_label"],
                value_vars=score_cols,
                var_name="score_type",
                value_name="score",
            )
            df_melted["score_type"] = pd.Categorical(
                df_melted["score_type"].map(dict(zip(score_cols, score_labels))),
                categories=score_labels,
                ordered=True,
            )
            plt.figure(figsize=(22, 8)) # <-- AND ADD IT HERE
            sns.barplot(data=df_melted, x="variant_label", y="score", hue="score_type")
            plt.xlabel("Loss Function")
            plt.ylabel(METRIC_NAME_MAP["tableak_score"])
            plt.legend(title=None)
            save_fig(os.path.join(out_dir, f"tableak_{diff_type}_ablation_loss_{data}.{FILE_TYPE}"))

def plot_ablation_prior(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    df = df.copy()
    df["prior_label"] = df["no_prior"].map({False: "With Prior", True: "No Prior"})
    df["prior_label"] = pd.Categorical(df["prior_label"], categories=["With Prior", "No Prior"], ordered=True)

    score_cols = ["tableak_score", "tableak_cat_score", "tableak_cont_score"]
    score_labels = ["Total", "Categorical", "Continuous"]

    if metric != "tableak_score":
        sns.barplot(data=df, x="prior_label", y=metric)
        plt.xlabel("Prior")
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_ablation_prior", data))
        return

    df_melted = df.melt(
        id_vars=["prior_label"],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score",
    )
    df_melted["score_type"] = pd.Categorical(
        df_melted["score_type"].map(dict(zip(score_cols, score_labels))),
        categories=score_labels,
        ordered=True,
    )
    sns.barplot(data=df_melted, x="prior_label", y="score", hue="score_type")
    plt.xlabel("Prior")
    plt.ylabel(METRIC_NAME_MAP["tableak_score"])
    plt.legend(title=None)
    save_fig(fig_path(out_dir, f"tableak_ablation_prior", data))


def _plot_known_subscores(df, x_col):
    """Add cat/cont sub-lines when plotting tableak_score."""
    sns.lineplot(
        data=df, x=x_col, y="tableak_cat_score", errorbar=("ci", 95), linestyle="--", label="Categorical Score"
    )
    sns.lineplot(
        data=df, x=x_col, y="tableak_cont_score", errorbar=("ci", 95), linestyle=":", label="Continuous Score"
    )
    _apply_metric_ylim("tableak_score")


def plot_known_noise(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    sns.lineplot(data=df, x="known_noise_amount", y=metric, errorbar=("ci", 95))
    if metric == "tableak_score":
        _plot_known_subscores(df, "known_noise_amount")
    else:
        _apply_metric_ylim(metric)
    save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_known_noise", data))


def plot_known_t(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    sns.lineplot(data=df, x="known_t_amount", y=metric, errorbar=("ci", 95))
    if metric == "tableak_score":
        _plot_known_subscores(df, "known_t_amount")
    else:
        _apply_metric_ylim(metric)
    save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_known_t", data))


def plot_known_tn(df, data, metric, out_dir):
    is_pct = metric in PERCENT_METRICS
    df_avg = df.groupby(["known_t_amount", "known_noise_amount"])[metric].mean().reset_index()
    pivot = df_avg.pivot(index="known_t_amount", columns="known_noise_amount", values=metric)
    fmt = ".1f" if is_pct else ".2f"
    ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap=COLOR_MAP_HEATMAP)
    if is_pct:
        for text in ax.texts:
            text.set_text(text.get_text() + "%")
    plt.xticks(
        ticks=[i + 0.5 for i in range(len(pivot.columns))],
        labels=[f"{x/100:.0%}" for x in pivot.columns],
    )
    plt.xlabel("Known Noise %")
    max_t = pivot.index.max()
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(pivot.index))],
        labels=[f"{x/max_t:.0%}" for x in pivot.index],
        rotation=0,
    )
    plt.ylabel("Known t %")
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    save_fig(fig_path(out_dir, f"{metric}_known_tn", data))


def plot_known_mask(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    sns.lineplot(data=df, x="known_mask_amount", y=metric, errorbar=("ci", 95))
    if metric == "tableak_score":
        _plot_known_subscores(df, "known_mask_amount")
    else:
        _apply_metric_ylim(metric)
    save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_known_mask", data))


def plot_known_tm(df, data, metric, out_dir):
    is_pct = metric in PERCENT_METRICS
    df_avg = df.groupby(["known_t_amount", "known_mask_amount"])[metric].mean().reset_index()
    pivot = df_avg.pivot(index="known_t_amount", columns="known_mask_amount", values=metric)
    fmt = ".1f" if is_pct else ".2f"
    ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap=COLOR_MAP_HEATMAP)
    if is_pct:
        for text in ax.texts:
            text.set_text(text.get_text() + "%")
    plt.xticks(
        ticks=[i + 0.5 for i in range(len(pivot.columns))],
        labels=[f"{x/100:.0%}" for x in pivot.columns],
    )
    plt.xlabel("Known Mask %")
    max_t = pivot.index.max()
    plt.yticks(
        ticks=[i + 0.5 for i in range(len(pivot.index))],
        labels=[f"{x/max_t:.0%}" for x in pivot.index],
        rotation=0,
    )
    plt.ylabel("Known t %")
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    save_fig(fig_path(out_dir, f"{metric}_known_tm", data))


def plot_known_tm_prior(df, data, metric, out_dir):
    """Three-panel figure: with-prior | no-prior | delta (with - no_prior)."""
    is_pct = metric in PERCENT_METRICS
    source_metric = metric_source(metric)
    df = df.copy()
    if "no_prior" not in df.columns:
        print(f"plot_known_tm_prior: no 'no_prior' column in {data} CSV, skipping")
        plt.clf()
        return
    if source_metric not in df.columns:
        print(f"plot_known_tm_prior: no '{source_metric}' column in {data} CSV, skipping")
        plt.clf()
        return
    df["no_prior"] = df["no_prior"].fillna(False).astype(bool)
    agg = df.groupby(["no_prior", "known_t_amount", "known_mask_amount"])[source_metric].mean()
    if False not in agg.index.get_level_values("no_prior") or True not in agg.index.get_level_values("no_prior"):
        print(f"plot_known_tm_prior: need both prior on/off rows for {data}, skipping")
        plt.clf()
        return
    piv_with = agg.loc[False].unstack("known_mask_amount")
    piv_no = agg.loc[True].unstack("known_mask_amount")
    delta = piv_with - piv_no

    fmt = ".1f" if is_pct else ".2f"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(min(piv_with.min().min(), piv_no.min().min()))
    vmax = float(max(piv_with.max().max(), piv_no.max().max()))
    for ax, piv, title in [
        (axes[0], piv_with, "With prior"),
        (axes[1], piv_no, "No prior"),
    ]:
        sns.heatmap(piv, ax=ax, annot=True, fmt=fmt, cmap=COLOR_MAP_HEATMAP, vmin=vmin, vmax=vmax)
        if is_pct:
            for text in ax.texts:
                text.set_text(text.get_text() + "%")
        ax.set_title(title)
        ax.set_xlabel("Known Mask %")
        ax.set_ylabel("Known t %")
    delta_abs_max = float(max(abs(delta.min().min()), abs(delta.max().max()), 1e-9))
    sns.heatmap(
        delta,
        ax=axes[2],
        annot=True,
        fmt=fmt if not is_pct else "+.1f",
        cmap="RdBu_r",
        center=0,
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )
    axes[2].set_title("Δ (prior contribution)")
    axes[2].set_xlabel("Known Mask %")
    axes[2].set_ylabel("Known t %")
    save_fig(fig_path(out_dir, f"{metric}_known_tm_prior", data))


def plot_known_tm_prior_1(df, data, metric, out_dir):
    """Three-panel for known-tm-prior at batch_size=1.

    With bs=1, (known_{t,mask}_amount * 1) // 100 maps {0,25,50,75} → 0 and {100} → 1.
    The 5×5 nominal grid collapses to a 2×2 effective grid: no known vs. fully known.
    """
    is_pct = metric in PERCENT_METRICS
    df = df.copy()
    if "no_prior" not in df.columns:
        print(f"plot_known_tm_prior_1: no 'no_prior' column in {data} CSV, skipping")
        plt.clf()
        return
    df["no_prior"] = df["no_prior"].fillna(False).astype(bool)

    batch_sz = int(df["batch_size"].iloc[0]) if "batch_size" in df.columns else 1
    df["eff_t"] = (df["known_t_amount"] * batch_sz) // 100
    df["eff_mask"] = (df["known_mask_amount"] * batch_sz) // 100

    agg = df.groupby(["no_prior", "eff_t", "eff_mask"])[metric].mean()
    if False not in agg.index.get_level_values("no_prior") or True not in agg.index.get_level_values("no_prior"):
        print(f"plot_known_tm_prior_1: need both prior on/off rows for {data}, skipping")
        plt.clf()
        return
    piv_with = agg.loc[False].unstack("eff_mask")
    piv_no = agg.loc[True].unstack("eff_mask")
    delta = piv_with - piv_no

    eff_labels = {0: "none", 1: "all"}
    fmt = ".1f" if is_pct else ".2f"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(min(piv_with.min().min(), piv_no.min().min()))
    vmax = float(max(piv_with.max().max(), piv_no.max().max()))
    for ax, piv, title in [
        (axes[0], piv_with, "With prior"),
        (axes[1], piv_no, "No prior"),
    ]:
        sns.heatmap(piv, ax=ax, annot=True, fmt=fmt, cmap=COLOR_MAP_HEATMAP, vmin=vmin, vmax=vmax)
        if is_pct:
            for text in ax.texts:
                text.set_text(text.get_text() + "%")
        ax.set_title(title)
        ax.set_xlabel("Known mask samples")
        ax.set_ylabel("Known t samples")
        ax.set_xticklabels([eff_labels.get(int(c), str(c)) for c in piv.columns], rotation=0)
        ax.set_yticklabels([eff_labels.get(int(i), str(i)) for i in piv.index], rotation=0)
    delta_abs_max = float(max(abs(delta.min().min()), abs(delta.max().max()), 1e-9))
    sns.heatmap(
        delta,
        ax=axes[2],
        annot=True,
        fmt=fmt if not is_pct else "+.1f",
        cmap="RdBu_r",
        center=0,
        vmin=-delta_abs_max,
        vmax=delta_abs_max,
    )
    axes[2].set_title("Δ (prior contribution)")
    axes[2].set_xlabel("Known mask samples")
    axes[2].set_ylabel("Known t samples")
    axes[2].set_xticklabels([eff_labels.get(int(c), str(c)) for c in delta.columns], rotation=0)
    axes[2].set_yticklabels([eff_labels.get(int(i), str(i)) for i in delta.index], rotation=0)
    save_fig(fig_path(out_dir, f"{metric}_known_tm_prior_1", data))


def plot_known_tnm(df, data, metric, out_dir):
    """Small-multiples: one (known_t x known_noise) heatmap per known_mask level."""
    is_pct = metric in PERCENT_METRICS
    mask_levels = sorted(df["known_mask_amount"].unique())
    n_panels = len(mask_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    vmin = float(df[metric].min())
    vmax = float(df[metric].max())
    fmt = ".1f" if is_pct else ".2f"
    for ax, m in zip(axes, mask_levels):
        sub = df[df["known_mask_amount"] == m]
        piv = (
            sub.groupby(["known_t_amount", "known_noise_amount"])[metric]
            .mean()
            .unstack("known_noise_amount")
        )
        sns.heatmap(
            piv,
            ax=ax,
            annot=True,
            fmt=fmt,
            cmap=COLOR_MAP_HEATMAP,
            vmin=vmin,
            vmax=vmax,
            cbar=(ax is axes[-1]),
        )
        if is_pct:
            for text in ax.texts:
                text.set_text(text.get_text() + "%")
        ax.set_title(f"Known Mask = {m}%")
        ax.set_xlabel("Known Noise %")
    axes[0].set_ylabel("Known t %")
    save_fig(fig_path(out_dir, f"{metric}_known_tnm", data))


def plot_pool_size(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    # get batch size from df
    batch_sizes = df["batch_size"].dropna().unique()
    if len(batch_sizes) > 1:
        print(f"Warning: multiple batch sizes found in pool size data for {data}: {batch_sizes}")
    batch_size = batch_sizes[0] if len(batch_sizes) > 0 else None
    # add pool_size=1 data point from accuracy-epochX experiment (epoch 0 only)
    try:
        present_fa = list(df["fast_align"].unique())
        present_pp = list(df["perfect_pooling"].unique())
        df_acc = load_csv(f"{MULTI_EPOCH_ACC}/{data}/performance_{data}.csv")
        df_acc = df_acc[(df_acc["epoch"] == 0) & (df_acc["batch_size"] == batch_size) & (df_acc["pool_size"] == 1)]
        try:
            df_acc["perfect_pooling"] = df_acc["perfect_pooling"].map({True: PERFECT_POOLING, False: NORMAL_POOLING})
        except Exception:
            pass
        df_acc = pd.concat(
            [
                df_acc.assign(fast_align=fa, perfect_pooling=pp)
                for fa in present_fa
                for pp in present_pp
            ],
            ignore_index=True,
        )
        df = pd.concat([df, df_acc], ignore_index=True)
        _refilter_align_categories(df)
    except FileNotFoundError:
        pass
    
    # if data point at 2, 4, 8 exist, remove as it's useless
    df = df[df["pool_size"] != 2]
    df = df[df["pool_size"] != 4]
    df = df[df["pool_size"] != 8]

    for diff_type in sorted(df["diffusion_type"].unique()):
        df_diff = df[df["diffusion_type"] == diff_type].copy()
        if df_diff.empty:
            continue

        style_col = "perfect_pooling" if df_diff["perfect_pooling"].nunique() > 1 else None

        ax = sns.lineplot(data=df_diff, x="pool_size", y=metric, errorbar=("ci", 95), hue="fast_align", style=style_col)
        plt.xlabel("Pool Size")
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        ax.xaxis.label.set_size(20)
        plt.xticks(ticks=df_diff["pool_size"].unique(), labels=df_diff["pool_size"].unique().astype(str))

        h, l = ax.get_legend_handles_labels()
        if style_col is not None:
            l[0] = "Alignment Algorithm"
            style_idx = df_diff["fast_align"].nunique() + 1
            if style_idx < len(l):
                l[style_idx] = "Alignment Basis"
            plt.legend(title=None, labels=l, handles=h, loc="lower right", prop={"size": LEGEND_FONT_SIZE_TINY})
        else:
            plt.legend(
                title="Alignment Algorithm",
                handles=h,
                labels=l,
                loc="lower right",
                prop={"size": LEGEND_FONT_SIZE_TINY},
            )

        _apply_metric_ylim(metric)
        save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_pool_size_{diff_type}", data))

        if metric == "tableak_score":
            _plot_tableak_subscores(df_diff, "pool_size", style_col, out_dir, f"tableak_subscores_pool_size_{diff_type}", data)


def plot_defense(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    # extract used batch size from defense experiment (should be consistent across rows)
    used_batch_size = df["batch_size"].dropna().unique()
    if len(used_batch_size) > 1:
        print(f"Warning: multiple batch sizes found in defense data for {data}: {used_batch_size}")
    used_batch_size = used_batch_size[0] if len(used_batch_size) > 0 else None

    # merge in no-defense data from accuracy-epochX experiment at epoch 1
    try:
        df_acc = load_csv(f"accuracy-epochX/{data}/performance_{data}.csv")
        df_acc = df_acc[df_acc["epoch"] == 0]
        df_acc = df_acc[df_acc["batch_size"] == used_batch_size]
        df = pd.concat([df, df_acc], ignore_index=True)
    except FileNotFoundError:
        pass
    df["defense"] = df["defense"].fillna("None")

    for diff_type in DIFFUSION_TYPES:
        tmp_df = df[df["diffusion_type"] == diff_type]
        if metric == "tableak_score":
            df_melted = tmp_df.melt(
                id_vars=["defense"],
                value_vars=["tableak_score", "tableak_cat_score", "tableak_cont_score"],
                var_name="metric",
                value_name="value",
            )
            ax = sns.barplot(
                data=df_melted,
                x="defense",
                y="value",
                hue="metric",
                hue_order=["tableak_cont_score", "tableak_score", "tableak_cat_score"],
            )
            plt.xticks(
                ticks=range(len(tmp_df["defense"].unique())),
                labels=[DEFENSES_MAP.get(d, d) for d in tmp_df["defense"].unique()],
                rotation=30,
                ha="right",
            )
            plt.ylabel("Reconstruction Accuracy (%)")
            plt.xlabel(None)
            h, l = ax.get_legend_handles_labels()
            plt.legend(
                title="Metric",
                labels=["Continuous Score", "Overall Score", "Categorical Score"],
                handles=h,
                loc="lower left",
                prop={"size": LEGEND_FONT_SIZE_SMALL},
            )
            save_fig(fig_path(out_dir, f"tableak_{diff_type}_defense", data))
        elif metric in PRIVACY_METRICS:
            sns.barplot(data=tmp_df, x="defense", y=metric)
            plt.xlabel("Defense Method")
            plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
            save_fig(fig_path(out_dir, f"{metric}_{diff_type}_defense", data))


LATENT_METRIC_DIFFUSION = {
    "t_mae": ("gaussian", "masked"),
    "noise_mse": ("gaussian",),
    "mask_accuracy": ("masked",),
}


def plot_latent_reconstruction(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    if metric not in df.columns or df[metric].isna().all():
        plt.clf()
        return
    ensure_diffusion_type(df)
    allowed = LATENT_METRIC_DIFFUSION.get(metric, ("gaussian", "masked"))
    df = df[df["diffusion_type"].isin(allowed) & df[metric].notna()]
    if df.empty:
        plt.clf()
        return

    use_hue = len(allowed) > 1
    sns.barplot(
        data=df,
        x="batch_size",
        y=metric,
        hue="diffusion_type" if use_hue else None,
    )
    plt.xlabel("Batch Size")
    plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
    save_fig(fig_path(out_dir, f"{metric}_latent", data))


def plot_init_strategy(df, data, metric, out_dir):
    sns.set_palette(COLOR_PALETTE_LINES)
    df = df[df["batch_size"] == INTEREPOCH_BATCH_SIZE].copy()
    df["epoch"] = df["epoch"] + 1
    df["init_strategy"] = df["init_strategy"].map(INIT_STRATEGY_MAP)

    ax = sns.lineplot(data=df, x="epoch", y=metric, errorbar=("ci", 95), hue="init_strategy", style="fast_align")
    plt.xlabel("Epoch")
    plt.ylabel(METRIC_NAME_MAP.get(metric, metric))

    h, l = ax.get_legend_handles_labels()
    l[0] = "Init Strategy"
    idx_align = len(INIT_STRATEGY_MAP) + 1
    if idx_align < len(l):
        l[idx_align] = "Alignment"
    plt.legend(title=None, labels=l, handles=h, prop={"size": LEGEND_FONT_SIZE})

    _apply_metric_ylim(metric)
    save_fig(fig_path(out_dir, f"{metric_prefix(metric)}_init_strategy", data))


# ── LaTeX table ───────────────────────────────────────────────────────────────


def init_table_metrics():
    table_metrics = {}
    for metric in METRICS_TO_PLOT:
        for diff_type in DIFFUSION_TYPES:
            table_metrics[(metric, diff_type)] = {m: {} for m in METHODS_BY_DIFFUSION[diff_type]}
    return table_metrics


def build_latex_table(methods, datasets, batch_sizes):
    n_cols = len(datasets) + 2
    col_spec = "cl" + "c" * len(datasets)
    header_cells = " & ".join(f"\\textbf{{{d.capitalize()}}}" for d in datasets)
    last_reference = next((m for m in reversed(methods) if m not in OUR_METHODS), None)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        f"\\textbf{{Batch Size}} & \\textbf{{Method}} & {header_cells}\\\\ \\hline",
    ]
    for bs in batch_sizes:
        lines.append(f"\\multirow{{{len(methods)}}}{{*}}{{{bs}}}")
        for method in methods:
            placeholders = " & ".join(["{}"] * len(datasets))
            label = "\\name 1-epoch" if method == NAME else method.replace(NAME, "\\name")
            row = f"& {label} & {placeholders} \\\\"
            if method == last_reference:
                row += f" \\cdashline{{2-{n_cols}}}"
            lines.append(row)
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    template = "\n".join(lines)
    MARKER = "\x00"
    template = template.replace("{}", MARKER)
    template = template.replace("{", "{{").replace("}", "}}")
    template = template.replace(MARKER, "{}")
    return template


def format_cell(score, std, is_percent):
    if isinstance(score, str) and score.startswith("\\textbf"):
        return f"${score}_{{{std:.1f}}}$"
    if is_percent:
        return f"${score:.1f}\\%_{{{std:.1f}}}$"
    return f"${score:.2f}_{{{std:.1f}}}$"


def print_latex_tables(table_metrics):
    for metric in ("tableak_score", "dcr"):
        for diff_type in DIFFUSION_TYPES:
            methods = METHODS_BY_DIFFUSION[diff_type]
            print(f"\n% Table: {metric} - {diff_type}")
            tmp = table_metrics[(metric, diff_type)]

            for batch_size in BATCH_SIZES:
                for data in ACCURACY_DATASETS:
                    best_score, best_method = None, None
                    for method in methods:
                        score_dict, _ = tmp[method].get(data, ({}, {}))
                        score = score_dict.get(batch_size)
                        if score is None:
                            continue
                        if best_score is None or (
                            (score < best_score) if metric in LOWER_IS_BETTER else (score > best_score)
                        ):
                            best_score, best_method = score, method
                    if best_method is not None:
                        fmt = (
                            f"\\textbf{{{best_score:.1f}\\%}}"
                            if metric in PERCENT_METRICS
                            else f"\\textbf{{{best_score:.2f}}}"
                        )
                        tmp[best_method][data][0][batch_size] = fmt

            data_values = []
            for batch_size in BATCH_SIZES:
                for method in methods:
                    for data in ACCURACY_DATASETS:
                        score_dict, std_dict = tmp[method].get(data, ({}, {}))
                        score = score_dict.get(batch_size)
                        data_values.append(
                            format_cell(score, std_dict.get(batch_size, 0), metric in PERCENT_METRICS)
                            if score is not None else "--"
                        )

            print(build_latex_table(methods, ACCURACY_DATASETS, BATCH_SIZES).format(*data_values))


# ── Entry point ───────────────────────────────────────────────────────────────

EXPERIMENT_PLOTTERS = {
    "accuracy": None,  # handled separately (needs table_metrics)
    "interepoch": plot_interepoch,
    "interepoch-simple": plot_interepoch,
    "ablation-tn": plot_ablation_tn,
    "ablation-loss": plot_ablation_loss,
    "ablation-prior": plot_ablation_prior,
    "known-noise": plot_known_noise,
    "known-t": plot_known_t,
    "known-tn": plot_known_tn,
    "known-mask": plot_known_mask,
    "known-tm": plot_known_tm,
    "known-tm-prior": plot_known_tm_prior,
    "known-tm-prior-1": plot_known_tm_prior_1,
    "known-tnm": plot_known_tnm,
    "pool-size": plot_pool_size,
    "pool-size-perfect": plot_pool_size,
    "defense": plot_defense,
    "init-strategy": plot_init_strategy,
    "latent-reconstruction": plot_latent_reconstruction,
    "pool-size-8": plot_pool_size,
    "interepoch-8": plot_interepoch,
    "interepoch-8-2": plot_interepoch,
}


def main():
    global EXPERIMENTS
    import argparse

    ap = argparse.ArgumentParser(description="Plot experiment results (pdf + svg).")
    ap.add_argument("--experiments", nargs="+", default=None, metavar="NAME",
                    help=f"subset to plot; default all. Available: {', '.join(ALL_EXPERIMENTS)}")
    args = ap.parse_args()
    if args.experiments:
        unknown = [e for e in args.experiments if e not in ALL_EXPERIMENTS]
        if unknown:
            print(f"unknown experiment(s): {unknown}")
            return
        EXPERIMENTS = args.experiments

    table_metrics = init_table_metrics()

    for experiment in EXPERIMENTS:
        metrics = EXPERIMENT_METRICS.get(experiment, METRICS_TO_PLOT)
        for metric in metrics:
            exp_dir = os.path.join(BASE_DIR, experiment)
            data_dir = os.path.join(DATA_BASE_DIR, experiment)

            for data in ACCURACY_DATASETS if experiment == "accuracy" else DATASETS:
                print(f"{experiment} - {data}")
                out_dir = os.path.join(exp_dir, data)

                # load CSV for this experiment/dataset
                try:
                    df = load_csv(f"{data_dir}/{data}/performance_{data}.csv")
                    os.makedirs(out_dir, exist_ok=True)
                except FileNotFoundError:
                    if experiment == "accuracy":
                        try:
                            epoch_experiment = CENSUS_EPOCH_ACCURACY[1] if data == "census" else MULTI_EPOCH_ACC
                            df = load_csv(f"{DATA_BASE_DIR}/{epoch_experiment}/{data}/performance_{data}.csv")
                            df = df[df["epoch"] == 0]
                            os.makedirs(out_dir, exist_ok=True)
                        except FileNotFoundError as e:
                            print(f"Could not read data for {experiment} - {data}: {e}")
                            continue
                    else:
                        continue

                apply_column_transforms(df)
                if experiment == "accuracy" and data == "census":
                    df = load_census_epochs().query("epoch == 0").copy()

                # skip time metrics except for interepoch
                if metric in TIME_METRICS and not experiment.startswith("interepoch"):
                    plt.clf()
                    continue

                if experiment == "accuracy":
                    plot_accuracy(df, data, metric, out_dir, table_metrics)
                elif experiment.startswith("interepoch"):
                    plot_interepoch(df, data, metric, out_dir, experiment)
                else:
                    plotter = EXPERIMENT_PLOTTERS.get(experiment)
                    if plotter:
                        plotter(df, data, metric, out_dir)
                    else:
                        print(f"No plotter for experiment: {experiment}")
                        plt.clf()

    print_latex_tables(table_metrics)


if __name__ == "__main__":
    main()
