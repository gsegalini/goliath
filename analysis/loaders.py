"""Results-CSV access with the aggregation rules the paper actually used.

Every function here exists because a naive version of it produced a wrong number during the
rebuttal. Each docstring names the failure it prevents, so the rule survives the next person who
reads only the code.

Layout: results/<experiment>/<dataset>/performance_<dataset>.csv, one row per attacked batch per
epoch, config and metric columns in one wide frame.

Note: `subset_size` is NOT logged to the CSV. Two runs of the same experiment at different subset
sizes are therefore indistinguishable inside a single file, which is why differing-protocol runs
must live in separate directories (e.g. accuracy-epochX-census at 100 batches/epoch vs
accuracy-epochX-census-b18 at 50).
"""
from __future__ import annotations

import glob
import os

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Directories whose numbers were superseded. Loading them by name is refused, because both
# mistakes are silent: the old dir has the same schema and plausible-looking values.
SUPERSEDED = {
    "ft-transformer": (
        "ft-transformer-v2",
        "v1 predates the _prepare_double_cossim fix on the FT-Transformer path and uses the "
        "untuned backbone; it reports 0.360/0.359 where the paper reports 0.465/0.422. "
        "The data is archived at results/archive/ft-transformer-v1-superseded/ "
        "(see its SUPERSEDED.md).",
    ),
    "dp-defense": (
        "dp-defense-b1-c",
        "batch 256 with clip C=1.0; the paper reports batch 1 with C calibrated to the median "
        "per-example norm (0.3 adult / 0.4 credit). Kept only for the trainable-regime comparison.",
    ),
    "dp-defense-b1": (
        "dp-defense-b1-c",
        "uncalibrated C=1.0 sweep, superseded by the calibrated-clip re-run.",
    ),
    "dp-defense-eps": ("dp-defense-eps-c", "uncalibrated C=1.0 epsilon-targeted sweep."),
    "dp-utility": ("dp-utility-c", "uncalibrated C=1.0 utility sweep."),
}

# Rows written per seed-run per (batch, diffusion) cell, i.e. subset_size / batch.
# rebuttal-metrics uses (batch, subset) in {(1,32), (8,128), (32,256)}.
ROWS_PER_RUN_REBUTTAL_METRICS = {1: 32, 8: 16, 32: 8}


def _dataset_csvs(experiment: str) -> list[str]:
    return sorted(glob.glob(os.path.join(RESULTS_DIR, experiment, "*", "performance_*.csv")))


def ensure_diffusion_type(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing/NaN diffusion_type with 'gaussian' (mirrors results/plot.py)."""
    if "diffusion_type" not in df.columns:
        df["diffusion_type"] = "gaussian"
    df["diffusion_type"] = df["diffusion_type"].fillna("gaussian")
    return df


def load_experiment(experiment: str, allow_superseded: bool = False) -> pd.DataFrame:
    """Load one experiment as a single frame, with a `dataname` column guaranteed.

    Raises on a superseded directory. That guard is the reason published Tables 12/13 cannot
    silently regress to the v1 FT-Transformer numbers.
    """
    if experiment in SUPERSEDED and not allow_superseded:
        replacement, why = SUPERSEDED[experiment]
        raise ValueError(
            f"'{experiment}' is superseded by '{replacement}'. {why}\n"
            f"Pass allow_superseded=True only to inspect the historical run."
        )
    paths = _dataset_csvs(experiment)
    if not paths:
        raise FileNotFoundError(f"no CSVs under results/{experiment}/*/performance_*.csv")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "dataname" not in df.columns:
            df["dataname"] = os.path.basename(os.path.dirname(p))
        frames.append(df)
    return ensure_diffusion_type(pd.concat(frames, ignore_index=True))


def split_seed_runs(df: pd.DataFrame, rows_per_run: int, n_runs: int | None = None) -> list[pd.DataFrame]:
    """Split one (dataset, batch, diffusion) cell into its independent seed runs.

    Runs are appended by PerformanceManager(append=True) at a FIXED STRIDE of `rows_per_run`
    (= subset_size / batch). Chunking by len(df)//n_runs instead is wrong whenever datasets have
    different run counts -- adult accumulated a fourth rebuttal-metrics run while the others had
    three, and the naive split silently misaligned every cell.

    Incomplete trailing runs are dropped rather than averaged in.
    """
    out = []
    total = len(df)
    limit = total // rows_per_run if n_runs is None else n_runs
    for r in range(limit):
        seg = df.iloc[r * rows_per_run : (r + 1) * rows_per_run]
        if len(seg) == rows_per_run:
            out.append(seg)
    return out


def complete_cells_only(df: pd.DataFrame, group_cols: list[str], expected_rows: int) -> pd.DataFrame:
    """Drop groups that did not reach `expected_rows`.

    Abalone's batch-1 DP sweep and a census masked cell were both stopped mid-cell; including the
    partial rows shifts the cell mean toward whichever batches happened to finish first.
    """
    return df.groupby(group_cols, dropna=False).filter(lambda g: len(g) >= expected_rows)


def epsilon_from_z(z: float, delta: float = 1e-5) -> float:
    """Exact analytic Gaussian-mechanism epsilon for a single release (Balle & Wang, 2018).

    NEVER read the CSV column `dp_achieved_epsilon`: those values were written by an earlier RDP
    accountant and are looser than what we can certify. Epsilon is always recomputed from the
    logged `dp_noise_multiplier`.
    """
    import sys

    sys.path.insert(0, os.path.dirname(RESULTS_DIR))
    from src.dp_accountant import analytic_gaussian_epsilon

    return float("inf") if z <= 0 else analytic_gaussian_epsilon(z, delta)


def latent_ablation_reference(dataset: str, batch: int) -> float:
    """The `random_latents == "all"` score: the attack with t/noise/mask frozen at random values.

    This is NOT a chance floor. The dummy data is still optimised against the gradient, so the
    value sits above true chance; it isolates the contribution of recovering the latents. It is
    also measured under MASKED diffusion only, so comparing it against a gaussian sweep is
    indicative rather than exact. Both caveats are stated in the paper.
    """
    df = load_experiment("random-baseline")
    m = df[(df.dataname == dataset) & (df.batch_size == batch) & (df.random_latents == "all")]
    return float(m.tableak_score.mean())


def mean_std_over_runs(runs: list[pd.DataFrame], column: str) -> tuple[float, float]:
    """Per-run means, then mean and (population) std across runs -- the paper's convention."""
    import numpy as np

    vals = [float(r[column].dropna().mean()) for r in runs if r[column].notna().any()]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def epoch_slice(df: pd.DataFrame, epochs_observed: int) -> pd.DataFrame:
    """Rows for "k epochs observed", i.e. the 0-indexed epoch k-1.

    Off-by-one here silently shifts the whole accuracy-versus-epochs table.
    """
    return df[df.epoch == epochs_observed - 1]
