"""Data-integrity checks, run before any table is emitted.

Each check guards against a data problem that would otherwise silently produce a wrong number:
fail loudly instead of publishing a quietly-skewed mean.
"""
from __future__ import annotations

import os

from analysis import loaders as L

# Experiments the tables depend on, with the per-cell row count we expect.
# None = variable/unchecked row count.
REQUIRED = {
    "fedavg": None,
    "fedavg-masked": None,
    "leakage-metrics": None,
    "accuracy-epochX": None,
    "accuracy-epochX-census": 500,
    "accuracy-epochX-census-b18": 250,
    "census-cardinality": 32,
    "tableak-adaptive": None,
    "reshuffle": None,
    "dp-defense-b1-c": 64,
    "dp-defense-eps-c": 64,
    "dp-utility-c": None,
    "interepoch": None,
    "align-diagnostic": None,
    # expected=None on purpose: the tables use the loss_fn="auto" cells only
    # (analysis/tables._ft_cells), and an interrupted shard can leave per-cell counts that are
    # not a multiple of the run length.
    "ft-transformer-v2": None,
    "random-baseline": None,
}


def check() -> list[str]:
    """Return a list of problems; empty means healthy."""
    problems: list[str] = []

    for exp, expected in REQUIRED.items():
        if not os.path.isdir(os.path.join(L.RESULTS_DIR, exp)):
            problems.append(f"MISSING dir: results/{exp}")
            continue
        try:
            df = L.load_experiment(exp)
        except Exception as e:  # noqa: BLE001 - surface any load failure
            problems.append(f"UNREADABLE results/{exp}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            problems.append(f"EMPTY results/{exp}")
            continue

        if expected:
            key = [c for c in ("dataname", "batch_size", "diffusion_type", "dp_noise_multiplier")
                   if c in df.columns]
            sizes = df.groupby(key, dropna=False).size()
            for idx, n in sizes[sizes % expected != 0].items():
                where = dict(zip(key, idx if isinstance(idx, tuple) else (idx,)))
                problems.append(
                    f"PARTIAL CELL results/{exp} {where}: {n} rows, not a multiple of "
                    f"{expected} -- an interrupted run skews this cell's mean"
                )

    # DP: epsilon is recomputed from z, so the noise multiplier must be logged
    for exp in ("dp-defense-b1-c", "dp-defense-eps-c", "dp-utility-c"):
        if os.path.isdir(os.path.join(L.RESULTS_DIR, exp)):
            if "dp_noise_multiplier" not in L.load_experiment(exp).columns:
                problems.append(f"results/{exp} lacks dp_noise_multiplier; epsilon cannot be recomputed")

    # seed-run counts: an appended run should be noticed, not silently averaged in
    try:
        rm = L.load_experiment("leakage-metrics")
        for ds, cell in rm.groupby("dataname"):
            for b, rpr in L.ROWS_PER_RUN_LEAKAGE_METRICS.items():
                g = cell[(cell.batch_size == b) & (cell.diffusion_type == "gaussian")]
                if len(g) and len(g) % rpr:
                    problems.append(
                        f"leakage-metrics {ds} batch {b}: {len(g)} rows is not a multiple of "
                        f"{rpr} (rows per seed run) -- partial run present"
                    )
    except FileNotFoundError:
        pass
    return problems


def report(problems: list[str]) -> bool:
    """Print the report; return True if healthy."""
    if not problems:
        print("healthcheck: OK")
        return True
    print(f"healthcheck: {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    return False
