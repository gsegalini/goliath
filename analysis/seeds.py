"""How many independent seed runs sit behind each published cell.

This exists because "did we run it more than once?" is not answerable by eye: runs are appended
into the same CSV by `PerformanceManager(append=True)`, so three seed runs and one long run are the
same rows in the same order.

Two ways to count, in order of preference:

1. **The `seed` column.** `--experiment` sets `seed = -1`, which `main.py` resolves to a fresh
   random seed and logs. Distinct values in a cell = distinct runs. This is exact and needs no
   knowledge of the protocol. Rows with no logged seed (NaN) count as one run.
2. **Row stride**, for experiments whose CSVs have no `seed` column. One run writes
   `subset_size / batch * epochs` rows, and `subset_size` is NOT logged (see loaders.py), so the
   divisor must be declared in ROWS_PER_RUN below and kept in sync with experiments/params.py.

    uv run analysis/build.py --seeds

The report also flags cells whose runs wrote *different* row counts. That is not cosmetic: it means
either an interrupted run, or two runs at different `subset_size` -- the latter is invisible in the
CSV and silently reweights a pooled mean toward the longer run.
"""
from __future__ import annotations

import os

import pandas as pd

from analysis import loaders as L

# Fallback divisors, used only for experiments with no `seed` column.
# int   -> same for every batch;  dict -> per batch size
ROWS_PER_RUN: dict[str, object] = {
    "leakage-metrics": {1: 32, 8: 16, 32: 8},  # subset/batch, _lm_batch_subset = [(1,32),(8,128),(32,256)]
    "reshuffle": 50,
    "tableak-adaptive": {1: 60, 8: 30, 16: 30, 32: 30},  # subset = max(batch*30, 60)
    "align-diagnostic": {8: 8, 32: 4, 128: 2, 256: 2},  # subset/batch, subset = 256
}

# Experiments that log `seed`; the divisor is unnecessary for these.
SEED_COLUMN_EXPERIMENTS = [
    "ft-transformer-v2",
    "fedavg",
    "fedavg-masked",
    "census-cardinality",
    "dp-defense-b1-c",
    "dp-defense-eps-c",
    "accuracy-epochX-census",
    "accuracy-epochX-census-b18",
]

ALL = list(ROWS_PER_RUN) + SEED_COLUMN_EXPERIMENTS

# The grouping whose repetition counts as a seed.
CELL_KEYS: dict[str, list[str]] = {
    "fedavg": ["dataname", "batch_size", "fedavg_local_steps"],
    "fedavg-masked": ["dataname", "batch_size", "fedavg_local_steps"],
    "dp-defense-b1-c": ["dataname", "dp_noise_multiplier"],
    "dp-defense-eps-c": ["dataname", "dp_noise_multiplier"],
    "reshuffle": ["dataname", "diffusion_type", "reshuffle", "global_ensemble", "fast_align"],
    "census-cardinality": ["dataname"],
}
DEFAULT_CELL_KEY = ["dataname", "batch_size", "diffusion_type"]

# How to add one more seed. `--experiment` sets seed = -1, so re-invoking appends an independent
# run. ft-transformer-v2 uses a script that runs only the loss_fn="auto" cells the tables read.
RERUN: dict[str, str] = {
    "leakage-metrics": "uv run main.py --experiment leakage-metrics",
    "reshuffle": "uv run main.py --experiment reshuffle",
    "tableak-adaptive": "uv run main.py --experiment tableak-adaptive",
    "ft-transformer-v2": "bash scripts/run_ft_v2_auto_seed.sh abalone adult bean credit default gesture letter magic shoppers",
    "fedavg": "uv run main.py --experiment fedavg",
    "fedavg-masked": "uv run main.py --experiment fedavg-masked",
    "census-cardinality": "uv run main.py --experiment census-cardinality",
    "dp-defense-b1-c": "uv run main.py --experiment dp-defense-b1-c",
    "dp-defense-eps-c": "uv run main.py --experiment dp-defense-eps-c",
    "align-diagnostic": "uv run main.py --experiment align-diagnostic",
    "accuracy-epochX-census": "uv run main.py --experiment accuracy-epochX-census",
    "accuracy-epochX-census-b18": "uv run main.py --experiment accuracy-epochX-census-b18",
}


def _rows_per_run(exp: str, batch) -> int | None:
    rpr = ROWS_PER_RUN.get(exp)
    if isinstance(rpr, dict):
        try:
            return rpr.get(int(batch))
        except (TypeError, ValueError):
            return None
    return int(rpr) if rpr else None


def seed_counts(exp: str) -> dict:
    """Seed runs per cell for one experiment.

    Returns {'min', 'max', 'how', 'uneven': [...]}, where `uneven` names cells whose runs wrote
    differing row counts (interrupted run, or two runs at different subset_size).
    """
    df = L.load_experiment(exp)
    keys = [c for c in CELL_KEYS.get(exp, DEFAULT_CELL_KEY) if c in df.columns]
    by_seed = "seed" in df.columns and df["seed"].notna().any()
    counts, uneven = [], []

    for idx, g in df.groupby(keys, dropna=False):
        idx = idx if isinstance(idx, tuple) else (idx,)
        cell = {k: v for k, v in zip(keys, idx)}
        if by_seed:
            sizes = g.groupby("seed", dropna=False).size().tolist()
            n = len(sizes)
        else:
            rpr = _rows_per_run(exp, cell.get("batch_size", 0))
            if not rpr:
                continue
            n, rem = divmod(len(g), rpr)
            sizes = [rpr] * n + ([rem] if rem else [])
            if rem:  # a short run is still a run; count it, but say so
                n += 1
        counts.append(n)
        if n > 1 and len(set(sizes)) > 1:
            uneven.append(f"{cell}: rows per run {sorted(sizes)}")
        elif n == 1 and sizes and sizes[0] != _rows_per_run(exp, cell.get("batch_size", 0)) and not by_seed:
            uneven.append(f"{cell}: incomplete run, {sizes[0]} rows")

    return {
        "min": min(counts, default=0),
        "max": max(counts, default=0),
        "how": "seed column" if by_seed else "row stride",
        "uneven": uneven,
    }


def report(verbose: bool = True) -> None:
    print(f"{'experiment':30}{'seeds/cell':>12}  {'counted by':<13} add one more seed")
    print("-" * 108)
    for exp in ALL:
        if not os.path.isdir(os.path.join(L.RESULTS_DIR, exp)):
            print(f"{exp:30}{'MISSING':>12}")
            continue
        s = seed_counts(exp)
        span = str(s["min"]) if s["min"] == s["max"] else f"{s['min']}-{s['max']}"
        flag = "   <-- single run" if s["max"] <= 1 else ""
        print(f"{exp:30}{span:>12}  {s['how']:<13} {RERUN.get(exp, '')}{flag}")
        if verbose:
            for u in s["uneven"]:
                print(f"{'':30}{'':>12}  uneven: {u}")
