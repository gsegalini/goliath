#!/usr/bin/env python3
"""Build camera-ready tables and figures from the results CSVs.

    uv run analysis/build.py                     # healthcheck + tables + figures
    uv run analysis/build.py --tables            # only tables (.tex + .csv)
    uv run analysis/build.py --figures           # only figures (.pdf + .svg)
    uv run analysis/build.py --only dp fedavg    # restrict to named tables
    uv run analysis/build.py --seeds             # seed runs per cell + how to add one

The healthcheck runs first and refuses to emit tables from incomplete or inconsistent data
(--skip-healthcheck to override).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import emit, healthcheck  # noqa: E402
from analysis.tables import TABLES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_TABLES = os.path.join(ROOT, "results", "camera_ready", "tables")


def build_tables(keys: list[str]) -> dict[str, object]:
    frames = {}
    for k in keys:
        spec = TABLES[k]
        frames[k] = spec.fn()
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true", help="write .tex and .csv")
    ap.add_argument("--figures", action="store_true", help="write .pdf and .svg")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="restrict to these table keys")
    ap.add_argument("--skip-healthcheck", action="store_true")
    ap.add_argument("--seeds", action="store_true",
                    help="report independent seed runs per cell, and how to add one")
    args = ap.parse_args()

    if args.seeds:
        from analysis import seeds

        seeds.report()
        return 0

    keys = args.only or list(TABLES)
    unknown = [k for k in keys if k not in TABLES]
    if unknown:
        print(f"unknown table(s): {unknown}\navailable: {list(TABLES)}")
        return 2

    if not args.skip_healthcheck:
        if not healthcheck.report(healthcheck.check()):
            print("\nrefusing to emit tables while the data is unhealthy (--skip-healthcheck to override)")
            return 1

    frames = build_tables(keys)

    do_all = not (args.tables or args.figures)

    if args.tables or do_all:
        for k, df in frames.items():
            spec = TABLES[k]
            emit.write_latex(df, spec, os.path.join(OUT_TABLES, f"tab_{k}.tex"))
            emit.write_csv(df, os.path.join(OUT_TABLES, f"tab_{k}.csv"))
        print(f"wrote {len(frames)} table(s) x (.tex + .csv) to {os.path.relpath(OUT_TABLES, ROOT)}")

    if args.figures or do_all:
        from analysis import figures

        figures.build_all()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
