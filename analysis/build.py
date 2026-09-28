#!/usr/bin/env python3
"""Build camera-ready tables and figures, and guard the published numbers.

    uv run analysis/build.py                     # healthcheck + tables + figures
    uv run analysis/build.py --check             # only: do the numbers still match fixtures?
    uv run analysis/build.py --tables            # only tables (.tex + .csv)
    uv run analysis/build.py --figures           # only figures (.pdf + .svg)
    uv run analysis/build.py --update-fixtures   # re-bless fixtures, printing what changed
    uv run analysis/build.py --only dp fedavg    # restrict to named tables
    uv run analysis/build.py --seeds             # seed runs per cell + how to add one

`--check` is the acceptance test: it proves the extractor still reproduces the numbers in the
paper. When a number legitimately changes (a new seed run, a corrected aggregation), re-bless with
`--update-fixtures`, read the printed diff, and commit the fixture change alongside the paper edit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import emit, healthcheck  # noqa: E402
from analysis.tables import TABLES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_TABLES = os.path.join(ROOT, "results", "camera_ready", "tables")
FIXTURES = os.path.join(HERE, "fixtures.json")
TOL = 5e-4


def _flatten(df) -> dict[str, str]:
    """Table -> {"<row-label>|<column>": "<cell>"} so fixtures are readable and diffable."""
    out = {}
    first = df.columns[0]
    for _, row in df.iterrows():
        rk = str(row[first])
        for c in df.columns[1:]:
            out[f"{rk}|{c}"] = str(row[c])
    return out


def _load_fixtures() -> dict:
    if not os.path.exists(FIXTURES):
        return {}
    with open(FIXTURES) as f:
        return json.load(f)


def build_tables(keys: list[str]) -> dict[str, object]:
    frames = {}
    for k in keys:
        spec = TABLES[k]
        frames[k] = spec.fn()
    return frames


def cmd_check(frames: dict, fixtures: dict) -> bool:
    if not fixtures:
        print("no fixtures yet -- run with --update-fixtures to create them")
        return False
    ok = True
    for k, df in frames.items():
        want = fixtures.get(k)
        if want is None:
            print(f"  {k}: NOT IN FIXTURES (new table -- re-bless to pin it)")
            ok = False
            continue
        got = _flatten(df)
        diffs = []
        for cell, wv in want.items():
            gv = got.get(cell)
            if gv is None:
                diffs.append((cell, wv, "<missing>"))
            elif gv != wv:
                try:  # numeric tolerance where both sides parse as a plain number
                    if abs(float(gv) - float(wv)) <= TOL:
                        continue
                except ValueError:
                    pass
                diffs.append((cell, wv, gv))
        extra = [c for c in got if c not in want]
        if diffs or extra:
            ok = False
            print(f"  {k}: {len(diffs)} changed, {len(extra)} new")
            for cell, wv, gv in diffs[:8]:
                print(f"      {cell}\n        fixture: {wv}\n        current: {gv}")
            if len(diffs) > 8:
                print(f"      ... and {len(diffs)-8} more")
            for c in extra[:4]:
                print(f"      new cell: {c} = {got[c]}")
        else:
            print(f"  {k}: OK ({len(want)} cells)")
    return ok


def cmd_update(frames: dict, fixtures: dict) -> None:
    import datetime

    new = {k: _flatten(df) for k, df in frames.items()}
    changed = 0
    for k, cells in new.items():
        old = fixtures.get(k, {})
        for cell, v in cells.items():
            if old.get(cell) != v:
                print(f"  {k} {cell}: {old.get(cell, '<new>')} -> {v}")
                changed += 1
    # A targeted --only update must preserve fixtures for every other table.
    payload = {**fixtures, **new}
    payload["_meta"] = {
        "blessed": datetime.date.today().isoformat(),
        "note": "Pinned published values. Regenerate with: uv run analysis/build.py --update-fixtures",
        "tolerance": TOL,
    }
    with open(FIXTURES, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"fixtures written ({changed} cell(s) changed)" if changed else "fixtures written (no changes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare against fixtures and exit non-zero on drift")
    ap.add_argument("--tables", action="store_true", help="write .tex and .csv")
    ap.add_argument("--figures", action="store_true", help="write .pdf and .svg")
    ap.add_argument("--update-fixtures", action="store_true", help="re-bless fixtures from current data")
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
        if not healthcheck.report(healthcheck.check()) and not args.update_fixtures:
            print("\nrefusing to emit tables while the data is unhealthy (--skip-healthcheck to override)")
            return 1

    frames = build_tables(keys)
    fixtures = _load_fixtures()

    do_all = not (args.check or args.tables or args.figures or args.update_fixtures)

    if args.update_fixtures:
        cmd_update(frames, fixtures)
        return 0

    if args.check:
        print("checking against fixtures:")
        return 0 if cmd_check(frames, fixtures) else 1

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
