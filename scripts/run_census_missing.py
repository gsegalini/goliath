"""Run the Census cells missing from the main accuracy tables.

Runs one configuration at a time and refuses to append to a partial cell. The
small-batch Census sweeps are kept separate from batch 16/32 because they have
different numbers of attacked batches per epoch.

    uv run scripts/run_census_missing.py
"""

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path("/tmp/census-main-runs")


@dataclass(frozen=True)
class Run:
    name: str
    result_dir: str
    batch: int
    diffusion: str
    rows: int
    epochs: int
    flags: tuple[str, ...] = ()
    random_latents: str = ""

    @property
    def csv_path(self) -> Path:
        return ROOT / self.result_dir / "census" / "performance_census.csv"

    def existing_rows(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open(newline="") as source:
            return sum(
                1 for row in csv.DictReader(source)
                if row["batch_size"] == str(self.batch)
                and row["diffusion_type"] == self.diffusion
                and (row.get("random_latents") or "") == self.random_latents
            )

    def command(self) -> list[str]:
        args = [
            "uv", "run", "main.py", "--dataname", "census", "--attack",
            "--batch_sz", str(self.batch), "--epochs", str(self.epochs),
            "--timesteps", "500", "--seed", "-1",
            "--subset-size", str(self.batch * self.rows // self.epochs),
            "--save-dir", str(ROOT / self.result_dir / "census"),
        ]
        if self.diffusion == "masked":
            args += ["--diffusion-type", "masked"]
        return args + list(self.flags)


def runs() -> list[Run]:
    jobs = [
        Run("goliath-b16-gaussian", "results/accuracy-epochX-census-b16", 16, "gaussian", 500, 5),
        Run("goliath-b16-masked", "results/accuracy-epochX-census-b16", 16, "masked", 500, 5),
    ]
    for batch in (1, 8, 16, 32):
        jobs.append(Run(
            f"tableak-{batch}", "results/tableak-baseline-1", batch, "gaussian", 50, 1,
            ("--baseline", "tableak", "--no-t", "--no-noise", "--pool-size", "1", "--loss-fn", "cossim"),
        ))
        for variant in ("mask", "all"):
            jobs.append(Run(
                f"random-{variant}-{batch}", "results/random-baseline", batch, "masked", 50, 1,
                ("--random-latents", variant, "--disable-prior", "--loss-fn", "cossim"),
                variant,
            ))
        for diffusion in ("gaussian", "masked"):
            subset = max(batch * 30, 60)
            jobs.append(Run(
                f"adaptive-{diffusion}-{batch}", "results/tableak-adaptive", batch, diffusion,
                subset // batch, 1,
                ("--baseline", "tableak", "--pool-size", "1", "--loss-fn", "cossim")
                + (("--disable-prior",) if diffusion == "masked" else ()),
            ))
    return jobs


def execute(command: list[str], name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"starting {name}; log: {log_path}", flush=True)
    with log_path.open("w") as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--goliath-only", action="store_true")
    parser.add_argument("--gidm-only", action="store_true")
    args = parser.parse_args()
    if sum((args.baselines_only, args.goliath_only, args.gidm_only)) > 1:
        parser.error("choose one run group")

    for job in runs():
        if args.gidm_only:
            continue
        if args.baselines_only and job.name.startswith("goliath-"):
            continue
        if args.goliath_only and not job.name.startswith("goliath-"):
            continue
        found = job.existing_rows()
        if found == job.rows:
            print(f"complete {job.name}: {found} rows", flush=True)
            continue
        if found:
            raise RuntimeError(f"{job.name} has {found}/{job.rows} rows; inspect the partial cell before resuming")
        execute(job.command(), job.name)
        found = job.existing_rows()
        if found != job.rows:
            raise RuntimeError(f"{job.name} produced {found}/{job.rows} rows")
        print(f"complete {job.name}: {found} rows", flush=True)

    if args.goliath_only or args.baselines_only:
        return

    checkpoint = ROOT / "saved_models/census/gaussian_mlp_model_pretrained.pt"
    if not checkpoint.exists():
        execute([
            "uv", "run", "main.py", "--dataname", "census", "--save-model",
            "--batch_sz", "1024", "--epochs", "1000", "--timesteps", "500",
            "--seed", "-1", "--save-dir", str(ROOT / "results/census-pretraining"),
        ], "gidm-pretrain")
    if not checkpoint.exists():
        raise RuntimeError("GIDM pretraining did not write its checkpoint")
    for batch in (1, 8, 16, 32):
        job = Run(
            f"gidm-{batch}", "results/gidm-baseline", batch, "gaussian", 50, 1,
            ("--baseline", "gidm", "--loss-fn", "mse"),
        )
        found = job.existing_rows()
        if found == job.rows:
            print(f"complete {job.name}: {found} rows", flush=True)
            continue
        if found:
            raise RuntimeError(f"{job.name} has {found}/{job.rows} rows; inspect the partial cell before resuming")
        execute(job.command(), job.name)
        found = job.existing_rows()
        if found != job.rows:
            raise RuntimeError(f"{job.name} produced {found}/{job.rows} rows")
        print(f"complete {job.name}: {found} rows", flush=True)


if __name__ == "__main__":
    main()
