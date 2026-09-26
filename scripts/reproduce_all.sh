#!/usr/bin/env bash
# Re-run the experiments behind the paper's tables, then build them with `uv run analysis/build.py`.
#
#   bash scripts/reproduce_all.sh                       # everything, cheapest first
#   bash scripts/reproduce_all.sh --only fedavg reshuffle
#   bash scripts/reproduce_all.sh --smoke               # tiny config per path, just checks it runs
#   bash scripts/reproduce_all.sh --list
set -uo pipefail
cd "$(dirname "$0")/.."

# Cheapest first, so a broken path fails fast rather than 20 hours in.
# Repeated entries are intentional: each invocation appends one independently seeded run, and the
# paper reports census-cardinality over 2 runs, leakage-metrics as mean +- std over 3 runs, and
# ft-transformer-v2 over 2 runs.
EXPERIMENTS=(
  align-diagnostic
  census-cardinality
  census-cardinality
  fedavg
  fedavg-masked
  reshuffle
  dp-defense-eps-c
  dp-defense-b1-c
  dp-utility-c
  accuracy-epochX-census-b18
  accuracy-epochX-census
  leakage-metrics
  leakage-metrics
  leakage-metrics
  tableak-adaptive
  random-baseline
  interepoch
  ft-transformer-v2
  ft-transformer-v2
  accuracy-epochX
)

# ft-transformer-v2 is run through scripts/run_ft_v2_auto_seed.sh, which runs only the
# loss_fn="auto" cells the tables read (one seed per invocation).
FT_DATASETS=(abalone adult bean credit default gesture letter magic shoppers)

LOG_DIR="results/repro_logs"
ONLY=(); SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do ONLY+=("$1"); shift; done ;;
    --smoke) SMOKE=1; shift ;;
    --list)  printf '%s\n' "${EXPERIMENTS[@]}" | awk '!seen[$0]++'; exit 0 ;;
    *) echo "unknown flag: $1"; exit 2 ;;
  esac
done

run_one() {
  if [[ $SMOKE -eq 1 ]]; then
    # one tiny config: proves the code path runs without spending hours on it
    uv run main.py --dataname adult --attack --batch_sz 8 --epochs 1 --timesteps 500 \
      --subset-size 16 --optim_iters 30 --skip_eval --save-dir results/tests/smoke_repro
    return
  fi
  case "$1" in
    ft-transformer-v2) bash scripts/run_ft_v2_auto_seed.sh "${FT_DATASETS[@]}" ;;
    *) uv run main.py --experiment "$1" ;;
  esac
}

mkdir -p "$LOG_DIR"
TARGETS=("${EXPERIMENTS[@]}")
if [[ ${#ONLY[@]} -gt 0 ]]; then TARGETS=("${ONLY[@]}"); fi

started=$(date +%s); failed=()
for exp in "${TARGETS[@]}"; do
  log="$LOG_DIR/${exp}.log"
  echo ">>> $exp  ($(date '+%F %T'))  -> $log"
  t0=$(date +%s)

  run_one "$exp" >>"$log" 2>&1
  rc=$?

  dt=$(( $(date +%s) - t0 ))
  if [[ $rc -ne 0 ]]; then
    echo "    FAILED (rc=$rc) after ${dt}s -- see $log"
    failed+=("$exp")
  else
    echo "    done in ${dt}s"
  fi
done

echo
echo "total $(( ($(date +%s) - started) / 60 )) min"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "failed: ${failed[*]}"
  exit 1
fi
echo "Now build the tables and figures:"
echo "  uv run analysis/build.py"
