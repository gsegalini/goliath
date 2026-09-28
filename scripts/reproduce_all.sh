#!/usr/bin/env bash
# Re-run the experiments behind the paper's tables. See REPRODUCE.md.
#
#   bash scripts/reproduce_all.sh                       # everything, cheapest first
#   bash scripts/reproduce_all.sh --only fedavg reshuffle
#   bash scripts/reproduce_all.sh --smoke               # tiny config per path, just checks it runs
#   bash scripts/reproduce_all.sh --list
#
# You do NOT need this to rebuild the paper: the results are committed and
# `uv run analysis/build.py` regenerates every table and figure from them.
set -uo pipefail
cd "$(dirname "$0")/.."

# Cheapest first, so a broken path fails fast rather than 20 hours in.
# rebuttal-metrics appears 3x on purpose: the paper reports mean +- std over 3 appended seed runs.
EXPERIMENTS=(
  align-diagnostic
  fedavg
  fedavg-masked
  reshuffle
  dp-defense-eps-c
  dp-defense-b1-c
  dp-utility-c
  accuracy-epochX-census-b18
  accuracy-epochX-census
  rebuttal-metrics
  rebuttal-metrics
  rebuttal-metrics
  tableak-adaptive
  random-baseline
  interepoch
  accuracy-epochX
)

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

# census-cardinality and ft-transformer-v2 are not plain --experiment runs; see REPRODUCE.md.
run_special() {
  case "$1" in
    census-cardinality)
      for i in 1 2; do
        uv run main.py --dataname census --attack --batch_sz 1 --epochs 1 --timesteps 500 \
          --subset-size 32 --seed -1 --save-dir results/census-cardinality/census || return 1
      done ;;
    *) return 127 ;;
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

  if [[ $SMOKE -eq 1 ]]; then
    # one tiny config: proves the code path runs without spending hours on it
    uv run main.py --dataname adult --attack --batch_sz 8 --epochs 1 --timesteps 500 \
      --subset-size 16 --optim_iters 30 --skip_eval --save-dir results/tests/smoke_repro \
      >>"$log" 2>&1
  elif run_special "$exp" >>"$log" 2>&1; then
    :
  elif [[ $? -eq 127 ]]; then
    uv run main.py --experiment "$exp" >>"$log" 2>&1
  fi

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
echo "Now verify the numbers still reproduce:"
echo "  uv run analysis/build.py --check"
