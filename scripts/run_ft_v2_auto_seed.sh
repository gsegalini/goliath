#!/usr/bin/env bash
# One extra SEED for the ft-transformer-v2 sweep, given a list of datasets.
#
#   bash scripts/run_ft_v2_auto_seed.sh adult abalone     # two datasets, sequentially
#
# Sharding is BY DATASET on purpose: each dataset writes its own performance_<ds>.csv, so parallel
# shards never append to the same file. The attack is launch/CPU-bound rather than compute-bound,
# so several shards run at close to free throughput on one GPU.
#
# WHY THIS EXISTS INSTEAD OF `--experiment ft-transformer-v2`:
# the registered experiment also defines loss_fn="both" cells, but the published FT tables are
# built from loss_fn="auto" only. Since analysis/tables._ft_cells averages every row at batch 32,
# running the full experiment would fold a DIFFERENT loss configuration into the published mean
# while appearing to add a seed. This script runs the auto cells only, so the rows it appends are
# genuine additional seeds of the published configuration.
#
# Cells must match experiments/params.py::_params_ft_transformer_v2 for loss_fn="auto":
#   batch {1,8,16,32} x {gaussian,masked}, 500 steps, 5 epochs, subset = max(batch*4, 32),
#   t+noise optimized (defaults), seed randomized (-1) exactly as --experiment does.
#
# --optim_iters MUST be passed explicitly. The batch-dependent budget lives in params.py, so a run
# launched this way would otherwise silently fall back to the 2000 default, which over-optimizes
# the gradient residual at larger batches and measurably LOWERS reconstruction accuracy
# (Adult batch 32: 0.4146 at 2000 steps vs 0.4479 at 1000).
set -u
cd "$(dirname "$0")/.."

iters_for () { case "$1" in 1|8) echo 2000 ;; 16|32) echo 1000 ;; *) echo 2000 ;; esac; }

for ds in "$@"; do
  for b in 1 8 16 32; do
    ss=$(( b * 4 )); [ "$ss" -lt 32 ] && ss=32
    it=$(iters_for "$b")
    for diff in gaussian masked; do
      uv run main.py --dataname "$ds" --attack --backbone ft-transformer \
        --timesteps 500 --batch_sz "$b" --epochs 5 --subset-size "$ss" \
        --optim_iters "$it" --diffusion-type "$diff" --loss-fn auto --seed -1 \
        --save-dir "results/ft-transformer-v2/$ds" >/dev/null 2>&1 \
        && echo "done $ds b=$b $diff" || echo "FAILED $ds b=$b $diff"
    done
  done
done
echo "AUTO-SEED SHARD DONE: $*"
