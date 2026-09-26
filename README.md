# GOLIATH

Official code for **GOLIATH: Gradient Inversion of Tabular Diffusion Models** (NeurIPS 2026).

Giulio Segalini (Université de Neuchâtel), Aditya Shankar (TU Delft), Jérémie Decouchant (TU Delft),
Lydia Y. Chen (Université de Neuchâtel and TU Delft)

A server observes the gradient (or weight update) a client computes while training a tabular
diffusion model on a private batch. GOLIATH reconstructs that batch by jointly optimizing dummy rows
**and** the diffusion latents that produced the gradient: the timestep `t` and the noise `ε` for
Gaussian diffusion, or `t` and the feature mask for masked diffusion. Gradients flow to `t` through
interpolated noise-schedule coefficients. Categorical features are relaxed with temperature-annealed
softmax, and reconstructions from several epochs are aligned and ensembled.

## Setup

```bash
uv sync                                          # Python 3.12, pinned dependencies (pyproject.toml / uv.lock)
uv run python -m src.download_and_process_datasets
```

The download script fetches the UCI and OpenML datasets (abalone, adult, bean, census, credit,
default, gesture, letter, magic, shoppers). It writes `datasets/<name>/{data,train,test}.csv` and
`info.json` using a 70/30 split with seed 42. The optional `australia` dataset needs a manual
download from
[Kaggle](https://www.kaggle.com/datasets/luisblanche/quarterly-tourism-in-australia) saved as
`datasets/archive.zip`; it is skipped otherwise.

A CUDA GPU is strongly recommended.

## Running

A single attack run:

```bash
uv run main.py --dataname adult --attack --batch_sz 8 --epochs 1 --diffusion-type gaussian \
    --subset-size 16 --save-dir results/tests
```

Each run appends one row per attacked batch to `<save-dir>/performance_<dataset>.csv`. A row
records the TabLeak similarity and its subscores, DCR/NNDR, the latent-recovery errors,
reconstruction time and peak VRAM. `uv run main.py --help` lists every option. The most important
ones are:

| flag | meaning |
|---|---|
| `--diffusion-type {gaussian,masked}` | diffusion process being trained / attacked |
| `--batch_sz`, `--epochs`, `--timesteps` | FL training setup |
| `--no-t`, `--no-noise` | disable optimization of the corresponding latent |
| `--loss-fn` | gradient-matching loss (`cossim`, `mse`, `both`, `auto`) |
| `--pool-size`, `--no-align`, `--no-fast-align`, `--anchor` | pooling and cross-epoch alignment |
| `--defense {clipping,compression,noise,censor,...}` | gradient defenses |
| `--baseline {gidm,tableak}` | baseline attacks |
| `--skip_eval` | skip DCR/NNDR/TabLeak scoring |

### Paper experiments

Every experiment in the paper is registered in `experiments/params.py` and runs with one command:

```bash
uv run main.py --experiment <name>               # writes results/<name>/<dataset>/performance_*.csv
uv run experiments/params.py <name>              # list the configurations it will run
```

`--experiment` draws a fresh random seed per invocation and appends to the CSV. Invoking the same
experiment again therefore adds an independent seed.

| experiment | what it measures |
|---|---|
| `accuracy`, `accuracy-epochX` | main reconstruction accuracy vs batch size and epochs |
| `interepoch`, `interepoch-simple` | multi-epoch ensembling and alignment variants |
| `pool-size`, `pool-size-perfect` | parallel inversions and pooling |
| `ablation-tn`, `ablation-loss`, `ablation-prior`, `init-strategy` | ablations (latent optimization, loss, coupling prior, initialization) |
| `known-t`, `known-noise`, `known-tn`, `known-mask`, `known-tm`, `known-tm-prior`, `known-tnm` | attacks with partially known latents |
| `latent-reconstruction` | recovery quality of `t`, `ε` and the mask |
| `random-baseline` | random-latent reference |
| `defense` | clipping, compression, additive noise, CENSOR |
| `dp-defense-b1-c`, `dp-defense-eps-c`, `dp-utility-c` | DP-SGD with calibrated clipping: attack success vs ε, and model utility |
| `save-model` → `gidm-baseline` | GIDM baseline (`save-model` must run first; it pre-trains the models GIDM needs) |
| `tableak-baseline`, `tableak-baseline-1`, `tableak-adaptive` | TabLeak baseline, and TabLeak with latent optimization |
| `leakage-metrics` | per-cardinality, per-column, whole-row and re-identification leakage |
| `census-cardinality`, `accuracy-epochX-census`, `accuracy-epochX-census-b18` | high-cardinality census dataset |
| `align-diagnostic` | greedy vs Hungarian alignment on the same reconstructions |
| `reshuffle` | robustness to per-epoch reshuffling |
| `fedavg`, `fedavg-masked` | per-client FedAvg updates with K local steps |
| `ft-transformer-v2` | FT-Transformer backbone (run through `scripts/run_ft_v2_auto_seed.sh`) |
| `scalability` | time and memory vs batch size |

`scripts/reproduce_all.sh` runs the experiments behind the tables, cheapest first, and logs each
run to `results/repro_logs/`:

```bash
bash scripts/reproduce_all.sh --list             # what it will run
bash scripts/reproduce_all.sh --smoke            # one tiny config, checks the code paths
bash scripts/reproduce_all.sh --only fedavg reshuffle
bash scripts/reproduce_all.sh                    # everything (days on a single GPU)
```

The sequential total is several days on one GPU. The largest experiments are `accuracy-epochX` at
about 20 h and `interepoch` at about 12 h; most of the others take between 10 minutes and 6 hours.

### Tables and figures

Once the experiments have written their CSVs:

```bash
uv run analysis/build.py                         # health-check CSVs, then write tables + figures
uv run analysis/build.py --tables --only dp fedavg
uv run analysis/build.py --seeds                 # seed runs behind each cell, and how to add one
cd results && uv run plot.py --experiments accuracy defense   # per-experiment plots
```

Tables go to `results/camera_ready/tables/` as `.tex` and `.csv`. Figures go to
`results/camera_ready/figures/`, and per-experiment plots to `results/plots/<experiment>/<dataset>/`.

## Notes on reproducing

- **Don't interrupt an experiment mid-configuration.** A partial cell biases that cell's mean toward
  the batches that finished. `analysis/build.py` runs a health check first and flags row counts that
  are not a multiple of the expected stride; delete partial rows before aggregating.
- **`subset_size` is not logged to the CSV.** Don't merge the output of runs made with different
  subset sizes into one directory. `accuracy-epochX-census` and `accuracy-epochX-census-b18` are
  separate experiments for this reason.
- **Privacy budget.** ε is always recomputed from the logged `dp_noise_multiplier` with the exact
  analytic Gaussian accountant (`src/dp_accountant.py`), never read from the CSV. The DP clipping
  norm is calibrated per dataset to the median per-example gradient norm;
  `scripts/dp_diagnostics.py` measures these norms.
- **Standard deviations.** `fedavg` reports sample std (ddof=1). The leakage, cardinality and
  sensitive-attribute tables report population std (ddof=0).

## Repository layout

```
main.py                   CLI entry point
experiments/params.py     registry of paper experiments
src/
  train_diffusion.py      FL training loop; runs the attack and logs metrics
  attack.py               gradient inversion, alignment / ensembling, GIDM and TabLeak baselines
  diffusion_process.py    Gaussian and masked diffusion processes (training + attack-time relaxations)
  model.py                MLP and FT-Transformer denoisers
  defense.py              gradient defenses and DP-SGD
  dp_accountant.py        analytic Gaussian DP accountant
  metrics.py              TabLeak similarity and leakage metrics
  utils.py                preprocessing, matching, CSV logging
  download_and_process_datasets.py
analysis/                 tables and figures from the result CSVs
scripts/                  reproduction helpers
results/plot.py           per-experiment plots
```

## Citation

```bibtex
@inproceedings{goliath2026,
  title     = {{GOLIATH}: Gradient Inversion of Tabular Diffusion Models},
  author    = {Segalini, Giulio and Shankar, Aditya and Decouchant, J{\'e}r{\'e}mie and Chen, Lydia Y.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
