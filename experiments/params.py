from tqdm.contrib import itertools as tqit


# main accuracy experiment
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers"]
timesteps_list = [500]
batch_sizes = [1, 8, 256, 512][::-1]
epochs_amounts = [1]
params_opt = [(True, True)]  # , ("simple", None)] # optimize both or do simple attack
runs = 1
extra_args = [None, {"diffusion_type": "masked"}]  # [{"pool_size": 8}]
params_acc = tqit.product(datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args)

# main accuracy, but up to epoch X
EPOCH_MAX = 5
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
timesteps_list = [500]
batch_sizes = [1, 8, 16, 32][::-1]
epochs_amounts = [EPOCH_MAX]
params_opt = [(True, True)]  # , ("simple", None)] # optimize both or do simple attack
runs = 1
extra_args = [None, {"diffusion_type": "masked"}] # [None] # [{"pool_size": 8}]
params_acc_epochX = tqit.product(
    datanames + ["census"], timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt,
    extra_args,
)

# Census-only slice of the same grid. `census` is appended to accuracy-epochX above so a full
# re-run covers it, but running that whole experiment again just to add one dataset would redo
# 9 datasets we already have; this runs the new one alone, into results/accuracy-epochX-census/.
params_acc_epochX_census = tqit.product(
    ["census"], timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# Small-batch census slice: batch 1 and 8, both diffusion types, 5 epochs, 50 batches per epoch.
# The full grid above costs ~10h because subset defaults to batch*100; fixing 50 batches per cell
# caps it at 250 rows per cell (1000 total) so it finishes overnight alongside the transformer work.
#
# Separate experiment (own results dir) rather than appending to accuracy-epochX-census: that run
# used 100 batches per epoch, and `subset_size` is NOT logged to the CSV, so the two protocols
# would be indistinguishable in a single file. Compare across dirs, not within one.
_census_b18 = [(1, 50), (8, 400)]  # (batch, subset) -- subset = 50 * batch


def _params_acc_epochX_census_b18():
    for batch, subset in _census_b18:
        for diffusion in ({}, {"diffusion_type": "masked"}):
            yield ("census", 500, batch, 5, 0, (True, True), {"subset_size": subset, **diffusion})

# ablation experiment
timesteps_list = [500]
batch_sizes = [256]
datanames = ["abalone", "adult"]  # , "magic", "bean", "default", "gesture", "letter", "shoppers"]
epochs_amounts_ablation = [1]
params_opt_ablation = [(False, False), (True, False), (False, True), (True, True)][::-1]
runs = 1
extra_args = [None]
params_ablation = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts_ablation, range(runs), params_opt_ablation, extra_args
)

# interepoch experiment
timesteps_list = [500]
batch_sizes = [32]
datanames = ["adult", "abalone", "default", "credit"]
# datanames = ["adult", "abalone"]
epochs_amounts_interepoch = [10]
params_opt_interepoch = [(True, True)]
runs = 1
extra_args_alignment = [{}, {"no_fast_align": True}, {"no_align": True}][-1:]
extra_args_pooling = [{}] #, {"perfect_pooling": True}]
extra_args_diffusion = [{}, {"diffusion_type": "masked"}]
# all combination, but if no align, then no perfect pooling
extra_args = []
for align_args in extra_args_alignment:
    for pool_args in extra_args_pooling:
        if align_args.get("no_align", False) and pool_args.get("perfect_pooling", False):
            continue
        for diffusion_args in extra_args_diffusion:
            combined = {**align_args, **pool_args, **diffusion_args}
            extra_args.append(combined)
params_interepoch = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts_interepoch, range(runs), params_opt_interepoch, extra_args
)
params_interepoch_simple = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts_interepoch, range(runs), [("simple", None)], extra_args
)

# known t experiment
timesteps_list = [500]
batch_sizes = [256]
datanames = ["adult", "abalone", "magic", "bean"]
epochs_amounts = [1]
params_opt = [(True, True)]
runs = 1
# make known t extra_args
# extra_args = [{"known_t_amount": i, "known_noise_amount": 100} for i in range(0, 257, 32)]
extra_args = [{"known_t_amount": i, "known_noise_amount": 100} for i in (0, 25, 50, 75, 100)]
params_known_t = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# known noise experiment
timesteps_list = [500]
batch_sizes = [256]
datanames = ["adult", "abalone", "magic", "bean"]
epochs_amounts = [1]
params_opt = [(True, True)]
runs = 1
# make known noise extra_args
# this is dependant on dataset feature size, so we pass a percentage
# extra_args = [{"known_noise_amount": i, "known_t_amount": batch_sizes[0]} for i in range(0, 100, 20)]
extra_args = [{"known_noise_amount": i, "known_t_amount": 100} for i in (0, 25, 50, 75, 100)]
params_known_noise = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)


# 4x4 grid search for known t and known noise
timesteps_list = [500]
batch_sizes = [256]
# datanames = ["adult" , "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers"]
datanames = ["abalone", "adult"]  # , "default", "bean"]
epochs_amounts = [1]
params_opt = [(True, True)]
runs = 2
# make known t and known noise extra_args
extra_args = []
# for known_t in range(0, batch_sizes[0]+1, batch_sizes[0]//4):
for known_t in (0, 25, 50, 75, 100):
    # for known_noise in [0, 25, 50, 75, 99, 100]:
    for known_noise in (0, 25, 50, 75, 100):
        extra_args.append({"known_t_amount": known_t, "known_noise_amount": known_noise})
params_known_tn = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# known mask experiments (masked diffusion only): --- 1D sweep of mask knowledge, t and noise fully known
_km_datanames = ["credit", "adult"]
_km_extra = [
    {
        "diffusion_type": "masked",
        "known_mask_amount": i,
        "known_t_amount": 100,
        "known_noise_amount": 100,
    }
    for i in (0, 25, 50, 75, 100)
]
params_known_mask = tqit.product(_km_datanames, [500], [256], [1], range(1), [(True, True)], _km_extra)

# known-tm: 5x5 grid t x mask (noise=0) to isolate t<->mask coupling
_ktm_extra = [
    {
        "diffusion_type": "masked",
        "known_t_amount": t,
        "known_mask_amount": m,
        "known_noise_amount": 0,
    }
    for t in (0, 25, 50, 75, 100)
    for m in (0, 25, 50, 75, 100)
]
params_known_tm = tqit.product(_km_datanames, [500], [256], [1], range(1), [(True, True)], _ktm_extra)

# known-tm-prior: same 5x5 grid paired with/without prior regularizer (delta heatmap = prior contribution)
_ktm_prior_extra = []
for _t in (0, 25, 50, 75, 100):
    for _m in (0, 25, 50, 75, 100):
        _base = {
            "diffusion_type": "masked",
            "known_t_amount": _t,
            "known_mask_amount": _m,
            "known_noise_amount": 0,
            "subset_size": 128,
        }
        _ktm_prior_extra.append(_base)
        _ktm_prior_extra.append({**_base, "no_prior": True})

params_known_tm_prior = tqit.product(
    _km_datanames, [500], [4], [1], range(1), [(True, True)], _ktm_prior_extra
)

# known-tnm: 3x3x3 cube over known_t x known_noise x known_mask for small-multiples 3D visualization
_levels = (0, 50, 100)
_ktnm_extra = [
    {
        "diffusion_type": "masked",
        "known_t_amount": t,
        "known_noise_amount": n,
        "known_mask_amount": m,
    }
    for t in _levels
    for n in _levels
    for m in _levels
]
params_known_tnm = tqit.product(_km_datanames, [500], [256], [1], range(1), [(True, True)], _ktnm_extra)

# tests different pools sizes
timesteps_list = [500]
batch_sizes = [32]
# datanames = ["adult" , "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers"]
datanames = ["adult", "default"] #, "credit", "abalone"]
#datanames = ["abalone", "credit"]
epochs_amounts = [1]
params_opt = [(True, True)]
runs = 1
# make known t and known noise extra_args
types = [{"no_align": True}, {"no_fast_align": True}, {}]
types = types
types =  types + [{"diffusion_type": "masked", **t} for t in types]  # with and without masked diffusion

# pools 5, 9 (odd only — even-size pools break median)
extra_args = [{**t, "pool_size": pool_size} for pool_size in [5, 9] for t in types]
extra_args_perfect = [{**a, "perfect_pooling": True} for a in extra_args]
params_pool_size = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)
params_pool_size_perfect = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args_perfect
)

# defenses
datanames = ["adult", "abalone", "magic", "credit", "bean", "default", "gesture", "letter", "shoppers"]
timesteps_list = [500]
batch_sizes = [1]
epochs_amounts = [1]
params_opt = [(True, True)]  # , ("simple", None)] # optimize both or do simple attack
runs = 1
_shared_args = {"subset_size": 64, "loss_fn": "auto"}
extra_args = [
    ({"defense": defense, **_shared_args}, {"defense": defense, "diffusion_type": "masked", **_shared_args})
    for defense in ["clipping", "compression", "noise", "censor"]
]
extra_args = [item for sublist in extra_args for item in sublist]  # flatten list of tuples
params_def = tqit.product(datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args)

# experiment to save models in advance
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
timesteps_list = [500]
batch_sizes = [1024]
epochs_amounts = [1000]
params_opt = [(True, True)]
runs = 1
extra_args = [{"save_model": True}]  # , {"save_model": True, "diffusion_type": "masked"}]
params_save_model = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# experiment to run GIDM baseline. always MSE loss
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
timesteps_list = [500]
batch_sizes = [1, 8][::-1]
epochs_amounts = [1]
params_opt = [(True, True)]  # , ("simple", None)] # optimize both or do simple attack
runs = 1
extra_args = [{"baseline": "gidm", "loss_fn": "mse"}]  # , {"baseline": "gidm", "diffusion_type": "masked"}]
params_gidm = tqit.product(datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args)

# experiment to simulate tableak. always use cossim loss to match their setup, and disable optimization since they do not optimize t/noise
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
timesteps_list = [500]
batch_sizes = [1, 8, 16, 32][::-1]  # 1, 8, 256 512 is old
epochs_amounts = [1]
params_opt = [(False, False)]  # no optimization, just reconstruction
runs = 1
extra_args = [
    {"baseline": "tableak", "pool_size": 8, "loss_fn": "cossim"}
]  # , {"baseline": "tableak", "pool_size": 8, "diffusion_type": "masked"}]  # big pool helps reconstruction stability
params_tableak = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

extra_args_1 = [
    {"baseline": "tableak", "pool_size": 1, "loss_fn": "cossim"}
]  # , {"baseline": "tableak", "pool_size": 1, "diffusion_type": "masked"}]  # pool size 1 if too slow
params_tableak_1 = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args_1
)

# adaptive TabLeak: same setup as tableak baseline, but ALSO optimize t + noise (params_opt=(True,True)).
# stays strictly weaker than GOLIATH: cossim loss (no table-aware loss), gaussian (no coupling prior),
# epochs=1 (no cross-epoch ensembling) -- enforced by the guard in Attack.__init__.
# NO pooling (pool_size=1), matching GOLIATH's accuracy runs -> fair head-to-head and ~8x cheaper.
# Subset capped to ~30 attacks/config (batch*30, min 60) for stable per-cell means.
_ta_datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
_ta_batch_sizes = [1, 8, 16, 32]
_TA_ATTACKS = 30
# Both diffusion families, matching the main GOLIATH tables. Masked variant also optimizes the mask
# latent (params_opt=(True,True) -> t/noise on, mask on by default), but with no_prior=True so it does
# NOT get the mask<->t coupling prior (a GOLIATH-only component; also required by the __init__ guard).
_ta_diffusion = [
    {},
    {"diffusion_type": "masked", "no_prior": True},
]


def _params_tableak_adaptive():
    for dataname, batch_sz, diffusion in tqit.product(_ta_datanames, _ta_batch_sizes, _ta_diffusion):
        subset_size = max(batch_sz * _TA_ATTACKS, 60)
        yield (
            dataname, 500, batch_sz, 1, 0, (True, True),
            {"baseline": "tableak", "pool_size": 1, "loss_fn": "cossim", "subset_size": subset_size, **diffusion},
        )

# init strategy comparison (previous vs uniform) x alignment (fast align vs no align)
datanames = ["abalone", "adult"]
timesteps_list = [500]
batch_sizes = [256]
epochs_amounts = [10]
params_opt = [(True, True)]
runs = 1
extra_args_init = [{"init_strategy": "uniform"}, {"init_strategy": "previous"}]
extra_args_align = [{}, {"no_align": True}]
extra_args = []
for init_args in extra_args_init:
    for align_args in extra_args_align:
        extra_args.append({**init_args, **align_args})
params_init_strategy = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# ablation on loss function
datanames = ["adult", "bean", "gesture", "shoppers", "magic", "credit", "abalone", "default", "letter"]
timesteps_list = [500]
batch_sizes = [16]
epochs_amounts = [2]
params_opt = [(True, True)]
runs = 1
_loss_variants = ["auto"]  # ["mse", "cossim", "both"]
_double_cossim_alphas = [0.3, 0.5, 0.7]
_diffusion_variants = [None, {"diffusion_type": "masked"}]
extra_args = [{"subset_size": 256, "loss_fn": fn, **(d or {})} for fn in _loss_variants for d in _diffusion_variants]
# ] + [
#     {"subset_size": 256, "loss_fn": "double-cossim", "double_cossim_alpha": alpha, **(d or {})}
#     for alpha in _double_cossim_alphas
#     for d in _diffusion_variants
# ]
params_ablation_loss = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# ablation on KL prior regularizer (masked diffusion only)
_ap_datanames = ["adult", "credit", "abalone", "magic", "bean", "gesture", "shoppers", "default", "letter"]
_ap_datanames = ["bean", "gesture", "shoppers", "default", "letter"]
_ap_extra_args = [
    {"subset_size": 64, "diffusion_type": "masked", "loss_fn": "auto"},
    {"subset_size": 64, "diffusion_type": "masked", "loss_fn": "auto", "no_prior": True},
]
params_ablation_prior = tqit.product(_ap_datanames, [500], [1], [1], range(1), [(True, True)], _ap_extra_args)

# scalability experiment: per-batch vs full-dataset attack across sizes
datanames = ["abalone", "adult"]
timesteps_list = [500]
batch_sizes = [1, 8, 32, 64, 128, 256, 512]
epochs_amounts = [1]
params_opt = [(True, True)]
runs = 1

extra_args = [None]  # , {"full_dataset_attack": True}]

# Use a single tqit.product to generate the progress bar and combinations
params_scalability = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)


# latent reconstruction evaluation (small-scale)
_lr_datanames = ["adult", "magic", "abalone", "credit"]
_lr_batch_subset = [(1, 64)]  # (8, 64), (256, 2048)
_lr_diffusion = [{"diffusion_type": "masked"}, None][::-1]  # ,


def _params_latent_recon():
    for dataname, (batch_sz, subset_size), diffusion, run in tqit.product(
        _lr_datanames, _lr_batch_subset, _lr_diffusion, range(2)
    ):
        extra_args = {**(diffusion or {}), "subset_size": subset_size}
        yield (dataname, 500, batch_sz, 1, run, (True, True), extra_args)


# random baseline for mask inversion (masked diffusion only)
# variants: freeze mask to random init, or freeze all latents (t, noise, mask)
datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]
timesteps_list = [500]
batch_sizes = [1, 8, 16, 32][::-1]  # old 256 512
epochs_amounts = [1]
params_opt = [(True, True)]  # keep no_t/no_noise off; random_latents drives freezing
runs = 1
# disable prior regularization term
extra_args = [
    {
        "diffusion_type": "masked",
        "random_latents": "mask",
        "disable_prior": True,
        "loss_fn": "cossim",
    },  # cossim loss to match tableak and other baselines
    {"diffusion_type": "masked", "random_latents": "all", "disable_prior": True, "loss_fn": "cossim"},
]
params_random = tqit.product(
    datanames, timesteps_list, batch_sizes, epochs_amounts, range(runs), params_opt, extra_args
)

# rebuttal metrics breakdown (2.1 per-cardinality, 2.2 per-column/sensitive-attribute, 2.3 whole-row + re-id).
# Full GOLIATH attack (loss_fn defaults to "auto"); curated datasets span high cardinality (letter=26,
# gesture), low/binary (magic), continuous-heavy (abalone), and sensitive demographics (adult).
_rm_datanames = ["adult", "letter", "gesture", "magic", "abalone"]
# GPU-conscious: batch=1 kept small (re-id trivial at N=1; larger batches give more rows per attack).
_rm_batch_subset = [(1, 32), (8, 128), (32, 256)]  # largest batch = 32, matching the paper's main table
_rm_diffusion = [None, {"diffusion_type": "masked"}]


def _params_rebuttal_metrics():
    for dataname, (batch_sz, subset_size), diffusion in tqit.product(
        _rm_datanames, _rm_batch_subset, _rm_diffusion
    ):
        extra_args = {**(diffusion or {}), "subset_size": subset_size}
        yield (dataname, 500, batch_sz, 1, 0, (True, True), extra_args)


# architecture generalization (PRIMARY weakness per meta-review): GOLIATH on an FT-Transformer backbone
# instead of the MLP, to show the attack is not MLP-specific. GOLIATH attack (loss auto), both diffusion,
# ALL 9 datasets (batch sizes 8/32 as representative; epochs=1 single-epoch generalization demo).
_ft_datanames = ["adult", "abalone", "magic", "bean", "default", "gesture", "letter", "shoppers", "credit"]


def _params_ft_transformer():
    # 5 epochs (multi-epoch, matching GOLIATH's ensembling). Subset = batch*4 (4 batches/epoch) keeps
    # the 9-dataset x 5-epoch transformer sweep tractable; the transformer is ~3x slower than the MLP.
    for dataname, batch_sz, diffusion in tqit.product(
        _ft_datanames, [8, 32], [None, {"diffusion_type": "masked"}]
    ):
        extra = {"backbone": "ft-transformer", "subset_size": batch_sz * 4, **(diffusion or {})}
        yield (dataname, 500, batch_sz, 5, 0, (True, True), extra)


# Architecture generalization, v2. Supersedes `ft-transformer`, whose numbers predate two fixes:
#   (a) model: the feature tokenizer's bias was init'd to zeros, so every zero-valued feature
#       produced an identical token -- fatal on sparse one-hot data. It is now random per feature
#       (src/model.py FTTransformerDiffusion).
#   (b) attack: non-MLP backbones were silently downgraded from `double-cossim` to `both`. The
#       split is now backbone-aware and, for the transformer, keys on the feature tokenizer block
#       (src/attack.py _build_cont_block_mask_tokenizer).
# Full grid so the claim can be made across the board: 9 datasets x batch {1,8,16,32} x
# {gaussian, masked} x {auto, both}. `auto` is the primary schema/backbone-aware attack;
# explicit `both` records the legacy global-loss ablation. Subset = batch*4 matches the original
# sweep (floored at 32 so batch 1 still attacks enough rows to be stable).
def _params_ft_transformer_v2():
    for dataname, batch_sz, diffusion, loss in tqit.product(
        _ft_datanames, [1, 8, 16, 32], [None, {"diffusion_type": "masked"}], ["auto", "both"],
    ):
        extra = {
            "backbone": "ft-transformer",
            "subset_size": max(batch_sz * 4, 32),
            "loss_fn": loss,
            # Adult screening found clear over-optimization at larger batches: the
            # lowest-residual iterate was often less identifiable. These fixed, disclosed
            # budgets maximize the reported five-epoch endpoint without changing the objective.
            "optim_iters": {1: 2000, 8: 2000, 16: 1000, 32: 1000}[batch_sz],
            **(diffusion or {}),
        }
        yield (dataname, 500, batch_sz, 5, 0, (True, True), extra)


# DIAGNOSTIC: the first ft-transformer run used the MLP's attack hyperparameters (lr=0.06, iters=2000)
# and did not converge (flat trajectory, below-random, continuous ~0). Sweep attack_lr x optim_iters on
# the two best datasets (single-epoch, gaussian) to test whether it's an under-tuning artifact.
def _params_ft_transformer_tune():
    for dataname, lr, iters in tqit.product(["credit", "abalone"], [0.02, 0.06, 0.2], [8000]):
        yield (
            dataname, 500, 32, 1, 0, (True, True),
            {"backbone": "ft-transformer", "subset_size": 128, "attack_lr": lr, "optim_iters": iters},
        )


# reshuffling robustness (3.1 / 6bVX-Q3, esrr-Q3, 8B8y): 4 variants show the full story —
#   (1) fixed order + per-slot ensemble  (baseline: ensembling works)
#   (2) reshuffle  + per-slot ensemble   (the problem: slot-based blending mixes rows)
#   (3) reshuffle  + GLOBAL ensemble      (the fix: order-invariant pool matching, greedy)
#   (4) reshuffle  + GLOBAL + Hungarian   (large-N pool matching: greedy vs optimal)
# The global ensemble matches each batch against the full 320-row pool, so (3) vs (4) also exposes the
# greedy-vs-Hungarian alignment difference at large N. 5 epochs -> visible per-epoch trajectory.
_rs_datanames = ["adult", "abalone"]
_rs_variants = [
    {},
    {"reshuffle": True},
    {"reshuffle": True, "global_ensemble": True},
    {"reshuffle": True, "global_ensemble": True, "no_fast_align": True},
]


def _params_reshuffle():
    for dataname in _rs_datanames:
        for diffusion in [None, {"diffusion_type": "masked"}]:
            for v in _rs_variants:
                yield (dataname, 500, 32, 5, 0, (True, True), {"subset_size": 320, **v, **(diffusion or {})})


# certified DP-SGD defense (8B8y-Q1): per-example clip + calibrated Gaussian noise, GOLIATH attack.
# We sweep the NOISE MULTIPLIER z directly (mapped to single-step epsilon by the accountant) to capture
# the reconstruction-vs-privacy TRANSITION. Fixing epsilon in {1,8} put every point past the transition
# (single-step DP with calibrated noise already destroys the one observed gradient), so the curve was
# flat at the alignment floor. Sweeping z from near-0 (no effect) to strong noise reveals the tradeoff.
# Datasets chosen for dynamic range (adult/credit drop clearly; abalone kept for contrast — its high
# alignment floor makes the attack barely beat random even undefended).
_dp_datanames = ["adult", "credit", "abalone"]
# Transition is sharp and at low z (attack uses a single gradient), so resolve the low end finely.
# z=0 = clipping-only anchor (eps=inf); z=0.69 -> eps=6.77 single-step (exact analytic accountant;
# the old RDP accountant reported 8.0 for the same z — recompute eps from the logged
# dp_noise_multiplier at report time, never reuse a CSV's dp_achieved_epsilon across accountants).
_dp_zs = [0.0, 0.003, 0.01, 0.03, 0.07, 0.15, 0.3, 0.69]


def _params_dp_defense():
    for dataname, z in tqit.product(_dp_datanames, _dp_zs):
        yield (
            dataname, 500, 256, 1, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": 1.0, "dp_steps": 1,
             "loss_fn": "auto", "subset_size": 1024},
        )


# Batch-1 DP sweep, matching the batch size and subset used by the paper's other defense
# experiments (params_def: batch 1, subset 64) so the certified-DP numbers are directly
# comparable to the clipping / compression / noise / CENSOR results. Batch 1 is also the
# attacker's strongest setting, so defeating it there is the stronger claim.
def _params_dp_defense_b1():
    for dataname, z in tqit.product(_dp_datanames, _dp_zs):
        yield (
            dataname, 500, 1, 1, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": 1.0, "dp_steps": 1,
             "loss_fn": "auto", "subset_size": 64},
        )


# Epsilon-targeted DP sweep (8B8y-Q1). The z-space sweep above resolves WHERE the attack breaks,
# but it lands almost entirely in a vacuous regime: z=0.003..0.3 maps to eps=6.1e4..21.6, and only
# the last point (z=0.69) reaches eps=8. A reviewer asking for a "calibrated (eps,delta)-DP defense"
# wants points in the standard eps range, so we sweep in EPSILON space instead and let the
# accountant pick z. eps=10 was never sampled by the z-grid at all.
# z values are get_noise_multiplier(eps, delta=1e-5, steps=1) under the EXACT analytic Gaussian
# accountant (Balle-Wang; less noise than the RDP-calibrated values previously used: e.g. eps=1
# needs z=3.73, not 4.90); hardcoded so the config is static and reproducible without importing
# the accountant at module import time. results/dp-defense-eps/ was produced with the OLD
# RDP-calibrated z values — its rows' true exact epsilons are lower than the targets in this list.
_dp_eps_targets = [(10.0, 0.4999), (4.0, 1.0812), (2.0, 1.9938), (1.0, 3.7306), (0.5, 7.0318)]
# abalone is excluded: its alignment floor makes the attack barely beat random even undefended,
# so the defense curve is degenerate there and it is not reported.
_dp_eps_datanames = ["adult", "credit"]


def _params_dp_defense_eps():
    for dataname, (_eps, z) in tqit.product(_dp_eps_datanames, _dp_eps_targets):
        yield (
            dataname, 500, 1, 1, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": 1.0, "dp_steps": 1,
             "loss_fn": "auto", "subset_size": 64},
        )


# Utility side of the DP privacy-utility curve (8B8y-Q1). The dp-defense sweeps only ever expose the
# defended gradient to the ATTACKER; the client keeps stepping on clean grads, so they measure privacy
# with no utility cost and "DP defeats the attack at noise z" is close to tautological. Here the client
# trains on its OWN defended gradients (--defended-training) with no attack, and we log per-epoch
# train_loss. Untrained loss is ~1.0 (MSE against unit-variance noise), so a run stuck near 1.0 learned
# nothing.
#
# The two regimes mirror the two attack sweeps exactly, so each utility number pairs with a privacy
# number from the identical configuration: (batch 1, subset 64) = dp-defense-b1, and
# (batch 256, subset 1024) = dp-defense. Epoch counts are chosen so both regimes take a comparable
# number of optimizer steps (~1200-1300).
#
# dp_steps accounting convention: the ATTACK sweeps use dp_steps=1 — the adversary observes a single
# released gradient, so the single-release epsilon is the right threat-model quantity. The UTILITY
# runs train on epochs * ceil(subset/batch) defended releases, so their honest training-level
# epsilon composes over all of them: dp_steps = 1280 (b1) / 1200 (b256), q=1 RDP composition.
# (Older dp-utility CSVs logged the steps=1 epsilon — recompute from dp_noise_multiplier at report
# time.) z is given directly, so dp_steps only affects the logged epsilon, not training behaviour.
_dp_util_zs = [0.0, 0.003, 0.01, 0.03, 0.07, 0.69]
_dp_util_regimes = [(1, 64, 20), (256, 1024, 300)]  # (batch, subset, epochs)


def _params_dp_utility():
    for dataname, (batch, subset, epochs), z in tqit.product(
        _dp_eps_datanames, _dp_util_regimes, _dp_util_zs
    ):
        steps = epochs * ((subset + batch - 1) // batch)  # releases actually consumed by training
        yield (
            dataname, 500, batch, epochs, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": 1.0, "dp_steps": steps,
             "subset_size": subset, "defended_training": True, "attack": False,
             "skip_eval": True},
        )


# Calibrated-clip DP sweeps (rebuttal 8B8y-Q1 follow-up). scripts/dp_diagnostics.py measured the
# per-example gradient L2 norms at the released step (random-init model, the attack threat model):
# batch-1 medians are adult 0.278, credit 0.370, abalone 0.905 (3 seeds, subset 64). The original
# C=1.0 therefore NEVER clips adult/credit (frac_clipped=0), so the noise z*C was ~3x oversized
# relative to the actual signal for a given epsilon. Standard practice (Abadi et al. 2016, Sec. 3)
# sets C ~= the median per-example norm; the spread across datasets is 3.3x, so C is per-dataset
# (median rounded to 1 significant figure). Epsilon per z is C-invariant — recalibrating C moves
# only the attack/utility transition, not the accounting.
_dp_clip_cal = {"adult": 0.3, "credit": 0.4, "abalone": 0.9}


def _params_dp_defense_b1_c():
    for dataname, z in tqit.product(_dp_datanames, _dp_zs):
        yield (
            dataname, 500, 1, 1, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": _dp_clip_cal[dataname],
             "dp_steps": 1, "loss_fn": "auto", "subset_size": 64},
        )


def _params_dp_defense_eps_c():
    for dataname, (_eps, z) in tqit.product(_dp_eps_datanames, _dp_eps_targets):
        yield (
            dataname, 500, 1, 1, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": _dp_clip_cal[dataname],
             "dp_steps": 1, "loss_fn": "auto", "subset_size": 64},
        )


def _params_dp_utility_c():
    for dataname, (batch, subset, epochs), z in tqit.product(
        _dp_eps_datanames, _dp_util_regimes, _dp_util_zs
    ):
        steps = epochs * ((subset + batch - 1) // batch)
        yield (
            dataname, 500, batch, epochs, 0, (True, True),
            {"defense": "dp", "dp_noise_multiplier": z, "dp_clip": _dp_clip_cal[dataname],
             "dp_steps": steps, "subset_size": subset, "defended_training": True,
             "attack": False, "skip_eval": True},
        )


# alignment-bias diagnostic (2.4 / 8B8y-Q4): greedy fast-align vs Hungarian on the SAME
# reconstruction. Larger batches (bigger N) stress the greedy-vs-optimal assignment gap most.
_ad_datanames = ["adult", "abalone"]
_ad_batch_subset = [(8, 64), (32, 128), (128, 256), (256, 512)]


def _params_align_diagnostic():
    for dataname, (batch_sz, subset_size) in tqit.product(_ad_datanames, _ad_batch_subset):
        yield (dataname, 500, batch_sz, 1, 0, (True, True), {"subset_size": subset_size, "align_diagnostic": True})


# per-client FedAvg weight-update inversion (rebuttal, per-client FedAvg setting, AC-approved scope).
# Very small subset (32/64), full-batch local steps (batch = subset), sweep the number of local steps K.
# K=1 is the single-gradient (FedSGD) reference; K>1 is the multi-step FedAvg update.
def _params_fedavg():
    for dataname in ["adult", "abalone"]:
        for subset in [32, 64]:
            for K in [1, 2, 4, 8]:
                yield (
                    dataname, 500, subset, 1, 0, (True, True),
                    {"fedavg": True, "fedavg_local_steps": K, "subset_size": subset},
                )


# Masked-diffusion counterpart of `fedavg` (6bVX Q2). The original sweep never set diffusion_type,
# so all 12 of its cells inherited the gaussian default, making it the only rebuttal experiment that
# does not cover both types. Kept as a separate experiment rather than folding masked into
# `_params_fedavg` so the completed gaussian runs are not repeated; results land in
# results/fedavg-masked/ and are read alongside results/fedavg/.
def _params_fedavg_masked():
    for dataname in ["adult", "abalone"]:
        for subset in [32, 64]:
            for K in [1, 2, 4, 8]:
                yield (
                    dataname, 500, subset, 1, 0, (True, True),
                    {"fedavg": True, "fedavg_local_steps": K, "subset_size": subset,
                     "diffusion_type": "masked"},
                )


params = {
    "accuracy": params_acc,
    "accuracy-epochX": params_acc_epochX,
    "accuracy-epochX-census": params_acc_epochX_census,
    "accuracy-epochX-census-b18": _params_acc_epochX_census_b18(),
    "ablation-tn": params_ablation,
    "interepoch": params_interepoch,
    "known-t": params_known_t,
    "known-noise": params_known_noise,
    "known-tn": params_known_tn,
    "known-mask": params_known_mask,
    "known-tm": params_known_tm,
    "known-tm-prior": params_known_tm_prior,
    "known-tnm": params_known_tnm,
    "pool-size": params_pool_size,
    "pool-size-perfect": params_pool_size_perfect,
    "defense": params_def,
    "save-model": params_save_model,
    "gidm-baseline": params_gidm,
    "tableak-baseline": params_tableak,
    "tableak-baseline-1": params_tableak_1,
    "tableak-adaptive": _params_tableak_adaptive(),
    "interepoch-simple": params_interepoch_simple,
    "init-strategy": params_init_strategy,
    "ablation-loss": params_ablation_loss,
    "ablation-prior": params_ablation_prior,
    "scalability": params_scalability,
    "latent-reconstruction": _params_latent_recon(),
    "random-baseline": params_random,
    "rebuttal-metrics": _params_rebuttal_metrics(),
    "align-diagnostic": _params_align_diagnostic(),
    "dp-defense": _params_dp_defense(),
    "dp-defense-b1": _params_dp_defense_b1(),
    "dp-defense-eps": _params_dp_defense_eps(),
    "dp-utility": _params_dp_utility(),
    "dp-defense-b1-c": _params_dp_defense_b1_c(),
    "dp-defense-eps-c": _params_dp_defense_eps_c(),
    "dp-utility-c": _params_dp_utility_c(),
    "reshuffle": _params_reshuffle(),
    # "ft-transformer" deregistered: superseded by ft-transformer-v2, which is a strict
    # superset of the grid (batch 1/8/16/32 vs 8/32) and identical in model config now that
    # main.py defaults to --ft-d-model 256 --ft-nlayers 4. Re-running the old name would have
    # written v2-quality rows into results/ft-transformer/, which v1 rows cannot be told apart
    # from (they predate ft_d_model logging). Generator kept below for reference.
    "ft-transformer-v2": _params_ft_transformer_v2(),
    "ft-transformer-tune": _params_ft_transformer_tune(),
    "fedavg": _params_fedavg(),
    "fedavg-masked": _params_fedavg_masked(),
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        param_set = params.get(experiment_name, None)
        if param_set is None:
            print(f"Experiment {experiment_name} not found.")
        else:
            # print all parameter combinations for the given experiment
            for param in param_set:
                print(param)
    else:
        for experiment, param_set in params.items():
            print(f"Experiment: {experiment}, #configs: {len(list(param_set))}")
