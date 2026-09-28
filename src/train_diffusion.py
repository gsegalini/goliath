import time
import warnings
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import PerformanceManager, Preprocessor, Matcher
from .model import (
    MLPDiffusion,
    FTTransformerDiffusion,
    build_diffusion_model,
    diffusion_model_config,
)
from .diffusion_process import MaskedDiffusionProcess, create_diffusion_process
from src.attack import Attack
from syntheval import SynthEval
from .download_and_process_datasets import CATCOLS
from .metrics import TableakSimilarity
from collections import OrderedDict
from copy import deepcopy
from .defense import additive_noise, gradient_clipping, gradient_compression, orthogonal_gradient, dp_sgd_gradient
from .dp_accountant import get_noise_multiplier, get_epsilon


def _normalize_attack_latent_args(args):
    """Resolve the simplified attack without exposing an unrequested masked latent."""
    if args.simple_attack:
        args.known_t_amount = 100
        args.known_noise_amount = 100
        args.known_mask_amount = 100
        return

    args.known_mask_amount = getattr(args, "known_mask_amount", 0)
    args.simple_attack = (
        args.known_t_amount >= 100
        and args.known_noise_amount >= 100
        and (
            args.diffusion_type != "masked"
            or args.known_mask_amount >= 100
        )
    )


def _log_common_metrics(pm, args, attack, epoch, full_dataset_attack):
    """Log the standard set of experiment configuration metrics."""
    pm.log_metric("dataname", args.dataname)
    pm.log_metric("seed", args.seed)
    pm.log_metric("diffusion_type", args.diffusion_type)
    pm.log_metric("backbone", getattr(args, "backbone", "mlp"))
    if getattr(args, "backbone", "mlp") == "ft-transformer":
        pm.log_metric("ft_d_model", getattr(args, "ft_d_model", 256))
        pm.log_metric("ft_nhead", getattr(args, "ft_nhead", 8))
        pm.log_metric("ft_nlayers", getattr(args, "ft_nlayers", 4))
        pm.log_metric("ft_dim_feedforward", getattr(args, "ft_dim_feedforward", 512))
        pm.log_metric("ft_init_std", getattr(args, "ft_init_std", 1.0))
        pm.log_metric("ft_head_init_std", getattr(args, "ft_head_init_std", 0.01))
        pm.log_metric("ft_lr", getattr(args, "ft_lr", 0.0001))
        pm.log_metric("ft_analytic_weight", getattr(args, "ft_analytic_weight", 0.0))
        pm.log_metric("ft_analytic_init", getattr(args, "ft_analytic_init", False))
    pm.log_metric("optim_iters", args.optim_iters)
    pm.log_metric("model_lr", getattr(args, "_model_lr", args.lr))
    pm.log_metric("pool_size", args.pool_size)
    pm.log_metric("simple_attack", args.simple_attack)
    pm.log_metric("attack_lr", args.attack_lr)
    pm.log_metric("batch_size", args.batch_sz)
    pm.log_metric("known_t_amount", args.known_t_amount)
    pm.log_metric("known_noise_amount", args.known_noise_amount)
    pm.log_metric("known_mask_amount", getattr(args, "known_mask_amount", 0))
    pm.log_metric("epoch", epoch)
    pm.log_metric("optimize_t", attack.diffusion_process.get_optimizable_flags(args).get("t", False))
    pm.log_metric("optimize_noise", attack.diffusion_process.get_optimizable_flags(args).get("noise", False))
    pm.log_metric("optimize_mask", attack.diffusion_process.get_optimizable_flags(args).get("mask", False))
    pm.log_metric("epochs", args.epochs)
    pm.log_metric("timesteps", args.timesteps)
    pm.log_metric("defense", args.defense)
    if args.defense == "dp":
        pm.log_metric("dp_clip", getattr(args, "dp_clip", None))
        pm.log_metric("dp_epsilon", getattr(args, "dp_epsilon", None))
        pm.log_metric("dp_delta", getattr(args, "dp_delta", None))
        pm.log_metric("dp_steps", getattr(args, "dp_steps", None))
        pm.log_metric("dp_noise_multiplier", getattr(args, "_dp_noise_multiplier", None))
        pm.log_metric("dp_achieved_epsilon", getattr(args, "_dp_achieved_epsilon", None))
    pm.log_metric("baseline", args.baseline)
    pm.log_metric("random_latents", getattr(args, "random_latents", "") or "")
    pm.log_metric("fast_align", not args.no_fast_align and not args.no_align)
    pm.log_metric("align", not args.no_align)
    pm.log_metric("reshuffle", getattr(args, "reshuffle", False))
    pm.log_metric("global_ensemble", getattr(args, "global_ensemble", False))
    pm.log_metric("fedavg", getattr(args, "fedavg", False))
    pm.log_metric("perfect_pooling", args.perfect_pooling)
    pm.log_metric("init_strategy", getattr(args, "init_strategy", None))
    pm.log_metric("warm_data", getattr(args, "warm_data", False))
    pm.log_metric("warm_latents", getattr(args, "warm_latents", True))
    pm.log_metric("t_init", getattr(args, "t_init", "random"))
    pm.log_metric("noise_lr_mult", getattr(attack.diffusion_process, "noise_lr_mult", None))
    pm.log_metric("loss_fn", getattr(args, "loss_fn", "cossim"))
    # `loss_fn` is the request; `auto` and the backbone guard can resolve it to something else.
    pm.log_metric("effective_loss_fn", getattr(attack, "_effective_loss_fn", ""))
    pm.log_metric("full_dataset_attack", full_dataset_attack)
    pm.log_metric("anchor", getattr(args, "anchor", "first"))
    pm.log_metric("prior_peak", getattr(args, "prior_peak", None))
    pm.log_metric("prior_cooldown_start", getattr(args, "prior_cooldown_start", None))
    pm.log_metric("no_prior", getattr(args, "no_prior", False))
    pm.log_metric("double_cossim_alpha", getattr(args, "double_cossim_alpha", None))


def _resolve_dp_noise(args):
    """Resolve the DP-SGD noise multiplier z from either an explicit value or a target (eps, delta).

    Stores the achieved epsilon on args for logging. Called once per run when defense == 'dp'.
    """
    if getattr(args, "dp_noise_multiplier", None) is not None:
        z = float(args.dp_noise_multiplier)
    elif getattr(args, "dp_epsilon", None) is not None:
        z = get_noise_multiplier(float(args.dp_epsilon), float(args.dp_delta), int(args.dp_steps))
    else:
        raise ValueError("defense='dp' requires --dp-epsilon or --dp-noise-multiplier")
    args._dp_noise_multiplier = z
    # z=0 is the clipping-only anchor (no noise -> no privacy): epsilon is infinite.
    # get_epsilon is the exact analytic Gaussian mechanism at dp_steps=1 (single released
    # gradient, the attack threat model) and RDP composition for multi-release training runs.
    args._dp_achieved_epsilon = (
        float("inf") if z <= 0 else get_epsilon(float(args.dp_delta), z, int(args.dp_steps))
    )
    return z


def _apply_defense(args, grads, model, data, latents, diffusion_process, device):
    """Apply the configured gradient defense mechanism."""
    if args.defense == "clipping":
        return gradient_clipping(grads)
    elif args.defense == "compression":
        return gradient_compression(grads)
    elif args.defense == "noise":
        return additive_noise(model, grads)
    elif args.defense == "censor":
        grads, _ = orthogonal_gradient(args, grads, model, data, latents, diffusion_process, trials=20, device=device)
        return grads
    elif args.defense == "dp":
        return dp_sgd_gradient(args, model, data, latents, diffusion_process, device)
    return grads


def _evaluate_reconstruction(
    args, reconstruction, ground_truth, prepper, tableak_evaluator, performance_manager,
    suffix="",
):
    """Destandardize, decode, compute SynthEval metrics, and log TableLeak scores.

    When suffix is non-empty, log only tableak_score{suffix} (skip SynthEval DCR/NNDR
    to keep the diagnostic path cheap).
    """
    inverted = prepper.destandardize_np(reconstruction)
    inverted = prepper.decodeNpToDf(inverted)
    original = prepper.decodeNpToDf(prepper.destandardize_np(ground_truth.cpu().numpy()))

    if not args.skip_eval:
        if not suffix:
            evaluator = SynthEval(
                real_dataframe=original, cat_cols=CATCOLS[args.dataname], verbose=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message="Degrees of freedom",
                )
                results = evaluator.evaluate(inverted, dcr={}, nndr={})
            assert results is not None, "SynthEval evaluation failed"
            nndr = results[results["metric"] == "avg_nndr"]["val"].values[0]
            dcr = results[results["metric"] == "median_DCR"]["val"].values[0]

        matcher = Matcher(prepper.cat_cols, prepper.num_cols)
        matcher.update_numeric_std(prepper.numeric_std)
        _, new_indices = matcher.row_matching(original, inverted)

        # Alignment-bias diagnostic (2.4 / 8B8y-Q4): align the SAME reconstruction with the
        # greedy fast-align heuristic as well as optimal Hungarian, and record the score gap
        # plus the fraction of rows the two assign differently. Runs once per inversion (the
        # expensive part is the inversion itself, not the assignment).
        greedy_score = None
        align_disagreement = None
        if getattr(args, "align_diagnostic", False) and not suffix:
            _, greedy_indices = matcher.greedy_matching(original, inverted)
            greedy_score, _, _ = tableak_evaluator.tableak_similarity_linear(
                inverted.iloc[greedy_indices], original
            )
            align_disagreement = float(np.mean(np.asarray(greedy_indices) != np.asarray(new_indices)))

        inverted = inverted.iloc[new_indices]
        tableak_score, cat_score, cont_score = tableak_evaluator.tableak_similarity_linear(inverted, original)

        if performance_manager is not None:
            performance_manager.log_metric(f"tableak_score{suffix}", tableak_score)
            if greedy_score is not None:
                performance_manager.log_metric("tableak_score_greedy", greedy_score)
                performance_manager.log_metric("tableak_score_hungarian", tableak_score)
                performance_manager.log_metric("align_disagreement", align_disagreement)
            if not suffix:
                performance_manager.log_metric("tableak_cat_score", cat_score)
                performance_manager.log_metric("tableak_cont_score", cont_score)
                performance_manager.log_metric("dcr", dcr)
                performance_manager.log_metric("nndr", nndr)

                # Rebuttal breakdowns (2.1 per-cardinality, 2.2 per-column, 2.3 whole-row + re-id).
                # Computed on the aligned (inverted, original) pair. Per-dataset CSVs keep column
                # sets consistent, so per-column scores go into the main CSV with a `pcol_` prefix.
                detail = tableak_evaluator.detailed_similarity(inverted, original)
                performance_manager.log_metric("tableak_cat_high_score", detail["cat_high_score"])
                performance_manager.log_metric("tableak_cat_medium_score", detail["cat_medium_score"])
                performance_manager.log_metric("tableak_cat_low_score", detail["cat_low_score"])
                performance_manager.log_metric("whole_row_acc", detail["whole_row_acc"])
                performance_manager.log_metric("near_row_80", detail["near_row_80"])
                performance_manager.log_metric("near_row_90", detail["near_row_90"])
                performance_manager.log_metric("reid_rate", tableak_evaluator.reidentification_rate(inverted, original))
                for col, acc in detail["per_column"].items():
                    performance_manager.log_metric(f"pcol_{col}", acc)
    else:
        new_indices = list(range(len(inverted)))  # indices unused when skip_eval is on
    return new_indices


def main(args):
    if args.seed == -1:
        args.seed = np.random.randint(0, 1 << 32)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _normalize_attack_latent_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_sample = None if args.experiment is not None else 42  # ensure deterministic sampling for non-experiment runs
    prepper = Preprocessor(args.dataname, args.subset_size, random_state=random_sample)
    n_cats_info = prepper.OneHotEncoder.categories_
    n_cats_info = [len(cat) for cat in n_cats_info]
    cont_info = len(prepper.num_cols)
    features_info = {"cont": cont_info, "cat": n_cats_info}
    diffusion_process = create_diffusion_process(args, device, feat_info=features_info)
    if args.defense == "dp":
        z = _resolve_dp_noise(args)
        if args.verbose:
            tqdm.write(
                f"DP-SGD: clip={getattr(args, 'dp_clip', 1.0)}, noise_multiplier z={z:.4f}, "
                f"target_eps={getattr(args, 'dp_epsilon', None)}, achieved_eps={args._dp_achieved_epsilon:.3f} "
                f"(delta={args.dp_delta}, steps={args.dp_steps})"
            )
    train_df = prepper.df_train
    train_data_encoded = prepper.encodeDfToNp(train_df)
    train_data_standardized = prepper.standardize_np(train_data_encoded)
    x_train = torch.tensor(train_data_standardized, dtype=torch.float32)
    attack = Attack(args, prepper, diffusion_process)
    d_in, d_out = diffusion_process.get_model_dims(train_data_standardized.shape[1])
    if getattr(args, "backbone", "mlp") == "ft-transformer":
        model = FTTransformerDiffusion(
            d_in=d_in,
            d_out=d_out,
            d_model=getattr(args, "ft_d_model", 256),
            nhead=getattr(args, "ft_nhead", 8),
            nlayers=getattr(args, "ft_nlayers", 4),
            dim_feedforward=getattr(args, "ft_dim_feedforward", 512),
            feat_init_std=getattr(args, "ft_init_std", 1.0),
            head_init_std=getattr(args, "ft_head_init_std", 0.01),
        ).to(device)
    else:
        model = MLPDiffusion(d_in=d_in, d_out=d_out).to(device)

    performance_manager = PerformanceManager()

    tableak_evaluator = TableakSimilarity(prepper)

    # shuffle=True reshuffles minibatch composition every epoch, so batch slot i holds different
    # rows each round -- this is what breaks the cross-epoch ensemble's slot-based blending (3.1).
    # The per-batch (single-epoch) attack is unaffected; only the multi-epoch enhancement assumes
    # fixed order.
    train_loader = DataLoader(
        x_train,
        batch_size=args.batch_sz,
        shuffle=getattr(args, "reshuffle", False),
        pin_memory=device.type == "cuda",
    )

    model_lr = (
        getattr(args, "ft_lr", 0.0001)
        if getattr(args, "backbone", "mlp") == "ft-transformer"
        else args.lr
    )
    args._model_lr = model_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr, weight_decay=0)

    model.train()

    # attack mode: no training progress bar
    pbar = tqdm(range(args.epochs), desc="Training") if not args.attack else range(args.epochs)
    for epoch in pbar:

        total_loss = 0.0
        n_batches = 0
        collected_batches = [] if args.full_dataset_attack else None

        for batch in train_loader:
            optimizer.zero_grad()

            batch = batch.to(device)
            latents = diffusion_process.sample_latents(batch, device)
            batch_noised = diffusion_process.forward_diffuse(batch, latents)
            loss = diffusion_process.compute_training_loss(model, batch_noised, latents)
            loss.backward()
            total_loss += loss.item()
            n_batches += 1

            if args.attack and args.full_dataset_attack:
                # Collect gradients and model snapshot for full-dataset attack
                model_snapshot = deepcopy(model.state_dict())
                grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]

                grads = _apply_defense(args, grads, model, batch, latents, diffusion_process, batch.device)
                known_latents = diffusion_process.mask_known_latents(latents, args, batch.shape[0], device)

                collected_batches.append(
                    {
                        "grads": grads,
                        "model_state": model_snapshot,
                        "shape": batch.shape,
                        "known_latents": known_latents,
                        "ground_truth": batch.clone(),
                        "latents": {k: v.detach().clone() for k, v in latents.items()},
                    }
                )

            elif args.attack:
                # Per-batch attack (original flow)
                _log_common_metrics(performance_manager, args, attack, epoch, full_dataset_attack=False)

                if getattr(args, "fedavg", False):
                    evaluate_attack_fedavg(
                        args, attack, diffusion_process, batch, latents, model,
                        prepper, n_cats_info, tableak_evaluator, performance_manager, device,
                    )
                else:
                    evaluate_attack(
                        args,
                        attack,
                        diffusion_process,
                        batch.shape,
                        batch,
                        latents,
                        model,
                        prepper,
                        n_cats_info,
                        tableak_evaluator,
                        performance_manager,
                    )

            if args.attack and not args.full_dataset_attack:
                performance_manager.commit_row(verbose=args.verbose)
                performance_manager.save(os.path.join(args.save_dir, f"performance_{args.dataname}.csv"), append=True)

            if getattr(args, "defended_training", False) and args.defense:
                # Pay the defense's utility cost: the client STEPS on the same defended gradient it
                # would release. Without this flag `_apply_defense` only builds the attacker's view
                # and `p.grad` is left untouched, so every defense looks free (see TODO.md).
                # Note: with --attack this re-draws the DP noise, so the released and stepped-on
                # gradients differ. Intended for utility runs (--no-attack), which is what the
                # `dp-utility` experiment uses.
                defended = _apply_defense(
                    args,
                    [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()],
                    model, batch, latents, diffusion_process, batch.device,
                )
                with torch.no_grad():
                    for p, g in zip(model.parameters(), defended):
                        if p.grad is not None and g is not None:
                            p.grad.copy_(g)

            optimizer.step()
            del batch

        # Full-dataset attack: run joint reconstruction after all batches collected
        if args.attack and args.full_dataset_attack and collected_batches:
            reconstructions, attack_metrics = attack.inversion_full_dataset(collected_batches, model, n_cats_info)

            # Evaluate each batch's reconstruction
            for i, (recon, batch_info) in enumerate(zip(reconstructions, collected_batches)):
                _log_common_metrics(performance_manager, args, attack, epoch, full_dataset_attack=True)

                # Log timing/vram metrics (shared across all batches)
                performance_manager.log_metric("reconstruction_time", attack_metrics["reconstruction_time"])
                performance_manager.log_metric("time_optimization_s", attack_metrics["time_optimization_s"])
                performance_manager.log_metric("time_pooling_s", attack_metrics["time_pooling_s"])
                performance_manager.log_metric("time_alignment_s", attack_metrics["time_alignment_s"])
                performance_manager.log_metric("estimated_vram_mb", attack_metrics["estimated_vram_mb"])

                indices = _evaluate_reconstruction(
                    args, recon, batch_info["ground_truth"], prepper, tableak_evaluator, performance_manager
                )

                all_final_latents = attack_metrics.get("all_final_latents")
                if all_final_latents is not None and i < len(all_final_latents):
                    latent_metrics = diffusion_process.evaluate_latent_reconstruction(
                        all_final_latents[i], batch_info["latents"], indices=indices
                    )
                    for name, value in latent_metrics.items():
                        performance_manager.log_metric(name, value)

                performance_manager.commit_row(verbose=args.verbose)
                performance_manager.save(os.path.join(args.save_dir, f"performance_{args.dataname}.csv"), append=True)

            del collected_batches
            del reconstructions
            torch.cuda.empty_cache()

        if getattr(args, "defended_training", False) and not args.attack:
            # Utility side of the privacy-utility curve: how well can the client still fit its own
            # data when it trains on its own defended gradients? One row per epoch.
            performance_manager.log_metric("dataname", args.dataname)
            performance_manager.log_metric("batch_size", args.batch_sz)
            performance_manager.log_metric("epoch", epoch)
            performance_manager.log_metric("epochs", args.epochs)
            performance_manager.log_metric("timesteps", args.timesteps)
            performance_manager.log_metric("diffusion_type", args.diffusion_type)
            performance_manager.log_metric("defense", args.defense)
            performance_manager.log_metric("defended_training", True)
            performance_manager.log_metric("dp_clip", getattr(args, "dp_clip", None))
            performance_manager.log_metric("dp_noise_multiplier", getattr(args, "_dp_noise_multiplier", None))
            performance_manager.log_metric("dp_achieved_epsilon", getattr(args, "_dp_achieved_epsilon", None))
            performance_manager.log_metric("train_loss", total_loss / max(n_batches, 1))
            performance_manager.commit_row(verbose=args.verbose)
            performance_manager.save(
                os.path.join(args.save_dir, f"performance_{args.dataname}.csv"), append=True
            )

        if args.attack:
            attack.new_epoch()
        if not args.attack:
            pbar.set_postfix(loss=total_loss)
        if args.baseline in ["gidm", "tableak"]:  # GIDM or Tableak would only attack a single epoch
            break

    if args.save_model:
        if not os.path.exists(f"saved_models/{args.dataname}/"):
            os.makedirs(f"saved_models/{args.dataname}/")
        backbone = getattr(args, "backbone", "mlp")
        torch.save(
            {
                "model_config": diffusion_model_config(model),
                "state_dict": model.state_dict(),
            },
            f"saved_models/{args.dataname}/{args.diffusion_type}_{backbone}_model_pretrained.pt",
        )
    elif args.attack:
        performance_manager.save(os.path.join(args.save_dir, f"performance_{args.dataname}.csv"), append=True)

    del attack
    del model
    del prepper
    torch.cuda.empty_cache()


def evaluate_attack_fedavg(
    args, attack, diffusion_process, ground_truth, latents, model,
    prepper, n_cats_info, tableak_evaluator, performance_manager, device,
):
    """Per-client FedAvg attack: run K local SGD steps on the batch to produce a weight delta,
    restore the model to the pre-update weights, and invert the delta (rebuttal FedAvg setting)."""
    K = int(getattr(args, "fedavg_local_steps", 2))
    local_lr = float(getattr(args, "fedavg_local_lr", None) or getattr(args, "_model_lr", args.lr))
    param_list = list(model.parameters())
    theta0 = [p.detach().clone() for p in param_list]

    # Real client update: K local SGD steps on this batch (fresh latents each step).
    # Preserve the exact K latent draws because the attacker can only be given known entries
    # from those releases, not from the unrelated outer training batch.
    real_latents = []
    for _k in range(K):
        lat = diffusion_process.sample_latents(ground_truth, device)
        real_latents.append(lat)
        x_noised = diffusion_process.forward_diffuse(ground_truth, lat)
        loss = diffusion_process.compute_training_loss(model, x_noised, lat)
        grads = torch.autograd.grad(loss, param_list)
        with torch.no_grad():
            for p, g in zip(param_list, grads):
                p -= local_lr * g
    weight_delta = [p.detach() - t0 for p, t0 in zip(param_list, theta0)]

    # restore the model to theta0 so the attack starts from the known global weights
    with torch.no_grad():
        for p, t0 in zip(param_list, theta0):
            p.copy_(t0)

    known_latents = [
        diffusion_process.mask_known_latents(lat, args, ground_truth.shape[0], device)
        for lat in real_latents
    ]
    reconstruction, attack_metrics = attack.inversion_fedavg(
        weight_delta, ground_truth.shape, model, n_cats_info, known_latents, K, local_lr,
        simple=args.simple_attack, performance_manager=performance_manager,
    )
    for m in ["reconstruction_time", "time_optimization_s", "time_pooling_s", "time_alignment_s", "estimated_vram_mb"]:
        performance_manager.log_metric(m, attack_metrics[m])
    performance_manager.log_metric("fedavg_local_steps", K)
    _evaluate_reconstruction(args, reconstruction, ground_truth, prepper, tableak_evaluator, performance_manager)


def evaluate_attack(
    args,
    attack,
    diffusion_process,
    shape,
    ground_truth,
    latents,
    model,
    prepper,
    n_cats_info,
    tableak_evaluator,
    performance_manager,
):
    # send latest gradient to attack process
    grads = [p.grad.detach() if p.grad is not None else None for p in model.parameters()]
    grads = _apply_defense(args, grads, model, ground_truth, latents, diffusion_process, ground_truth.device)
    # print dummy latents (mask-verbose only)
    if args.mask_verbose and isinstance(diffusion_process, MaskedDiffusionProcess):
        tqdm.write(
            f"t = {latents['t'].numpy(force=True).tolist()}\nmask = {latents['mask'].int().numpy(force=True).tolist()}"
        )

    known_latents = diffusion_process.mask_known_latents(latents, args, shape[0], ground_truth.device)

    reconstruction = None
    attack_metrics = {}
    if args.baseline != "gidm":
        reconstruction, attack_metrics = attack.inversion(
            grad=grads,
            shape=shape,
            model=model,
            n_cats_amounts=n_cats_info,
            known_latents=known_latents,
            ground_truth=ground_truth,
            simple=args.simple_attack,
            performance_manager=performance_manager,
        )
    else:
        try:
            models_dir = f"saved_models/{args.dataname}/"
            backbone = getattr(args, "backbone", "mlp")
            model_path = os.path.join(
                models_dir,
                f"{args.diffusion_type}_{backbone}_model_pretrained.pt",
            )
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=grads[0].device)
                model_pretrained = build_diffusion_model(checkpoint["model_config"]).to(grads[0].device)
                model_pretrained.load_state_dict(checkpoint["state_dict"])
            else:
                # Existing rebuttal GIDM checkpoints predate architecture-tagged checkpoint
                # packages. They are raw MLP state dicts; keep them usable without retraining
                # nine datasets while new checkpoints use the self-describing format above.
                legacy_path = os.path.join(
                    models_dir,
                    f"{args.diffusion_type}_model_pretrained.pt",
                )
                if backbone != "mlp" or not os.path.exists(legacy_path):
                    raise FileNotFoundError(model_path)
                model_pretrained = build_diffusion_model(
                    diffusion_model_config(model),
                ).to(grads[0].device)
                legacy_state_dict = torch.load(legacy_path, map_location=grads[0].device)
                model_pretrained.load_state_dict(legacy_state_dict)
        except FileNotFoundError:
            print(
                "Model has not been trained. Run main.py with --save-model to first save a pretrained model for the GIDM attack with the given dataset"
            )
            exit()
        reconstruction = attack.inversion_gidm(
            grad=grads,
            shape=shape,
            model=model,
            model_pretrained=model_pretrained,
            n_cats_amounts=n_cats_info,
            known_latents=known_latents,
            ground_truth=ground_truth,
            simple=args.simple_attack,
            performance_manager=performance_manager,
        )

    indices = _evaluate_reconstruction(
        args, reconstruction, ground_truth, prepper, tableak_evaluator, performance_manager
    )

    raw_reconstruction = attack_metrics.get("raw_reconstruction") if attack_metrics else None
    if raw_reconstruction is not None:
        _evaluate_reconstruction(
            args, raw_reconstruction, ground_truth, prepper, tableak_evaluator,
            performance_manager, suffix="_raw",
        )

    if performance_manager is not None and attack_metrics:
        performance_manager.log_metric("reconstruction_time", attack_metrics.get("reconstruction_time", 0))
        performance_manager.log_metric("time_optimization_s", attack_metrics.get("time_optimization_s", 0))
        performance_manager.log_metric("time_pooling_s", attack_metrics.get("time_pooling_s", 0))
        performance_manager.log_metric("time_alignment_s", attack_metrics.get("time_alignment_s", 0))
        performance_manager.log_metric("estimated_vram_mb", attack_metrics.get("estimated_vram_mb", 0))
        performance_manager.log_metric("epoch_lr_used", attack_metrics.get("epoch_lr_used"))
        performance_manager.log_metric("final_grad_loss", attack_metrics.get("final_grad_loss"))
        performance_manager.log_metric("final_cossim_dist", attack_metrics.get("final_cossim_dist"))
        performance_manager.log_metric("final_grad_mse", attack_metrics.get("final_grad_mse"))

        final_latents = attack_metrics.get("final_latents")
        if final_latents is not None:
            latent_metrics = diffusion_process.evaluate_latent_reconstruction(final_latents, latents, indices=indices)
            for name, value in latent_metrics.items():
                performance_manager.log_metric(name, value)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
