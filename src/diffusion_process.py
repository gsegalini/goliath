import math
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from .utils import calc_diffusion_params
import torch.nn.functional as F


def _parse_random_set(args):
    """Parse args.random_latents into a set of latent names to randomize+freeze."""
    val = (getattr(args, "random_latents", "") or "").strip()
    if not val:
        return set()
    if val == "all":
        return {"t", "noise", "mask"}
    return {s.strip() for s in val.split(",") if s.strip()}


class DiffusionProcess(ABC):
    """Abstracts diffusion-specific logic for training and attack.

    A DiffusionProcess defines what latent variables exist (e.g. timesteps + noise
    for Gaussian, timesteps + mask for masked diffusion), how forward diffusion works,
    how to compute training loss, and how to manage dummy latent variables during attack.
    """

    def __init__(self, device):
        self.device = device

    # --- Training ---

    @abstractmethod
    def sample_latents(self, batch, device) -> dict[str, torch.Tensor]:
        """Sample random latent variables for a training step.
        Returns a dict of tensors, e.g. {"t": ..., "noise": ...}."""

    @abstractmethod
    def forward_diffuse(self, x_clean, latents):
        """Apply forward diffusion to clean data given latent variables.
        Returns the noised tensor x_t."""

    @abstractmethod
    def compute_training_loss(self, model, x_noised, latents):
        """Compute training loss given model, noised data, and latent variables."""

    # --- Attack: known-variable masking ---

    @abstractmethod
    def mask_known_latents(self, latents, args, batch_size, device) -> dict[str, torch.Tensor]:
        """Apply known-amount masking to latent variables.
        Unknown values are replaced with NaN. Returns a new dict."""

    # --- Attack: dummy variable lifecycle ---

    @abstractmethod
    def init_dummy_latents(
        self, known_latents, args, previous_latents=None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Initialize dummy latent variables for attack optimization.
        Returns (dummy_latents, nan_masks, starting_values) — all dicts of tensors.

        If `previous_latents` is provided, unknown (NaN) positions are warm-started
        from it instead of the default random/midpoint init. Known positions always
        come from `known_latents`. Used by multi-epoch "previous" init strategy."""

    @abstractmethod
    def build_latent_optimizers(self, dummy_latents, args, simple) -> dict[str, torch.optim.Optimizer | None]:
        """Create optimizers for latent variables that need optimization.
        Returns a dict of name -> (optimizer or None)."""

    @abstractmethod
    def restore_known_latents(self, dummy_latents, nan_masks, starting_values):
        """Restore known (non-NaN) values after an optimizer step."""

    # --- Attack: forward pass + inner loss ---

    @abstractmethod
    def forward_diffuse_attack(self, dummy_data_softmax, dummy_latents, batch_size):
        """Forward diffusion during attack on dummy variables.
        Returns the noised tensor."""

    @abstractmethod
    def compute_attack_inner_loss(self, model, x_noised, dummy_latents):
        """Compute the 'inner' loss used to generate predicted gradients during attack.
        For Gaussian: MSE(model(x_t, t), noise)."""

    # --- Denoising (for baselines like GIDM) ---

    def fast_denoise(self, dummy_latent, model_pretrained):
        raise NotImplementedError(f"{type(self).__name__} does not support fast_denoise")

    def full_denoise(self, dummy_latent, model_pretrained):
        raise NotImplementedError(f"{type(self).__name__} does not support full_denoise")

    # --- Utility ---

    @abstractmethod
    def get_optimizable_flags(self, args) -> dict[str, bool]:
        """Return a dict of latent_name -> bool indicating which latents are optimized."""

    def evaluate_latent_reconstruction(
        self, dummy_latents: dict, real_latents: dict, indices: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Compare reconstructed latent variables against real ones.
        Returns a dict of metric_name -> value. Override in subclasses.
        Supply indices for re-ordering dummy_latents after reordering rows of tables."""
        return {}

    def compute_latent_prior_loss(self, dummy_latents) -> torch.Tensor:
        """Extra regularizer on dummy latents added to the outer (grad-matching) loss.
        Returns a scalar tensor. Default: zero. Override in subclasses for coupling terms."""
        return torch.zeros((), device=self.device)

    def set_attack_iteration(self, iteration: int):
        """Set diffusion-specific attack schedules for one outer optimization iteration."""

    def get_debug_stats(self, dummy_latents) -> dict[str, float]:
        """Per-iter diagnostic scalars for --debug-prior traces. Default: empty."""
        return {}

    @staticmethod
    def _interp_timewise(table: torch.Tensor, t_float: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate a timestep-indexed table at float t.

        Integer indexing (table[round(t)]) kills the gradient to t; linear
        interpolation between floor/ceil neighbours keeps autograd flowing.
        Returns a (B,) tensor on the same device as `table`.
        """
        T = table.shape[0]
        t_c = torch.clamp(t_float, 0.0, float(T - 1))
        t_floor = torch.floor(t_c).long()
        t_ceil = torch.clamp(t_floor + 1, 0, T - 1)
        frac = t_c - t_floor.float()
        return (1 - frac) * table[t_floor] + frac * table[t_ceil]

    def cleanup_dummy_latents(self, dummy_latents):
        """Delete dummy latent tensors to free memory. Override if needed."""
        for v in dummy_latents.values():
            del v


class GaussianDiffusionProcess(DiffusionProcess):
    """Variance-preserving (VP) Gaussian diffusion with linear beta schedule."""

    def __init__(self, args, device):
        super().__init__(device)
        self.args = args
        diffusion_params = calc_diffusion_params(args.beta_1, args.beta_T, args.timesteps)
        self.alpha_bars = torch.tensor(diffusion_params["alpha_bars"], dtype=torch.float32).to(device)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.diffusion_params = diffusion_params
        self.criterion = nn.MSELoss()
        requested_noise_lr_mult = getattr(args, "noise_lr_mult", None)
        if requested_noise_lr_mult is None:
            self.noise_lr_mult = (
                1.0 if getattr(args, "backbone", "mlp") == "ft-transformer" else 0.1
            )
        else:
            self.noise_lr_mult = requested_noise_lr_mult
        # Same knob for the timestep optimizer, which has always been hardcoded at 10x. The
        # noise multiplier turned out to matter a lot on the transformer, and t is the other
        # nuisance latent, so it is worth being able to condition it independently.
        requested_t_lr_mult = getattr(args, "t_lr_mult", None)
        self.t_lr_mult = 10.0 if requested_t_lr_mult is None else requested_t_lr_mult

        self.optimize_t = not (args.no_t or args.known_t_amount == 100)
        self.optimize_noise = not (args.no_noise or args.known_noise_amount == 100)

        random_set = _parse_random_set(args)
        self._random_t = "t" in random_set
        self._random_noise = "noise" in random_set
        if self._random_t:
            self.optimize_t = False
        if self._random_noise:
            self.optimize_noise = False

    # --- Training ---

    def sample_latents(self, batch, device):
        t = torch.randint(self.args.timesteps, size=(batch.shape[0],)).to(device)
        noise = torch.normal(0, 1, size=batch.shape).to(device)
        return {"t": t, "noise": noise}

    def forward_diffuse(self, x_clean, latents):
        t = latents["t"]
        noise = latents["noise"]
        coeff_1 = self.sqrt_alpha_bars[t].reshape((len(t), 1))
        coeff_2 = self.sqrt_one_minus_alpha_bars[t].reshape((len(t), 1))
        x_noised = coeff_1 * x_clean + coeff_2 * noise
        return x_noised

    def compute_training_loss(self, model, x_noised, latents):
        t = latents["t"].float()
        noise = latents["noise"]
        noise_predicted = model(x_noised, t)
        return self.criterion(noise_predicted, noise)

    # --- Attack: known-variable masking ---

    def mask_known_latents(self, latents, args, batch_size, device):
        known_t = self._mask_known_t(latents["t"], args, batch_size, device)
        known_noise = self._mask_known_noise(latents["noise"], args, batch_size, device)
        return {"t": known_t, "noise": known_noise}

    def _mask_known_t(self, timesteps, args, batch_size, device):
        if args.simple_attack or args.known_t_amount >= 100:
            return timesteps.clone().float()
        indices = np.random.choice(batch_size, (args.known_t_amount * batch_size) // 100, replace=False)
        masked = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=device)
        masked[indices] = timesteps[indices].float()
        return masked

    def _mask_known_noise(self, sigmas, args, batch_size, device):
        if args.simple_attack or args.known_noise_amount >= 100:
            return sigmas.clone()
        indices = np.random.choice(batch_size, (args.known_noise_amount * batch_size) // 100, replace=False)
        masked = torch.full_like(sigmas, float("nan"))
        masked[indices] = sigmas[indices]
        return masked.to(device)

    # --- Attack: dummy variable lifecycle ---

    def init_dummy_latents(self, known_latents, args, previous_latents=None):
        prev_t = previous_latents["t"] if previous_latents is not None else None
        prev_noise = previous_latents["noise"] if previous_latents is not None else None
        dummy_t, t_nan_mask, starting_t = self._init_dummy_t(known_latents["t"], prev_t)
        dummy_noise, noise_nan_mask, starting_noise = self._init_dummy_noise(known_latents["noise"], prev_noise)
        dummy_latents = {"t": dummy_t, "noise": dummy_noise}
        nan_masks = {"t": t_nan_mask, "noise": noise_nan_mask}
        starting_values = {"t": starting_t, "noise": starting_noise}
        return dummy_latents, nan_masks, starting_values

    def _init_dummy_t(self, known_t, previous_t=None):
        dummy_t = known_t.clone()
        t_nan_mask = torch.isnan(known_t)
        n_unknown = int(t_nan_mask.sum().item())
        if n_unknown > 0:
            if previous_t is not None:
                dummy_t[t_nan_mask] = previous_t.to(known_t.device)[t_nan_mask].float()
            elif self._random_t or getattr(self.args, "t_init", "random") == "random":
                rand_t = torch.randint(0, self.args.timesteps, (n_unknown,), device=known_t.device).float()
                dummy_t[t_nan_mask] = rand_t
            else:
                dummy_t[t_nan_mask] = float(self.args.timesteps // 2)
        dummy_t.requires_grad_(True)
        return dummy_t, t_nan_mask, dummy_t.detach().clone()

    def _init_dummy_noise(self, known_noise, previous_noise=None):
        dummy_noise = known_noise.clone()
        noise_nan_mask = torch.isnan(known_noise)
        if previous_noise is not None:
            dummy_noise[noise_nan_mask] = previous_noise.to(known_noise.device)[noise_nan_mask]
        else:
            dummy_noise[noise_nan_mask] = torch.randn_like(dummy_noise)[noise_nan_mask]
        dummy_noise.requires_grad_(True)
        return dummy_noise, noise_nan_mask, dummy_noise.detach().clone()

    def build_latent_optimizers(self, dummy_latents, args, simple) -> dict[str, torch.optim.Optimizer | None]:
        optimizer_t = (
            torch.optim.Adam([dummy_latents["t"]], lr=self.t_lr_mult * args.attack_lr, betas=(0.9, 0.9))
            if not simple and self.optimize_t
            else None
        )
        optimizer_noise = (
            torch.optim.Adam([dummy_latents["noise"]], lr=self.noise_lr_mult * args.attack_lr)
            if not simple and self.optimize_noise
            else None
        )
        return {"t": optimizer_t, "noise": optimizer_noise}

    def restore_known_latents(self, dummy_latents, nan_masks, starting_values):
        # Restore timesteps
        dummy_t = dummy_latents["t"]
        t_nan_mask = nan_masks["t"]
        starting_t = starting_values["t"]
        with torch.no_grad():
            # Keep dummy_t continuous so Adam's sub-integer updates survive.
            # _interp_timewise handles fractional t via linear interpolation.
            dummy_t.clamp_(0, self.args.timesteps - 1)
            dummy_t[~t_nan_mask] = starting_t[~t_nan_mask]
        dummy_t.requires_grad_(True)

        # Restore noise
        dummy_noise = dummy_latents["noise"]
        noise_nan_mask = nan_masks["noise"]
        starting_noise = starting_values["noise"]
        with torch.no_grad():
            dummy_noise[~noise_nan_mask] = starting_noise[~noise_nan_mask]
        dummy_noise.requires_grad_(True)

    # --- Attack: forward pass + inner loss ---

    def forward_diffuse_attack(self, dummy_data_softmax, dummy_latents, batch_size):
        dummy_t = dummy_latents["t"]
        dummy_noise = dummy_latents["noise"]
        # Linear interpolation over alpha_bars so gradient flows back to dummy_t
        # through x_noised. Using alpha_bars[round(t)] (the old path) kills this
        # gradient because integer indexing is non-differentiable, leaving only
        # a weak 2nd-order signal from model(x_noised, t) via time embedding.
        alpha_bar = self._interp_timewise(self.alpha_bars, dummy_t)
        coeff_1 = torch.sqrt(alpha_bar).reshape((batch_size, 1))
        coeff_2 = torch.sqrt(1 - alpha_bar).reshape((batch_size, 1))
        return coeff_1 * dummy_data_softmax + coeff_2 * dummy_noise

    def compute_attack_inner_loss(self, model, x_noised, dummy_latents):
        dummy_t = dummy_latents["t"]
        dummy_noise = dummy_latents["noise"]
        noise_predicted = model(x_noised, dummy_t)
        return self.criterion(noise_predicted, dummy_noise)

    def get_model_dims(self, original_dim):
        return original_dim, original_dim

    # --- Denoising (for GIDM baseline) ---

    def fast_denoise(self, dummy_latent, model_pretrained):
        timesteps = torch.full(size=(dummy_latent.shape[0],), fill_value=self.args.timesteps - 1).to(self.device)
        t_idx = self.args.timesteps - 1
        sigmas_predicted = model_pretrained(dummy_latent, timesteps)
        clean_estimate = (
            dummy_latent - self.sqrt_one_minus_alpha_bars[t_idx] * sigmas_predicted
        ) / self.sqrt_alpha_bars[t_idx]
        return clean_estimate

    def full_denoise(self, dummy_latent, model_pretrained):
        x_t = deepcopy(dummy_latent)
        with torch.no_grad():
            for t in range(self.args.timesteps - 1, -1, -1):
                timesteps = torch.full(size=(x_t.shape[0],), fill_value=t).to(self.device)
                alpha_bar_t = torch.tensor(self.diffusion_params["alpha_bars"][t]).to(self.device)
                alpha_t = torch.tensor(self.diffusion_params["alphas"][t]).to(self.device)
                beta_t = torch.tensor(self.diffusion_params["betas"][t]).to(self.device)
                sigmas_predicted = model_pretrained(x_t, timesteps)
                x_t = x_t / torch.sqrt(alpha_t) - (beta_t / torch.sqrt(alpha_t * (1 - alpha_bar_t))) * sigmas_predicted
        return x_t

    # --- Utility ---

    def get_optimizable_flags(self, args):
        return {"t": self.optimize_t, "noise": self.optimize_noise}

    def evaluate_latent_reconstruction(self, dummy_latents, real_latents, indices=None):
        metrics = {}
        real_t = real_latents["t"].cpu().float()
        pred_t = dummy_latents["t"].detach().cpu().float()
        metrics["t_mae"] = (real_t - pred_t).abs().mean().item()

        real_noise = real_latents["noise"].cpu()
        pred_noise = dummy_latents["noise"].detach().cpu()
        metrics["noise_mse"] = (real_noise - pred_noise).pow(2).mean().item()

        if indices is not None:
            aligned_dummy_latents = {k: v[indices] for k, v in dummy_latents.items()}
            pred_t_aligned = aligned_dummy_latents["t"].detach().cpu().float()
            pred_noise_aligned = aligned_dummy_latents["noise"].detach().cpu()
            metrics["t_mae_aligned"] = (real_t - pred_t_aligned).abs().mean().item()
            metrics["noise_mse_aligned"] = (real_noise - pred_noise_aligned).pow(2).mean().item()
        return metrics

    def get_all_optimizable_params(self, dummy_latents):
        """Return list of dummy latent tensors that should be optimized (for full-dataset shared optimizer)."""
        params = []
        if self.optimize_t:
            params.append(dummy_latents["t"])
        if self.optimize_noise:
            params.append(dummy_latents["noise"])
        return params

    def restore_known_t_only(self, dummy_latents, nan_masks, starting_values):
        """Restore only timesteps (used in full-dataset where we check flags individually)."""
        dummy_t = dummy_latents["t"]
        with torch.no_grad():
            dummy_t.clamp_(0, self.args.timesteps - 1)
            dummy_t[~nan_masks["t"]] = starting_values["t"][~nan_masks["t"]]
        dummy_t.requires_grad_(True)

    def restore_known_noise_only(self, dummy_latents, nan_masks, starting_values):
        """Restore only noise (used in full-dataset where we check flags individually)."""
        dummy_noise = dummy_latents["noise"]
        with torch.no_grad():
            dummy_noise[~nan_masks["noise"]] = starting_values["noise"][~nan_masks["noise"]]
        dummy_noise.requires_grad_(True)

    def get_debug_stats(self, dummy_latents) -> dict[str, float]:
        """Per-iter diagnostic scalars for --debug-prior traces in Gaussian diffusion."""
        with torch.no_grad():
            t = dummy_latents["t"].float()
            noise = dummy_latents["noise"]
            
            return {
                "dummy_t_mean": t.mean().item(),
                "dummy_t_std": t.std().item() if t.numel() > 1 else 0.0,
                "dummy_t_min": t.min().item(),
                "dummy_t_max": t.max().item(),
                "dummy_noise_mean": noise.mean().item(),
                "dummy_noise_std": noise.std().item() if noise.numel() > 1 else 0.0,
                "dummy_noise_min": noise.min().item(),
                "dummy_noise_max": noise.max().item(),
            }


class MaskedDiffusionProcess(DiffusionProcess):

    MASK_TEMP_INIT = 5.0
    MASK_TEMP_MIN = 0.01  # Target temperature at the final iteration

    # KL prior schedule: warmup (flat 0) -> linear ramp -> plateau at peak -> cosine cooldown to 0.
    PRIOR_WARMUP_FRAC = 0.05
    PRIOR_RAMP_END_FRAC = 0.15
    PRIOR_COOLDOWN_START_FRAC = 1.00
    PRIOR_PEAK = 0.5

    def __init__(self, args, device, feat_info):
        super().__init__(device)
        self.args = args
        self.T = args.timesteps
        diffusion_params = calc_diffusion_params(args.beta_1, args.beta_T, args.timesteps)
        self.alpha_bars = torch.tensor(diffusion_params["alpha_bars"], dtype=torch.float32).to(device)
        # Precompute masked diffusion ELBO weights: w(t) = (alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t)
        # Convention: alpha_bar_{-1} = 1 (no masking before t=0)
        alpha_bars_prev = np.concatenate([[1.0], diffusion_params["alpha_bars"][:-1]])
        elbo_weights = (alpha_bars_prev - diffusion_params["alpha_bars"]) / (1 - diffusion_params["alpha_bars"])
        # Floor ELBO weights so mask grads don't vanish at extreme t (w(t) -> 0 near t=0 and t=T).
        elbo_weights = np.maximum(elbo_weights, 1e-3)
        self.elbo_weights = torch.tensor(elbo_weights, dtype=torch.float32).to(device)
        self.mask_temperature = self.MASK_TEMP_MIN
        self._prior_iter = 0
        self._prior_schedule = self._build_prior_schedule(args.optim_iters)
        self.diffusion_params = diffusion_params
        self.optimize_t = not (args.no_t or args.known_t_amount == 100)
        self.optimize_mask = not (getattr(args, "no_mask", False) or getattr(args, "known_mask_amount", 0) == 100)
        self.optimize_noise = not (args.no_noise or args.known_noise_amount == 100)

        random_set = _parse_random_set(args)
        self._random_t = "t" in random_set
        self._random_noise = "noise" in random_set
        self._random_mask = "mask" in random_set
        if self._random_t:
            self.optimize_t = False
        if self._random_noise:
            self.optimize_noise = False
        if self._random_mask:
            self.optimize_mask = False

        self.n_cont = feat_info["cont"]
        self.n_cats = feat_info["cat"]
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.CrossEntropyLoss(reduction="none")
        requested_noise_lr_mult = getattr(args, "noise_lr_mult", None)
        self.noise_lr_mult = 1.0 if requested_noise_lr_mult is None else requested_noise_lr_mult
        requested_t_lr_mult = getattr(args, "t_lr_mult", None)
        self.t_lr_mult = 10.0 if requested_t_lr_mult is None else requested_t_lr_mult
        self._cached_st_mask = None
        self._cached_dummy_data = None
        self._cached_alpha_bar = None

        # --- PRECOMPUTED INDICES FOR VECTORIZATION ---
        self.C = len(self.n_cats)
        self.max_cat = max(self.n_cats)

        # 1. Forward pass scattering indices
        cat_mask_expansion_indices = []
        for i, n in enumerate(self.n_cats):
            cat_mask_expansion_indices.extend([i] * n)

        # Keep the original compact feature layout first and append one mask token per logical
        # categorical column. The FT-Transformer's per-token output head returns the leading
        # `original_dim` tokens, so interleaving mask tokens here would silently feed mask-token
        # outputs to the categorical reconstruction loss.
        n_cat_dims = sum(self.n_cats)
        self.cat_feature_indices = torch.arange(n_cat_dims, dtype=torch.long, device=device)
        self.cat_mask_indices = torch.arange(
            n_cat_dims, n_cat_dims + self.C, dtype=torch.long, device=device,
        )
        # Maps each sub-feature back to its parent categorical group (for expanding the Bxlen(n_cats) mask)
        self.cat_mask_expansion_indices = torch.tensor(cat_mask_expansion_indices, dtype=torch.long, device=device)

        # 2. Loss computation gathering indices (pads to max_cat size)
        pad_mask = torch.zeros((self.C, self.max_cat), dtype=torch.bool)
        gather_indices = torch.zeros((self.C, self.max_cat), dtype=torch.long)
        orig_idx = 0
        for i, n in enumerate(self.n_cats):
            pad_mask[i, :n] = True
            # The indices for padded spots don't matter (we use 0) as they get masked out
            gather_indices[i, :n] = torch.arange(orig_idx, orig_idx + n)
            orig_idx += n

        self.pad_mask = pad_mask.to(device)
        # Flattened for 1D advanced indexing: shape (C * max_cat,)
        self.gather_indices = gather_indices.view(-1).to(device)

    # --- Training ---

    def sample_latents(self, batch, device):
        B, D = batch.shape
        t = torch.randint(self.T, size=(B,)).to(device)
        noise = torch.normal(0, 1, size=(B, self.n_cont)).to(device)
        # gamma = (t.float() + 1) / self.T  # masking probability per sample
        """

        prob_i = torch.zeros(batch.shape[0], 2, device=batch.device)
                prob_i[:, 0] = alpha_bars[timesteps]
                prob_i[:, 1] = 1 - alpha_bars[timesteps]
                log_prob_i = torch.log(prob_i)
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
        """
        gamma = 1 - self.alpha_bars[t]
        probs = gamma.unsqueeze(1).expand(B, len(self.n_cats))
        mask = torch.bernoulli(probs).to(device)  # 1 = masked, 0 = kept
        return {"t": t, "mask": mask, "noise": noise}

    def forward_diffuse(self, x_clean, latents):
        mask = latents["mask"]  # (B, C)
        t = latents["t"]
        noise = latents["noise"]
        latents["x_clean"] = x_clean
        
        coeff_1 = torch.sqrt(self.alpha_bars[t]).reshape(-1, 1)
        coeff_2 = torch.sqrt(1 - self.alpha_bars[t]).reshape(-1, 1)
        
        x_cont_noised = coeff_1 * x_clean[:, :self.n_cont] + coeff_2 * noise
        
        # --- Vectorized Categorical Masking ---
        x_cat_clean = x_clean[:, self.n_cont:]
        B = x_clean.shape[0]
        
        # Expands the (B, C) mask to (B, sum(n_cats)) so it aligns with the feature probabilities
        expanded_mask = mask[:, self.cat_mask_expansion_indices]
        
        x_cat_noised = torch.zeros((B, sum(self.n_cats) + self.C), dtype=torch.float32, device=self.device)
        
        # Scatter features (zeroed out if masked)
        x_cat_noised[:, self.cat_feature_indices] = (1 - expanded_mask) * x_cat_clean
        # Scatter mask tokens
        x_cat_noised[:, self.cat_mask_indices] = mask

        return torch.concat((x_cont_noised, x_cat_noised), dim=1)

    def compute_training_loss(self, model, x_noised, latents):
        t = latents["t"].float()
        mask = latents["mask"]
        noise = latents["noise"]
        x_clean = latents["x_clean"]
        
        x_pred = model(x_noised, t)
        cont_loss = self.criterion1(x_pred[:, :self.n_cont], noise)
        
        # --- Vectorized Categorical Loss ---
        x_pred_cat = x_pred[:, self.n_cont:]
        x_clean_cat = x_clean[:, self.n_cont:]
        B = x_pred.shape[0]

        # Gather logits and targets into shape (B, C, max_cat)
        gathered_logits = x_pred_cat[:, self.gather_indices].view(B, self.C, self.max_cat)
        gathered_targets = x_clean_cat[:, self.gather_indices].view(B, self.C, self.max_cat)

        # Apply padding mask: set padded logits to -inf so log_softmax ignores them
        pad_mask_ext = self.pad_mask.unsqueeze(0)  # (1, C, max_cat)
        gathered_logits = gathered_logits.masked_fill(~pad_mask_ext, -float('inf'))
        
        # Compute log probabilities natively
        log_probs = F.log_softmax(gathered_logits, dim=-1)
        # Avoid 0 * -inf at padded positions below. Padded logits must be excluded from
        # log_softmax, but their post-softmax value can safely be zero because their target is zero.
        log_probs = log_probs.masked_fill(~pad_mask_ext, 0.0)

        # Zero out padding in targets to prevent errant loss accumulation
        gathered_targets = gathered_targets.masked_fill(~pad_mask_ext, 0.0)

        # Manual Cross Entropy calculation over the last dimension: -sum(target * log_prob)
        per_sample_feature_ce = -torch.sum(gathered_targets * log_probs, dim=-1)  # (B, C)

        w_t = self.elbo_weights[latents["t"]].unsqueeze(1)  # (B, 1)
        weighted_ce = per_sample_feature_ce * mask * w_t  # (B, C)

        # A flat .mean() computes (sum over B and C) / (B * C), perfectly matching your original loop logic
        cat_loss = weighted_ce.mean()

        return cont_loss + cat_loss

    # --- Attack: known-variable masking ---

    def mask_known_latents(self, latents, args, batch_size, device):
        known_t = self._mask_known_t(latents["t"], args, batch_size, device)
        known_mask = self._mask_known_mask(latents["mask"], args, batch_size, device)
        known_noise = self._mask_known_noise(latents["noise"], args, batch_size, device)
        return {"t": known_t, "mask": known_mask, "noise": known_noise}

    def _mask_known_t(self, timesteps, args, batch_size, device):
        if args.simple_attack or args.known_t_amount >= 100:
            return timesteps.clone().float()
        indices = np.random.choice(batch_size, (args.known_t_amount * batch_size) // 100, replace=False)
        masked = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=device)
        masked[indices] = timesteps[indices].float()
        return masked

    def _mask_known_noise(self, sigmas, args, batch_size, device):
        if args.simple_attack or args.known_noise_amount >= 100:
            return sigmas.clone()
        indices = np.random.choice(batch_size, (args.known_noise_amount * batch_size) // 100, replace=False)
        masked = torch.full_like(sigmas, float("nan"))
        masked[indices] = sigmas[indices]
        return masked.to(device)

    def _mask_known_mask(self, mask, args, batch_size, device):
        """Mask known mask values. Selects known_mask_amount% of sample rows to keep; NaN the rest."""
        known_mask_amount = getattr(args, "known_mask_amount", 0)
        if args.simple_attack or known_mask_amount >= 100:
            return mask.clone()
        indices = np.random.choice(batch_size, (known_mask_amount * batch_size) // 100, replace=False)
        masked = torch.full_like(mask, float("nan"))
        masked[indices] = mask[indices]
        return masked.to(device)

    # --- Attack: dummy variable lifecycle ---

    def _build_prior_schedule(self, optim_iters: int) -> torch.Tensor:
        warmup_end = int(self.PRIOR_WARMUP_FRAC * optim_iters)
        peak_start = int(self.PRIOR_RAMP_END_FRAC * optim_iters)
        cooldown_frac = getattr(self.args, "prior_cooldown_start", None)
        if cooldown_frac is None:
            cooldown_frac = self.PRIOR_COOLDOWN_START_FRAC
        cooldown_start = int(cooldown_frac * optim_iters)
        peak = getattr(self.args, "prior_peak", None)
        if peak is None:
            peak = self.PRIOR_PEAK
        
        ramp_len = max(peak_start - warmup_end, 1)
        cooldown_len = max(optim_iters - cooldown_start, 1)
        
        i = torch.arange(optim_iters, dtype=torch.float32)
        schedule = torch.zeros(optim_iters, dtype=torch.float32)
        
        # Warmup is natively 0.0
        
        # Linear Ramp
        ramp_mask = (i >= warmup_end) & (i < peak_start)
        schedule[ramp_mask] = peak * (i[ramp_mask] - warmup_end) / ramp_len
        
        # Plateau
        plateau_mask = (i >= peak_start) & (i < cooldown_start)
        schedule[plateau_mask] = peak
        
        # Cooldown
        cooldown_mask = i >= cooldown_start
        frac = (i[cooldown_mask] - cooldown_start) / cooldown_len
        schedule[cooldown_mask] = peak * 0.5 * (1 + torch.cos(math.pi * frac))
        
        return schedule

    def init_dummy_latents(self, known_latents, args, previous_latents=None):
        self.mask_temp_init = getattr(args, "mask_temp_init", self.MASK_TEMP_INIT)
        self.mask_temperature = self.mask_temp_init
        self.mask_temp_min = getattr(args, "mask_temp_min", self.MASK_TEMP_MIN)

        # Calculate adaptive exponential decay: T_t = T_0 * (T_min / T_0) ^ (t / N)
        if args.optim_iters > 1:
            self.mask_temp_decay = math.exp(math.log(self.mask_temp_min / self.mask_temperature) / (args.optim_iters - 1))
        else:
            self.mask_temp_decay = 1.0

        self._prior_iter = 0
        self._prior_schedule = self._build_prior_schedule(args.optim_iters).to(self.device)
        prev_t = previous_latents["t"] if previous_latents is not None else None
        prev_noise = previous_latents["noise"] if previous_latents is not None else None
        prev_mask_logits = previous_latents["mask_logits"] if previous_latents is not None else None
        dummy_t, t_nan_mask, starting_t = self._init_dummy_t(known_latents["t"], prev_t)
        dummy_mask_logits, mask_nan_mask, starting_mask_logits = self._init_dummy_mask_logits(
            known_latents["mask"], prev_mask_logits
        )
        dummy_noise, noise_nan_mask, starting_noise = self._init_dummy_noise(known_latents["noise"], prev_noise)
        dummy_latents = {"t": dummy_t, "mask_logits": dummy_mask_logits, "noise": dummy_noise}
        nan_masks = {"t": t_nan_mask, "mask": mask_nan_mask, "noise": noise_nan_mask}
        starting_values = {"t": starting_t, "mask_logits": starting_mask_logits, "noise": starting_noise}
        return dummy_latents, nan_masks, starting_values

    def set_attack_iteration(self, iteration: int):
        # Derive schedules from the outer iteration rather than mutating them in
        # restore_known_latents. This keeps joint/full-dataset attacks from advancing the
        # temperature and prior once per batch.
        self._prior_iter = int(iteration)
        self.mask_temperature = max(
            self.mask_temp_init * (self.mask_temp_decay ** int(iteration)),
            self.mask_temp_min,
        )

    def _init_dummy_t(self, known_t, previous_t=None):
        dummy_t = known_t.clone()
        t_nan_mask = torch.isnan(known_t)
        n_unknown = int(t_nan_mask.sum().item())
        if n_unknown > 0:
            if previous_t is not None:
                dummy_t[t_nan_mask] = previous_t.to(known_t.device)[t_nan_mask].float()
            elif self._random_t or getattr(self.args, "t_init", "random") == "random":
                rand_t = torch.randint(0, self.args.timesteps, (n_unknown,), device=known_t.device).float()
                dummy_t[t_nan_mask] = rand_t
            else:
                dummy_t[t_nan_mask] = float(self.args.timesteps // 2)
        dummy_t.requires_grad_(True)
        return dummy_t, t_nan_mask, dummy_t.detach().clone()

    def _init_dummy_noise(self, known_noise, previous_noise=None):
        dummy_noise = known_noise.clone()
        noise_nan_mask = torch.isnan(known_noise)
        if previous_noise is not None:
            dummy_noise[noise_nan_mask] = previous_noise.to(known_noise.device)[noise_nan_mask]
        else:
            dummy_noise[noise_nan_mask] = torch.randn_like(dummy_noise)[noise_nan_mask]
        dummy_noise.requires_grad_(True)
        return dummy_noise, noise_nan_mask, dummy_noise.detach().clone()

    def _init_dummy_mask_logits(self, known_mask, previous_mask_logits=None):
        """Initialize mask logits from known binary mask values.

        Known positions use large logits. Unknown positions get small per-sample
        random perturbation around 0 to break symmetry across the batch (prevents
        collapsed mid-value stall under permutation-invariant gradient matching).
        Random-mask baseline path (frozen mask) uses very large logits so sigmoid stays
        near 0/1 even under the initial mask temperature (~20).
        """
        mask_nan_mask = torch.isnan(known_mask)
        mask_logits = torch.where(
            known_mask > 0.5,
            torch.tensor(10.0, device=known_mask.device),
            torch.tensor(-10.0, device=known_mask.device),
        )
        if self._random_mask:
            big = 500.0
            rand_bin = (torch.rand_like(mask_logits) > 0.5).float()
            rand_logits = torch.where(
                rand_bin > 0.5,
                torch.tensor(big, device=known_mask.device),
                torch.tensor(-big, device=known_mask.device),
            )
            mask_logits[mask_nan_mask] = rand_logits[mask_nan_mask]
        elif previous_mask_logits is not None:
            mask_logits[mask_nan_mask] = previous_mask_logits.to(known_mask.device)[mask_nan_mask]
        else:
            init_sign = torch.where(torch.rand_like(mask_logits) > 0.5, 1.0, -1.0)
            init_noise = init_sign * 3.0
            mask_logits[mask_nan_mask] = init_noise[mask_nan_mask]
        mask_logits.requires_grad_(True)
        return mask_logits, mask_nan_mask, mask_logits.detach().clone()

    def build_latent_optimizers(self, dummy_latents, args, simple) -> dict[str, torch.optim.Optimizer | None]:
        optimizer_t = (
            torch.optim.Adam([dummy_latents["t"]], lr=self.t_lr_mult * args.attack_lr, betas=(0.9, 0.9))
            if not simple and self.optimize_t
            else None
        )
        optimizer_mask = (
            torch.optim.Adam([dummy_latents["mask_logits"]], lr=args.attack_lr, betas=(0.9, 0.9))
            if not simple and self.optimize_mask
            else None
        )
        optimizer_noise = (
            torch.optim.Adam([dummy_latents["noise"]], lr=self.noise_lr_mult * args.attack_lr)
            if not simple and self.optimize_noise
            else None
        )
        return {"t": optimizer_t, "mask": optimizer_mask, "noise": optimizer_noise}

    def restore_known_latents(self, dummy_latents, nan_masks, starting_values):
        # Keep dummy_t continuous so Adam's sub-integer updates survive.
        dummy_t = dummy_latents["t"]
        t_nan_mask = nan_masks["t"]
        starting_t = starting_values["t"]
        with torch.no_grad():
            dummy_t.clamp_(0, self.T - 1)
            dummy_t[~t_nan_mask] = starting_t[~t_nan_mask]
        dummy_t.requires_grad_(True)

        # Restore mask logits
        mask_logits = dummy_latents["mask_logits"]
        mask_nan_mask = nan_masks["mask"]
        starting_logits = starting_values["mask_logits"]
        with torch.no_grad():
            mask_logits[~mask_nan_mask] = starting_logits[~mask_nan_mask]
        mask_logits.requires_grad_(True)

        # Restore noise
        dummy_noise = dummy_latents["noise"]
        noise_nan_mask = nan_masks["noise"]
        starting_noise = starting_values["noise"]
        with torch.no_grad():
            dummy_noise[~noise_nan_mask] = starting_noise[~noise_nan_mask]
        dummy_noise.requires_grad_(True)

    # --- Attack: forward pass + inner loss ---

    def forward_diffuse_attack(self, dummy_data_softmax, dummy_latents, batch_size):
        mask_logits = dummy_latents["mask_logits"]
        soft_mask = torch.sigmoid(mask_logits / self.mask_temperature)
        hard_mask = (soft_mask > 0.5).float()
        st_mask = hard_mask + soft_mask - soft_mask.detach()
        self._cached_st_mask = st_mask
        self._cached_dummy_data = dummy_data_softmax
        
        t = dummy_latents['t']
        # Differentiable alpha_bar(t) via linear interpolation so the forward
        # diffusion (x_cont_noised below) carries gradient back to dummy_t.
        # Without this, coeff_1/coeff_2 would come from alpha_bars[round(t)],
        # which is integer-indexed and cuts the gradient path entirely.
        alpha_bar = self._interp_timewise(self.alpha_bars, t)
        self._cached_alpha_bar = alpha_bar
        noise = dummy_latents['noise']
        
        coeff_1 = torch.sqrt(alpha_bar).reshape(-1, 1)
        coeff_2 = torch.sqrt(1 - alpha_bar).reshape(-1, 1)
        
        x_cont_noised = coeff_1 * dummy_data_softmax[:, :self.n_cont] + coeff_2 * noise
        
        # --- Vectorized Categorical Attack Masking ---
        x_cat_clean = dummy_data_softmax[:, self.n_cont:]
        B = dummy_data_softmax.shape[0]
        
        expanded_mask = st_mask[:, self.cat_mask_expansion_indices]
        
        x_cat_noised = torch.zeros((B, sum(self.n_cats) + self.C), dtype=torch.float32, device=self.device)
        x_cat_noised[:, self.cat_feature_indices] = (1 - expanded_mask) * x_cat_clean
        x_cat_noised[:, self.cat_mask_indices] = st_mask

        return torch.concat((x_cont_noised, x_cat_noised), dim=1)

    def compute_attack_inner_loss(self, model, x_noised, dummy_latents):
        dummy_t = dummy_latents["t"]
        dummy_noise = dummy_latents['noise']
        st_mask = self._cached_st_mask
        dummy_data = self._cached_dummy_data
        # Differentiable ELBO weight w(t) via linear interpolation over the
        # integer-indexed self.elbo_weights table. Same reason as alpha_bar
        # interp in forward_diffuse_attack: keeps the gradient path to dummy_t
        # alive through the per-sample CE weighting below.
        w_t = self._interp_timewise(self.elbo_weights, dummy_t).unsqueeze(1)  # (B, 1)
        x_pred = model(x_noised, dummy_t)
        
        cont_loss = self.criterion1(x_pred[:, :self.n_cont], dummy_noise)
        
        # --- Vectorized Categorical Attack Loss ---
        x_pred_cat = x_pred[:, self.n_cont:]
        soft_cat = dummy_data[:, self.n_cont:]
        B = x_pred.shape[0]

        # Gather into (B, C, max_cat)
        gathered_logits = x_pred_cat[:, self.gather_indices].view(B, self.C, self.max_cat)
        gathered_soft = soft_cat[:, self.gather_indices].view(B, self.C, self.max_cat)

        pad_mask_ext = self.pad_mask.unsqueeze(0)
        
        # Logits -> log_probs
        gathered_logits = gathered_logits.masked_fill(~pad_mask_ext, -float('inf'))
        log_probs = F.log_softmax(gathered_logits, dim=-1)
        log_probs = log_probs.masked_fill(~pad_mask_ext, 0.0)

        # ST Estimator for Softmax Targets (vectorized over C groups)
        gathered_soft_masked = gathered_soft.masked_fill(~pad_mask_ext, -float('inf'))
        hard_cat = torch.zeros_like(gathered_soft)
        hard_cat.scatter_(-1, gathered_soft_masked.argmax(dim=-1, keepdim=True), 1.0)
        
        cat_target = hard_cat + gathered_soft - gathered_soft.detach()
        # Clean up padded targets just in case
        cat_target = cat_target.masked_fill(~pad_mask_ext, 0.0)

        per_sample_feature_ce = -torch.sum(cat_target * log_probs, dim=-1)  # (B, C)
        weighted_ce = per_sample_feature_ce * st_mask * w_t  # (B, C)

        cat_loss = weighted_ce.mean()

        return cont_loss + cat_loss

    def compute_latent_prior_loss(self, dummy_latents):
        """Mask <-> t coupling via rank-matched symmetric Bernoulli KL.

        Per sample b: p_b = mean_c sigmoid(logits[b,c]), gamma_b = 1 - alpha_bar(t_b).
        The two 1-D batch distributions are sorted ascending and matched by rank
        before the KL — this removes the slot-permutation ambiguity of grad-match
        (which sees only batch-summed gradients, so slot b's mask and slot b's t
        need not refer to the same true row). Under perfect per-slot alignment the
        sorted matching reduces to the per-slot KL; under permutation ambiguity it
        gives a canonical 1-D Wasserstein-like alignment grad-match cannot disturb.
        """
        if getattr(self.args, "no_prior", False):
            return torch.zeros((), device=self.device)
        t = dummy_latents["t"]
        # Let gradient flow into both t and mask_logits: the rank-sorted KL acts as a
        # batch-distribution constraint on p_b (pulls mask density toward gamma(t)).
        mask_logits = dummy_latents["mask_logits"]  # (B, n_cats)

        if self._cached_alpha_bar is not None:
            alpha_bar = self._cached_alpha_bar
        else:
            alpha_bar = self._interp_timewise(self.alpha_bars, t)
        eps = 1e-6
        gamma_b = (1 - alpha_bar).clamp(eps, 1 - eps)  # (B,)
        p_b = torch.sigmoid(mask_logits).mean(dim=1).clamp(eps, 1 - eps)  # (B,)

        # Rank-align the two batch distributions. sort preserves gradients through
        # the permuted values, so sorted_q still differentiates into t_b.
        # By the 1-D rearrangement inequality this is equivalent to 1-D optimal-transport (Wasserstein-2-like) matching between the two 1-D distributions.
        sorted_p, _ = torch.sort(p_b)
        sorted_q, _ = torch.sort(gamma_b)

        # Symmetric Bernoulli KL on the aligned pairs (zero iff the two sorted
        # batch distributions coincide).
        logit_p = torch.log(sorted_p) - torch.log1p(-sorted_p)
        logit_q = torch.log(sorted_q) - torch.log1p(-sorted_q)
        kl = ((sorted_p - sorted_q) * (logit_p - logit_q)).mean()

        idx = min(self._prior_iter, self._prior_schedule.shape[0] - 1)
        weight = self._prior_schedule[idx]
        return kl * weight

    def get_debug_stats(self, dummy_latents) -> dict[str, float]:
        with torch.no_grad():
            mask_logits = dummy_latents["mask_logits"]
            soft_mask = torch.sigmoid(mask_logits / max(self.mask_temperature, 1e-6))
            hard_mask = (soft_mask > 0.5).float()
            t = dummy_latents["t"].float()
            alpha_bar = self._interp_timewise(self.alpha_bars, t)
            eps = 1e-6
            gamma_b = (1 - alpha_bar).clamp(eps, 1 - eps)
            p_b = torch.sigmoid(mask_logits).mean(dim=1).clamp(eps, 1 - eps)
            logit_p = torch.log(p_b) - torch.log1p(-p_b)
            logit_q = torch.log(gamma_b) - torch.log1p(-gamma_b)
            kl_perslot = ((p_b - gamma_b) * (logit_p - logit_q)).mean().item()
            sorted_p, _ = torch.sort(p_b)
            sorted_q, _ = torch.sort(gamma_b)
            sl_logit_p = torch.log(sorted_p) - torch.log1p(-sorted_p)
            sl_logit_q = torch.log(sorted_q) - torch.log1p(-sorted_q)
            kl_sym = ((sorted_p - sorted_q) * (sl_logit_p - sl_logit_q)).mean().item()
            if self._prior_schedule is not None:
                sched_idx = min(self._prior_iter, self._prior_schedule.shape[0] - 1)
                scheduled_weight = float(self._prior_schedule[sched_idx].item())
            else:
                scheduled_weight = 0.0
            return {
                "prior_iter": int(self._prior_iter),
                "prior_weight": scheduled_weight,
                "mask_temperature": float(self.mask_temperature),
                "mask_density_soft": soft_mask.mean().item(),
                "mask_density_hard": hard_mask.mean().item(),
                "p_mean": p_b.mean().item(),
                "p_min": p_b.min().item(),
                "p_max": p_b.max().item(),
                "q_min": gamma_b.min().item(),
                "q_max": gamma_b.max().item(),
                "kl_sym": kl_sym,
                "kl_perslot": kl_perslot,
                "gamma_t_mean": gamma_b.mean().item(),
                "dummy_t_mean": t.mean().item(),
                "dummy_t_std": t.std().item() if t.numel() > 1 else 0.0,
            }

    # --- Utility ---

    def get_model_dims(self, original_dim):
        return original_dim + len(self.n_cats), original_dim

    def get_optimizable_flags(self, args):
        return {"t": self.optimize_t, "mask": self.optimize_mask, "noise": self.optimize_noise}

    def evaluate_latent_reconstruction(self, dummy_latents, real_latents, indices=None):
        metrics = {}
        real_t = real_latents["t"].cpu().float()
        pred_t = dummy_latents["t"].detach().cpu().float()
        metrics["t_mae"] = (real_t - pred_t).abs().mean().item()

        real_mask = real_latents["mask"].cpu()
        # evaluate what is fully returned at the end of the attack, not the underlying mask_logits which may be at an intermediate temperature
        pred_mask = (
            torch.sigmoid(dummy_latents["mask_logits"].detach().cpu() / max(self.mask_temperature, 1e-6)) > 0.5
        ).float()
        metrics["mask_accuracy"] = (pred_mask == real_mask).float().mean().item()

        real_noise = real_latents["noise"].cpu()
        pred_noise = dummy_latents["noise"].detach().cpu()
        metrics["noise_mse"] = (real_noise - pred_noise).pow(2).mean().item()

        if indices is not None:
            aligned_dummy_latents = {k: v[indices] for k, v in dummy_latents.items()}
            pred_t_aligned = aligned_dummy_latents["t"].detach().cpu().float()
            pred_mask_aligned = (
                torch.sigmoid(
                    aligned_dummy_latents["mask_logits"].detach().cpu() / max(self.mask_temperature, 1e-6)
                )
                > 0.5
            ).float()
            pred_noise_aligned = aligned_dummy_latents["noise"].detach().cpu()
            metrics["t_mae_aligned"] = (real_t - pred_t_aligned).abs().mean().item()
            metrics["mask_accuracy_aligned"] = (pred_mask_aligned == real_mask).float().mean().item()
            metrics["noise_mse_aligned"] = (real_noise - pred_noise_aligned).pow(2).mean().item()
        return metrics

    def get_all_optimizable_params(self, dummy_latents):
        """Return list of dummy latent tensors that should be optimized (for full-dataset shared optimizer)."""
        params = []
        if self.optimize_t:
            params.append(dummy_latents["t"])
        if self.optimize_mask:
            params.append(dummy_latents["mask_logits"])
        if self.optimize_noise:  # same flag as mask in practice
            params.append(dummy_latents["noise"])
        return params

    def restore_known_t_only(self, dummy_latents, nan_masks, starting_values):
        dummy_t = dummy_latents["t"]
        with torch.no_grad():
            dummy_t.clamp_(0, self.T - 1)
            dummy_t[~nan_masks["t"]] = starting_values["t"][~nan_masks["t"]]
        dummy_t.requires_grad_(True)

    def restore_known_mask_only(self, dummy_latents, nan_masks, starting_values):
        mask_logits = dummy_latents["mask_logits"]
        with torch.no_grad():
            mask_logits[~nan_masks["mask"]] = starting_values["mask_logits"][~nan_masks["mask"]]
        mask_logits.requires_grad_(True)

    def print_debug(self, dummy_latents, nan_masks, label=""):
        """Print recovered mask and timestep values for debugging (tqdm-safe)."""
        from tqdm import tqdm as _tqdm

        prefix = f"[masked debug{' ' + label if label else ''}]"
        with torch.no_grad():
            t = dummy_latents["t"].detach()
            mask_logits = dummy_latents["mask_logits"].detach()
            soft_mask = torch.sigmoid(mask_logits / max(self.mask_temperature, 1e-6)).int()  # harden for display

            t_known = ~nan_masks["t"]
            t_unknown = nan_masks["t"]

            lines = [f"{prefix} temperature={self.mask_temperature:.2e}"]

            # Timesteps
            t_vals = t.cpu().numpy()
            if t_unknown.any():
                t_opt = t_vals[t_unknown.cpu().numpy()]
                lines.append(f"{prefix} t (optimized, {t_opt.shape[0]} samples): {t_opt.round().astype(int).tolist()}")
            if t_known.any():
                t_fix = t_vals[t_known.cpu().numpy()]
                lines.append(f"{prefix} t (known,     {t_fix.shape[0]} samples): {t_fix.round().astype(int).tolist()}")

            # Mask: show the mask for each cell
            lines.append(f"{prefix} mask (soft): {soft_mask.cpu().numpy().tolist()}")

            for line in lines:
                _tqdm.write(line)

    def cleanup_dummy_latents(self, dummy_latents):
        """Free dummy latents and cached attack tensors."""
        for v in dummy_latents.values():
            del v
        self._cached_st_mask = None
        self._cached_dummy_data = None
        self._cached_alpha_bar = None


def create_diffusion_process(args, device, feat_info=None):
    if args.diffusion_type == "gaussian":
        return GaussianDiffusionProcess(args, device)
    elif args.diffusion_type == "masked":
        if feat_info is None:
            raise ValueError(
                f"Please specify the continuous and categorical column cardinalities when using masked diffusion"
            )
        else:
            return MaskedDiffusionProcess(args, device, feat_info)
    else:
        raise ValueError(f"Unknown diffusion type: {args.diffusion_type}")
