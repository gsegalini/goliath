import csv
import os
import time
import numpy as np
import torch.optim
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from .utils import Matcher
from .diffusion_process import MaskedDiffusionProcess, GaussianDiffusionProcess
from .model import build_diffusion_model, diffusion_model_config
import torch.nn.functional as F
from copy import deepcopy


class Attack:

    def __init__(self, args, prepper, diffusion_process):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.N = args.pool_size
        self.matcher = Matcher(prepper.cat_cols, prepper.num_cols)
        self.matcher.update_numeric_std(prepper.numeric_std)  # numeric tolerances assume the per-column std is known

        self.prepper = prepper
        self.diffusion_process = diffusion_process

        self.previous_ix = -1  # start at epoch 0
        self.size = -1

        self.epoch_ix = -1

        self.latest_reconstructions = []
        self.alignment_anchors = []  # used for alignment across epochs
        self.latest_latents = []  # warm-start latents when init_strategy resolves to "previous"
        self.epoch_lr = args.attack_epoch_lr
        # Blend weight decays linearly from attack_epoch_lr to 0 over args.epochs.
        self.epoch_lr_schedule = self.epoch_schedule(args.attack_epoch_lr, args.epochs, args.attack_epoch_lr_schedule)

        self.loss_fn = getattr(args, 'loss_fn', 'both')

        self._debug_prior = getattr(args, 'debug_prior', False)
        self._debug_trace_counter = 0
        self._debug_trace_stride = 25

        # `both` loss: r = n_num / total_dims weights MSE (numerical) vs cossim (categorical).
        # EMA-tracked scales keep both terms O(1) so neither dominates during optimization.
        n_num = len(prepper.num_cols)
        n_cat_dims = sum(len(c) for c in prepper.OneHotEncoder.categories_)
        C_mask = self.diffusion_process.C if isinstance(self.diffusion_process, MaskedDiffusionProcess) else 0
        self._n_cont = n_num
        self._feature_ratio = n_num / max(n_num + n_cat_dims + C_mask, 1)
        self._loss_scale_beta = 0.9
        self._mse_scale = None
        self._cossim_scale = None
        self._last_mse = None
        self._last_direction = None
        # Live scaled terms stashed for --debug-prior per-term autograd.grad logging.
        self._last_mse_term_live = None
        self._last_cossim_term_live = None

        # `double-cossim`: two cosine terms on disjoint parameter blocks.
        # Cont block = rows [0:n_cont] of the backbone's per-feature block (MLP: the output
        # linear's weight + bias; FT-Transformer: the feature tokenizer); rest = everything else.
        # alpha weights the cont block vs the rest block in the summed loss.
        self._double_cossim_alpha = getattr(args, "double_cossim_alpha", 0.5)
        # FT-Transformer closed-form tokenizer inversion (see _analytic_tokenizer_xt). Two
        # independent uses: seed the optimizer at the solution, and/or hold it there with a
        # penalty. Seeding is the cheaper of the two -- it costs one dot product, not a term
        # evaluated every iteration.
        self._ft_analytic_weight = getattr(args, "ft_analytic_weight", 0.0)
        self._ft_analytic_init = getattr(args, "ft_analytic_init", False)
        self._tokenwise_slice = None   # (feat_weight span, feat_bias span, n_feat, d_model)
        self._tokenwise_real = None
        self._cont_block_mask = None  # bool mask, built lazily on first _single_inversion call
        # Precomputed masking tensors for double-cossim (avoid boolean-index gather under
        # create_graph=True — gather backward is a scatter, expensive to double-backprop).
        self._dc_mf = None          # mask.float(), full-size; cached after first build
        self._dc_mf_inv = None      # 1 - mf, full-size; cached after first build
        self._dc_g_c_unit = None    # (grad_flat * mf) / ||grad_flat * mf||, per-inversion
        self._dc_g_r_unit = None    # (grad_flat * mf_inv) / ||grad_flat * mf_inv||, per-inversion

        # `auto`: schema-aware selection.
        # - n_cat_dims <= 3: cossim is near-degenerate at low cardinality; use pure MSE.
        # - FT-Transformer or masked diffusion on cat-column-dominant schemas: `both`'s
        #   global terms let the large categorical/downstream blocks swamp the continuous
        #   feature rows. Split via double-cossim, weighting the continuous block by its
        #   fraction of the logical schema columns.
        # - otherwise: `both`.
        # All explicit loss_fn choices pass through untouched.
        n_cat_cols = len(prepper.cat_cols)
        is_masked = isinstance(self.diffusion_process, MaskedDiffusionProcess)
        is_ft_transformer = getattr(args, "backbone", "mlp") == "ft-transformer"
        if self.loss_fn == "auto":
            if n_cat_dims <= 3:
                self._effective_loss_fn = "mse"
            elif (is_ft_transformer or is_masked) and n_cat_cols > n_num:
                self._effective_loss_fn = "double-cossim"
                self._double_cossim_alpha = n_num / (n_num + n_cat_cols)
            else:
                self._effective_loss_fn = "both"
        elif self.loss_fn == "double-cossim":
            # Degenerate if no continuous features: fall back to plain cossim on full vector.
            self._effective_loss_fn = "double-cossim" if n_num > 0 else "cossim"
        else:
            self._effective_loss_fn = self.loss_fn

        # double-cossim needs a parameter block whose rows are indexed by feature. Both supported
        # backbones have one (MLP: the output layer, via model.dim_t; FT-Transformer: the feature
        # tokenizer, via model.feat_weight) and _build_cont_block_mask dispatches on which. Any
        # other backbone falls back to the architecture-agnostic feature-ratio-weighted `both`.
        if (
            getattr(args, "backbone", "mlp") not in ("mlp", "ft-transformer")
            and self._effective_loss_fn == "double-cossim"
        ):
            self._effective_loss_fn = "both"

        # Baseline guard. The TabLeak baseline (including the "adaptive" variant that
        # optimizes t/noise) must not use GOLIATH's table-aware EMA loss or the mask<->t
        # coupling prior. Neither is gated on t/noise optimization, so block them here
        # rather than rely on configuration. (Cross-epoch ensembling needs no guard: train_diffusion.py breaks the
        # tableak run after a single epoch, so it can never execute regardless of --epochs.)
        if getattr(args, "baseline", None) == "tableak":
            if self.loss_fn not in ("cossim", "mse"):
                raise ValueError(
                    f"baseline='tableak' requires loss_fn in {{cossim, mse}} (got '{self.loss_fn}'): "
                    "the table-aware loss (auto/both/double-cossim) is a GOLIATH-only component."
                )
            if is_masked and not getattr(args, "no_prior", False):
                raise ValueError(
                    "baseline='tableak' with masked diffusion requires no_prior=True: "
                    "the mask<->t coupling prior is a GOLIATH-only component."
                )

    @staticmethod
    def epoch_schedule(start_lr, epochs, strategy="linear"):
        if strategy == "linear":
            return np.linspace(start_lr, 0.0, epochs)
        else:
            raise ValueError(f"Unknown epoch lr schedule strategy: {strategy}")

    def _resolve_init_strategy(self):
        """Resolve dummy-data init strategy.

        Precedence: explicit --init-strategy wins; otherwise falls back to "previous"
        when --warm-data is set and a prior epoch exists, else "uniform".
        """
        # Under order-invariant global ensembling the batch slot no longer maps to fixed rows, so a
        # per-slot data warm-start ("previous") would seed from the wrong rows: force fresh init.
        if getattr(self.args, "global_ensemble", False):
            return "uniform"
        if self.args.init_strategy is not None:
            if self.args.init_strategy == "previous" and self.size <= 0:
                return "uniform"
            return self.args.init_strategy
        if getattr(self.args, "warm_data", False) and self.size > 0:
            return "previous"
        return "uniform"

    def _initialise_dummy_data(self, shape, n_cats, strategy="uniform", previous_ix=None):
        # generate shape - n_cats values and concatenate uniform 0-1 for categorical
        shape_cont = (shape[0], shape[1] - n_cats)
        if strategy == "uniform":
            dummy_data = (2 * (torch.rand(shape_cont, dtype=torch.float32) - 0.5)).to(self.device).requires_grad_(True)
        elif strategy == "normal":
            dummy_data = torch.normal(0, 1, size=shape_cont, dtype=torch.float32).to(self.device).requires_grad_(True)
        elif strategy == "previous":
            source_ix = self.previous_ix if previous_ix is None else previous_ix
            dummy_data = (
                self.latest_reconstructions[source_ix]
                .detach()
                .clone()
                .to(self.device)
                .requires_grad_(True)
            )
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        if n_cats > 0 and strategy != "previous":
            dummy_cats = torch.rand((shape[0], n_cats), dtype=torch.float32, requires_grad=True, device=self.device)
            dummy_data = torch.cat((dummy_data, dummy_cats), dim=1)
        return nn.Parameter(dummy_data, requires_grad=True).to(self.device)

    def new_epoch(self):
        self.size = len(self.latest_reconstructions)
        self.previous_ix = 0
        self.epoch_ix += 1
        if self.args.verbose:
            tqdm.write(f"Attack: Epoch concluded with size {self.size}")

    def fast_align(self, existing, new_one, ground_truth=None, existing_df=None, new_one_df=None):
        """Greedy O(N^2) row alignment: repeatedly pick the global max in the similarity matrix."""
        # Use pre-decoded DataFrames if provided, otherwise decode. Alignment is non-differentiable
        # (greedy match on decoded values), so detach before ->numpy; the inputs may still track grad.
        if existing_df is None:
            existing_df = self.prepper.destandardize_np(existing.detach().cpu().numpy())
            existing_df = self.prepper.decodeNpToDf(existing_df)
        if new_one_df is None:
            new_one_df = self.prepper.destandardize_np(new_one.detach().cpu().numpy())
            new_one_df = self.prepper.decodeNpToDf(new_one_df)
        # Similarity = 1 - TableLeak cost.
        cost = self.matcher.cost_matrix(existing_df, new_one_df)
        sim = torch.tensor(1 - cost, dtype=torch.float32).to(self.device)

        N = sim.shape[0]
        matches = torch.zeros(N, dtype=torch.long, device=self.device)
        matched_similarities = torch.zeros(N, device=self.device)

        # Use a mask to prevent duplicate matches
        mask_val = -float("inf")
        for _ in range(N):
            # Find the global max
            idx = torch.argmax(sim)
            i, j = divmod(idx.item(), N)

            matches[i] = j
            matched_similarities[i] = sim[i, j]

            # Mask out the selected row and column
            sim[i, :] = mask_val
            sim[:, j] = mask_val
 
        return matches.cpu(), matched_similarities.cpu()

    def _match_rows_to_pool(self, new_one, pool):
        """Order-invariant cross-epoch matching. Assign each of new_one's B reconstructed rows to a
        DISTINCT row of the global pool (all accumulated reconstructions) by TableLeak similarity.
        Returns length-B `pool_idx` where pool_idx[i] is the pool row assigned to new_one[i].
        Uses optimal Hungarian if `no_fast_align` else the greedy O(N*B) heuristic. This is the
        matching that makes the ensemble robust to minibatch reshuffling; it operates over the full
        pool (large N), which is exactly where greedy and Hungarian diverge most.
        """
        new_df = self.prepper.decodeNpToDf(self.prepper.destandardize_np(new_one.detach().cpu().numpy()))
        pool_df = self.prepper.decodeNpToDf(self.prepper.destandardize_np(pool.detach().cpu().numpy()))
        cost = np.asarray(self.matcher.cost_matrix(pool_df, new_df), dtype=np.float64)  # (N_pool, B)
        N, B = cost.shape
        pool_idx = np.empty(B, dtype=int)
        if getattr(self.args, "no_fast_align", False):
            row_ind, col_ind = linear_sum_assignment(cost)  # rectangular -> B pairs
            pool_idx[col_ind] = row_ind
        else:
            c = cost.copy()
            for _ in range(B):
                k = int(np.argmin(c))
                i, j = divmod(k, B)  # i = pool row, j = new row
                pool_idx[j] = i
                c[i, :] = np.inf
                c[:, j] = np.inf
        return pool_idx

    def _ensemble_batch_global(self, new_one, latents=None):
        """Order-invariant ensembling: blend this batch's rows into the globally-matched pool
        entries instead of a fixed batch slot. Robust to reshuffling."""
        new_one = new_one.detach().cpu()
        B = new_one.shape[0]
        sizes = [r.shape[0] for r in self.latest_reconstructions]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        pool = torch.cat([r.detach().cpu() for r in self.latest_reconstructions], dim=0)
        pool_idx = self._match_rows_to_pool(new_one, pool)
        epoch_lr = self.epoch_lr_schedule[self.epoch_ix]
        # ensure stored tensors are detached (a prior warm-start may have set requires_grad in place)
        self.latest_reconstructions = [
            r.detach().cpu() if torch.is_tensor(r) else r for r in self.latest_reconstructions
        ]
        blended = torch.empty_like(new_one)
        for i in range(B):
            p = int(pool_idx[i])
            s = int(np.searchsorted(offsets, p, side="right") - 1)
            o = p - int(offsets[s])
            existing = self.latest_reconstructions[s][o]
            val = epoch_lr * new_one[i] + (1 - epoch_lr) * existing
            self.latest_reconstructions[s][o] = val
            blended[i] = val
        self.previous_ix += 1
        if latents is not None:
            return blended, {k: v.detach().cpu() for k, v in latents.items()}
        return blended

    def ensemble_batch(self, new_one, ground_truth=None, latents=None):
        if self.size == -1 or self.previous_ix >= self.size or self.previous_ix < 0:
            # did not make an epoch yet or out of bounds
            return (new_one, latents) if latents is not None else new_one
        elif getattr(self.args, "global_ensemble", False):
            # order-invariant path (reshuffle-robust)
            return self._ensemble_batch_global(new_one, latents=latents)
        else:
            # get previous reconstruction and ensemble. Ensembling is non-differentiable
            # post-processing (align + blend of finished reconstructions), so detach defensively.
            new_one = new_one.detach() if torch.is_tensor(new_one) else new_one
            existing = self.latest_reconstructions[self.previous_ix]
            existing = existing.detach() if torch.is_tensor(existing) else existing
            align_target = self.alignment_anchors[self.previous_ix]
            align_target = align_target.detach() if torch.is_tensor(align_target) else align_target
            if self.args.verbose:
                tqdm.write(f"Ensembling batch {self.previous_ix + 1}/{self.size}")

            if self.args.no_align:
                aligned_index_order = torch.arange(new_one.shape[0])
            elif self.args.no_fast_align:
                inverted_ex = self.prepper.destandardize_np(align_target)
                inverted_ex = self.prepper.decodeNpToDf(inverted_ex)
                inverted_new = self.prepper.destandardize_np(new_one)
                inverted_new = self.prepper.decodeNpToDf(inverted_new)
                _, new_indices = self.matcher.row_matching(inverted_ex, inverted_new)
                aligned_index_order = new_indices
            else:
                aligned_index_order, _ = self.fast_align(align_target, new_one, ground_truth=None)
            new_one_aligned = new_one[aligned_index_order]
            epoch_lr = self.epoch_lr_schedule[self.epoch_ix]
            final_recon = epoch_lr * new_one_aligned + (1 - epoch_lr) * existing
            self.latest_reconstructions[self.previous_ix] = final_recon.cpu()
            if self.args.anchor == "latest":
                self.alignment_anchors[self.previous_ix] = final_recon.cpu()
            # Keep per-row latents in the same row order as the aligned data so
            # they can be warm-started next epoch without a data<->latent mismatch.
            aligned_latents = None
            if latents is not None:
                order = torch.as_tensor(aligned_index_order).cpu()
                aligned_latents = {k: v.cpu()[order] for k, v in latents.items()}
            self.previous_ix += 1
            if latents is not None:
                return final_recon, aligned_latents
            return final_recon

    def softmax_cat(self, data, n_cats_amounts, temperature=1.0):
        # apply softmax to categorical parts of data
        start_ix = data.shape[1] - sum(n_cats_amounts)
        data = data.clone()
        for n_cats in n_cats_amounts:
            end_ix = start_ix + n_cats
            data[:, start_ix:end_ix] = F.softmax(data[:, start_ix:end_ix] / temperature, dim=-1)
            start_ix = end_ix
        return data

    def _write_prior_trace(self, rows):
        save_dir = os.path.join(self.args.save_dir, "prior_traces")
        os.makedirs(save_dir, exist_ok=True)
        idx = self._debug_trace_counter
        self._debug_trace_counter += 1
        path = os.path.join(
            save_dir,
            f"trace_seed{self.args.seed}_e{self.epoch_ix}_call{idx:04d}.csv",
        )
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _compute_gradient_matching_loss(self, grad_flat, pred_flat):
        """Gradient-matching loss selected by self._effective_loss_fn.

        Side effect: when --debug-prior is on, stashes live (graph-attached) scaled
        sub-terms in self._last_{mse,cossim}_term_live so the caller can autograd.grad
        each component w.r.t. dummy_data for per-term gradient-norm logging.
        """
        self._last_mse_term_live = None
        self._last_cossim_term_live = None
        if self._effective_loss_fn == "mse":
            mse = torch.mean((grad_flat - pred_flat) ** 2)
            self._last_mse = mse.detach()
            self._last_direction = None
            if self._debug_prior:
                self._last_mse_term_live = mse
            return mse
        if self._effective_loss_fn == "tokenwise":
            # One cosine per feature over its own tokenizer block, then averaged: every column
            # gets equal weight instead of the largest-magnitude ones setting the direction.
            pred_blocks = self._tokenwise_blocks(pred_flat)
            per_feat = F.cosine_similarity(self._tokenwise_real, pred_blocks, dim=1, eps=1e-8)
            total = 1.0 - per_feat.mean()
            self._last_mse = None
            self._last_direction = total.detach()
            if self._debug_prior:
                self._last_cossim_term_live = total
            return total
        if self._effective_loss_fn == "double-cossim":
            # Pointwise formulation — no gather/scatter under create_graph=True.
            # cos(g_mask, p_mask) = sum(p * g_unit_mask) / ||p * mf||  where
            # g_unit_mask = (g * mf) / ||g * mf|| is precomputed (grad_flat detached).
            eps = 1e-8
            mf = self._dc_mf
            mf_inv = self._dc_mf_inv
            g_c_unit = self._dc_g_c_unit
            g_r_unit = self._dc_g_r_unit
            p_sq = pred_flat * pred_flat
            norm_p_c = (p_sq * mf).sum().clamp(min=eps * eps).sqrt()
            norm_p_r = (p_sq * mf_inv).sum().clamp(min=eps * eps).sqrt()
            dot_c = (pred_flat * g_c_unit).sum()
            dot_r = (pred_flat * g_r_unit).sum()
            dir_cont = 1.0 - dot_c / norm_p_c.clamp(min=eps)
            dir_rest = 1.0 - dot_r / norm_p_r.clamp(min=eps)
            alpha = self._double_cossim_alpha
            total = alpha * dir_cont + (1.0 - alpha) * dir_rest
            self._last_mse = None
            self._last_direction = total.detach()
            if self._debug_prior:
                # Repurpose the two live slots so per-term grad-norm logging still works.
                self._last_mse_term_live = alpha * dir_cont
                self._last_cossim_term_live = (1.0 - alpha) * dir_rest
            return total
        direction = 1.0 - F.cosine_similarity(grad_flat, pred_flat, dim=0, eps=1e-8)
        if self._effective_loss_fn == "cossim":
            self._last_mse = None
            self._last_direction = direction.detach()
            if self._debug_prior:
                self._last_cossim_term_live = direction
            return direction
        # "both": feature-ratio weighted; EMA scales keep both terms O(1).
        mse = torch.mean((grad_flat - pred_flat) ** 2)
        if self._mse_scale is None:
            self._mse_scale = mse.detach().clamp(min=1e-12)
            self._cossim_scale = direction.detach().clamp(min=1e-12)
        else:
            beta = self._loss_scale_beta
            self._mse_scale = (beta * self._mse_scale + (1.0 - beta) * mse.detach()).clamp(min=1e-12)
            self._cossim_scale = (beta * self._cossim_scale + (1.0 - beta) * direction.detach()).clamp(min=1e-12)
        self._last_mse = mse.detach()
        self._last_direction = direction.detach()
        r = self._feature_ratio
        mse_term = r * (mse / self._mse_scale)
        cossim_term = (1.0 - r) * (direction / self._cossim_scale)
        if self._debug_prior:
            self._last_mse_term_live = mse_term
            self._last_cossim_term_live = cossim_term
        return mse_term + cossim_term

    def _build_cont_block_mask(self, model, total_size, device):
        """Mark the flat-gradient positions of the per-feature block's continuous rows.

        Every backbone has one parameter block whose rows are indexed by feature, so the
        continuous columns occupy a contiguous, un-mixed slice of it. Which block that is
        depends on the architecture; `_build_cont_block_mask_tokenizer` handles the
        FT-Transformer, the code below handles the MLP.
        """
        if hasattr(model, "feat_weight"):
            return self._build_cont_block_mask_tokenizer(model, total_size, device)
        return self._build_cont_block_mask_mlp(model, total_size, device)

    def _build_cont_block_mask_tokenizer(self, model, total_size, device):
        """FT-Transformer: the feature tokenizer (`feat_weight`/`feat_bias`, both `(d_in, d_model)`).

        Row `i` is feature `i`'s embedding vector, so `dL/dfeat_weight[i] = sum_b g[b,i] * x[b,i]`
        carries that feature's input value directly. This is the transformer's least-entangled
        block: self-attention mixes every token downstream and the head `Linear(d_model, 1)` is
        shared across tokens, so no later block is per-feature. Rows are ordered
        [continuous..., one-hot categorical...] (Preprocessor.encodeDfToNp), plus mask tokens
        appended last under masked diffusion, so the continuous rows are the leading `n_cont`.
        """
        d_model = model.d_model
        target_shape = (model.feat_weight.shape[0], d_model)
        offset = 0
        slices = []
        for name, p in model.named_parameters():
            if name in ("feat_weight", "feat_bias") and tuple(p.shape) == target_shape:
                slices.append((offset, offset + self._n_cont * d_model))
            offset += p.numel()
        assert offset == total_size, f"Flat-grad size mismatch: walked {offset}, expected {total_size}"
        assert len(slices) == 2, f"Could not locate feature tokenizer (expected 2 blocks, got {len(slices)})"
        mask = torch.zeros(total_size, dtype=torch.bool, device=device)
        for start, end in slices:
            mask[start:end] = True
        assert int(mask.sum().item()) == 2 * self._n_cont * d_model
        return mask

    def _build_cont_block_mask_mlp(self, model, total_size, device):
        """MLP: locate the last `nn.Linear` (the model output layer) and mark flat-gradient
        positions corresponding to its first `n_cont` output rows (weight) and
        first `n_cont` bias entries. Uses `named_parameters()` iteration order, which
        matches `torch.autograd.grad(..., model.parameters())`'s output order.
        """
        d_out = model.d_out
        dim_t = model.dim_t
        target_w_shape = (d_out, dim_t)
        target_b_shape = (d_out,)
        offset = 0
        w_slice = None  # (offset_start, n_cont_rows_count)
        b_slice = None  # (offset_start, n_cont_entries_count)
        for _name, p in model.named_parameters():
            shape = tuple(p.shape)
            n = p.numel()
            if shape == target_w_shape:
                w_slice = (offset, offset + self._n_cont * dim_t)
            elif shape == target_b_shape and w_slice is not None and b_slice is None:
                b_slice = (offset, offset + self._n_cont)
            offset += n
        assert offset == total_size, f"Flat-grad size mismatch: walked {offset}, expected {total_size}"
        assert w_slice is not None and b_slice is not None, (
            f"Could not locate output layer (d_out={d_out}, dim_t={dim_t})"
        )
        mask = torch.zeros(total_size, dtype=torch.bool, device=device)
        mask[w_slice[0]:w_slice[1]] = True
        mask[b_slice[0]:b_slice[1]] = True
        assert int(mask.sum().item()) == self._n_cont * (dim_t + 1)
        return mask

    def _prepare_tokenwise(self, grad_flat, model):
        """Cache the per-feature view of the FT-Transformer tokenizer gradient.

        The tokenizer is the one block with a genuine per-feature decomposition: feature i owns
        `feat_weight[i]` and `feat_bias[i]`, 2*d_model numbers. A single global cosine over the
        whole flattened gradient lets whichever features carry the largest gradient magnitude
        dominate -- severe on tabular data, where a handful of one-hot columns and a few
        wide-range continuous columns swamp the rest. Scoring each feature's own block separately
        and averaging gives every column equal say.
        """
        if self._effective_loss_fn != "tokenwise":
            return
        if self._tokenwise_slice is None:
            if not hasattr(model, "feat_weight"):
                raise ValueError("loss_fn='tokenwise' requires a backbone with a feature tokenizer")
            d_model = model.d_model
            n_feat = model.feat_weight.shape[0]
            offset, spans = 0, {}
            for name, p in model.named_parameters():
                if name in ("feat_weight", "feat_bias"):
                    spans[name] = (offset, offset + p.numel())
                offset += p.numel()
            assert len(spans) == 2, "could not locate the feature tokenizer"
            self._tokenwise_slice = (spans["feat_weight"], spans["feat_bias"], n_feat, d_model)
        # Observed side depends on the batch, so refresh it every inversion.
        self._tokenwise_real = self._tokenwise_blocks(grad_flat).detach()

    def _tokenwise_blocks(self, flat):
        """(n_feat, 2*d_model) -- each row is one feature's tokenizer gradient."""
        (ws, we), (bs, be), n_feat, d_model = self._tokenwise_slice
        w = flat[ws:we].reshape(n_feat, d_model)
        b = flat[bs:be].reshape(n_feat, d_model)
        return torch.cat([w, b], dim=1)

    def _analytic_tokenizer_xt(self, grad, model, batch_size):
        """Closed-form inversion of the FT-Transformer feature tokenizer.

        The tokenizer computes token_i = x_i * W_i + b_i, so for a SINGLE row the two parameter
        gradients share the same upstream term g_i:
            dL/dW_i = g_i * x_i        dL/db_i = g_i
        hence dL/dW_i = x_i * dL/db_i exactly, whatever g_i happens to be, and
            x_i = <dL/dW_i, dL/db_i> / ||dL/db_i||^2
        recovers the model's input (the NOISED row x_t) to machine precision -- no optimization.
        This is the transformer analogue of the first-layer leakage TabLeak exploits on MLPs.

        Only exact at batch_size == 1: for larger batches both gradients are sums over rows and
        the closed form collapses to a g-weighted average, so we return None rather than impose a
        constraint that is quietly wrong.
        """
        if batch_size != 1 or not hasattr(model, "feat_weight"):
            return None
        names = [n for n, _ in model.named_parameters()]
        try:
            iw, ib = names.index("feat_weight"), names.index("feat_bias")
        except ValueError:
            return None
        dW, db = grad[iw], grad[ib]                      # (d_in, d_model) each
        denom = (db * db).sum(-1)
        xhat = (dW * db).sum(-1) / denom.clamp(min=1e-30)
        # Features whose token gradient vanished carry no signal; leave them unconstrained.
        valid = denom > 1e-20
        if not bool(valid.any()):
            return None
        return xhat.detach().unsqueeze(0), valid.unsqueeze(0)

    def _seed_dummy_from_analytic(self, dummy_data, analytic_xt, dummy_latents, n_cont):
        """Start the optimization AT the closed-form solution instead of at random.

        `_analytic_tokenizer_xt` recovers x_t (the noised row) exactly, but `dummy_data` holds the
        CLEAN row, so invert the forward diffusion with the current latent guesses:
            x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        Only the continuous block is seeded: the categorical entries of `dummy_data` are logits fed
        through softmax(./T) with T annealed from ~1000, so at the start they are near-uniform
        regardless of their value and seeding them would have no effect.

        Gaussian diffusion only -- masked diffusion has a different forward process.
        """
        if analytic_xt is None or not isinstance(self.diffusion_process, GaussianDiffusionProcess):
            return
        if n_cont <= 0:
            return
        xt, valid = analytic_xt
        with torch.no_grad():
            ab = self.diffusion_process._interp_timewise(
                self.diffusion_process.alpha_bars, dummy_latents["t"].detach().float(),
            ).reshape(-1, 1)
            x0 = (xt - torch.sqrt((1 - ab).clamp(min=1e-12)) * dummy_latents["noise"].detach()) / torch.sqrt(
                ab.clamp(min=1e-12)
            )
            m = valid[:, :n_cont]
            dummy_data[:, :n_cont][m] = x0[:, :n_cont][m].to(dummy_data.dtype)

    def _prepare_double_cossim(self, grad_flat, model):
        """Precompute the continuous/rest split the double-cossim loss reads.

        `_compute_gradient_matching_loss` consumes `_dc_mf`/`_dc_g_*_unit`, so every inversion
        entry point must call this first or the loss dereferences None. The block mask is
        model-shaped and cached; the g-side unit vectors depend on the observed vector and are
        recomputed per inversion. No-op unless the resolved loss is double-cossim.
        """
        if self._effective_loss_fn != "double-cossim":
            return
        if self._cont_block_mask is None:
            self._cont_block_mask = self._build_cont_block_mask(
                model, grad_flat.numel(), grad_flat.device,
            )
            self._dc_mf = self._cont_block_mask.to(grad_flat.dtype)
            self._dc_mf_inv = 1.0 - self._dc_mf
        eps = 1e-8
        g_c = grad_flat * self._dc_mf
        g_r = grad_flat * self._dc_mf_inv
        self._dc_g_c_unit = g_c / g_c.norm().clamp(min=eps)
        self._dc_g_r_unit = g_r / g_r.norm().clamp(min=eps)

    def _single_inversion(
        self, grad, shape, model, n_cats_amounts, known_latents, simple=True,
        performance_manager=None, previous_latents=None,
    ):
        # detach grad to avoid any unwanted graph connections
        grad = [g.detach().clone() for g in grad]
        grad_flat = torch.cat([g.flatten() for g in grad])
        self._prepare_double_cossim(grad_flat, model)
        self._prepare_tokenwise(grad_flat, model)
        analytic_xt = (
            self._analytic_tokenizer_xt(grad, model, shape[0])
            if (self._ft_analytic_weight > 0 or self._ft_analytic_init) else None
        )
        strategy = self._resolve_init_strategy()
        dummy_data = self._initialise_dummy_data(shape, sum(n_cats_amounts), strategy=strategy)
        dummy_latents, nan_masks, starting_values = self.diffusion_process.init_dummy_latents(
            known_latents, self.args, previous_latents=previous_latents,
        )
        # Seed the search at the closed-form solution rather than at random init.
        if self._ft_analytic_init and strategy != "previous":
            self._seed_dummy_from_analytic(
                dummy_data.data, analytic_xt, dummy_latents, shape[1] - sum(n_cats_amounts),
            )

        # Adam optimizers for each dummy variable
        optimizer_data = torch.optim.Adam([dummy_data], lr=self.args.attack_lr)
        latent_optimizers = self.diffusion_process.build_latent_optimizers(dummy_latents, self.args, simple)

        assert self.args.optim_iters > 0, "Number of optimization iterations must be positive"
        softmax_temp = getattr(self.args, "softmax_temp_init", 1000.0)
        softmax_temp_min = getattr(self.args, "softmax_temp_min", 1e-6)
        if self.args.optim_iters > 1:
            softmax_temp_decay = np.exp(np.log(softmax_temp_min / softmax_temp) / (self.args.optim_iters - 1))
        else:
            softmax_temp_decay = 1.0
        debug_rows = [] if self._debug_prior else None

        # Reset EMA scales so each inversion starts fresh; stale scales from a prior
        # run would mis-weight the first iterations of the new optimization.
        self._mse_scale = None
        self._cossim_scale = None
        self._last_mse = None
        self._last_direction = None
        self._last_mse_term_live = None
        self._last_cossim_term_live = None

        for iters in range(self.args.optim_iters):
            self.diffusion_process.set_attack_iteration(iters)
            optimizer_data.zero_grad()

            for opt in latent_optimizers.values():
                if opt is not None:
                    opt.zero_grad()

            # apply softmax1d to categorical parts
            dummy_data_softmax = self.softmax_cat(dummy_data, n_cats_amounts, temperature=softmax_temp)
            softmax_temp = max(softmax_temp * softmax_temp_decay, softmax_temp_min)

            batch_noised = self.diffusion_process.forward_diffuse_attack(dummy_data_softmax, dummy_latents, shape[0])
            loss = self.diffusion_process.compute_attack_inner_loss(model, batch_noised, dummy_latents)

            grads_predicted = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Gradient-matching loss (flattened for fewer kernel launches)
            pred_flat = torch.cat([g.flatten() for g in grads_predicted])
            loss_grad_match = self._compute_gradient_matching_loss(grad_flat, pred_flat)

            # Latent prior regularizer (e.g. mask <-> t coupling for masked diffusion).
            # No-op for diffusion processes that don't override it.
            prior_term = self.diffusion_process.compute_latent_prior_loss(dummy_latents)
            loss_grad = loss_grad_match + prior_term

            # Pin the dummy's noised row to the tokenizer's closed-form solution (batch 1 only,
            # where it is exact). The gradient already determines x_t; this states it directly
            # instead of hoping the optimizer rediscovers it.
            if analytic_xt is not None:
                xt_target, xt_valid = analytic_xt
                loss_grad = loss_grad + self._ft_analytic_weight * torch.mean(
                    ((batch_noised - xt_target) ** 2)[xt_valid]
                )

            if self._debug_prior and (iters % self._debug_trace_stride == 0 or iters == self.args.optim_iters - 1):
                nan = float("nan")
                # Per-term grad norms on dummy_data (retain_graph so final backward still works).
                mse_term_grad_norm = nan
                cossim_term_grad_norm = nan
                if self._last_mse_term_live is not None:
                    g = torch.autograd.grad(
                        self._last_mse_term_live, dummy_data, retain_graph=True,
                    )[0]
                    mse_term_grad_norm = float(g.detach().norm().item())
                if self._last_cossim_term_live is not None:
                    g = torch.autograd.grad(
                        self._last_cossim_term_live, dummy_data, retain_graph=True,
                    )[0]
                    cossim_term_grad_norm = float(g.detach().norm().item())
                row = {
                    "batch_size": shape[0],
                    "loss_fn": self.loss_fn,
                    "iter": iters,
                    "loss_grad_match": float(loss_grad_match.detach().item()),
                    "mse_raw": float(self._last_mse.item()) if self._last_mse is not None else nan,
                    "direction_raw": float(self._last_direction.item()) if self._last_direction is not None else nan,
                    "mse_scale": float(self._mse_scale.item()) if self._mse_scale is not None else nan,
                    "cossim_scale": float(self._cossim_scale.item()) if self._cossim_scale is not None else nan,
                    "mse_term_grad_norm": mse_term_grad_norm,
                    "cossim_term_grad_norm": cossim_term_grad_norm,
                    "prior_term": float(prior_term.detach().item()),
                    "softmax_temp": float(softmax_temp),
                }
                row.update(self.diffusion_process.get_debug_stats(dummy_latents))
                debug_rows.append(row)

            loss_grad.backward()

            # Update with Adam
            optimizer_data.step()
            for opt in latent_optimizers.values():
                if opt is not None:
                    opt.step()
            self.diffusion_process.restore_known_latents(dummy_latents, nan_masks, starting_values)

            # free per-iter tensors
            del dummy_data_softmax, batch_noised
            del loss, grads_predicted, pred_flat
            del loss_grad_match, prior_term

        if self._debug_prior and debug_rows:
            self._write_prior_trace(debug_rows)

        if self.args.mask_verbose and hasattr(self.diffusion_process, 'print_debug'):
            self.diffusion_process.print_debug(dummy_latents, nan_masks)

        # Return and move to CPU immediately, clean up optimizers and parameters
        final_data = dummy_data.detach().cpu()
        final_latents = {k: v.detach().cpu() for k, v in dummy_latents.items()}
        del dummy_data, optimizer_data, latent_optimizers
        self.diffusion_process.cleanup_dummy_latents(dummy_latents)
        return final_data, loss_grad.detach().cpu(), final_latents

    def _single_inversion_gidm(
        self,
        grad,
        shape,
        model,
        model_pretrained,
        n_cats_amounts,
        known_latents,
        simple=True,
        performance_manager=None,
    ):
        # detach grad to avoid any unwanted graph connections
        grad = [g.detach().clone() for g in grad]
        grad_flat = torch.cat([g.flatten() for g in grad])
        self._prepare_double_cossim(grad_flat, model)
        self._prepare_tokenwise(grad_flat, model)
        dummy_latent = torch.normal(0, 1, size=shape, dtype=torch.float32).to(self.device).requires_grad_(True)
        dummy_latents, nan_masks, starting_values = self.diffusion_process.init_dummy_latents(known_latents, self.args)

        # Adam optimizers for each dummy variable
        optimizer_latent = torch.optim.Adam([dummy_latent], lr=self.args.attack_lr)
        latent_optimizers = self.diffusion_process.build_latent_optimizers(dummy_latents, self.args, simple)

        assert self.args.optim_iters > 0, "Number of optimization iterations must be positive"
        optimizer_data = None
        dummy_data = None
        loss_grad = None
        self._mse_scale = self._cossim_scale = None
        self._last_mse = self._last_direction = None
        self._last_mse_term_live = self._last_cossim_term_live = None

        for phase in range(2):
            softmax_temp = getattr(self.args, "softmax_temp_init", 1000.0)
            softmax_temp_min = getattr(self.args, "softmax_temp_min", 1e-6)
            if self.args.optim_iters > 1:
                softmax_temp_decay = np.exp(
                    np.log(softmax_temp_min / softmax_temp) / (self.args.optim_iters - 1)
                )
            else:
                softmax_temp_decay = 1.0
            for iters in range(self.args.optim_iters):
                self.diffusion_process.set_attack_iteration(iters)
                if phase == 0:
                    dummy_data = self.diffusion_process.fast_denoise(
                        dummy_latent, model_pretrained
                    )  # get clean sample estimate using the pretrained model
                    optimizer_latent.zero_grad()
                else:
                    if iters == 0:
                        with torch.no_grad():
                            dummy_data = self.diffusion_process.fast_denoise(dummy_latent, model_pretrained)
                        dummy_data.requires_grad_(True)
                        optimizer_data = torch.optim.Adam([dummy_data], lr=self.args.attack_lr)
                    optimizer_data.zero_grad()

                for opt in latent_optimizers.values():
                    if opt is not None:
                        opt.zero_grad()

                dummy_data_softmax = self.softmax_cat(dummy_data, n_cats_amounts, temperature=softmax_temp)
                softmax_temp = max(softmax_temp * softmax_temp_decay, softmax_temp_min)

                batch_noised = self.diffusion_process.forward_diffuse_attack(dummy_data_softmax, dummy_latents, shape[0])
                loss = self.diffusion_process.compute_attack_inner_loss(model, batch_noised, dummy_latents)

                grads_predicted = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                # Gradient-matching loss (flattened for fewer kernel launches).
                pred_flat = torch.cat([g.flatten() for g in grads_predicted])
                loss_grad = (
                    self._compute_gradient_matching_loss(grad_flat, pred_flat)
                    + self.diffusion_process.compute_latent_prior_loss(dummy_latents)
                )
                loss_grad.backward()

                # Update with Adam
                if phase == 0:
                    optimizer_latent.step()
                else:
                    optimizer_data.step()
                for opt in latent_optimizers.values():
                    if opt is not None:
                        opt.step()
                self.diffusion_process.restore_known_latents(dummy_latents, nan_masks, starting_values)

                # free per-iter tensors
                del dummy_data_softmax, batch_noised
                del loss, grads_predicted, pred_flat

        # Return and move to CPU immediately, clean up optimizers and parameters
        final_data = dummy_data.detach().cpu()
        del dummy_data, dummy_latent, optimizer_latent, latent_optimizers
        self.diffusion_process.cleanup_dummy_latents(dummy_latents)

        if optimizer_data is not None:
            del optimizer_data
        del loss_grad

        return final_data

    def _run_inversion_worker(
        self, idx, results_dict, grad_cpu, shape, model_state_dict, model_config, n_cats_amounts,
        known_latents_cpu, simple, performance_manager, previous_latents_cpu=None,
    ):
        """Worker function that runs in a separate process.
        Receives CPU tensors and model state_dict to avoid CUDA IPC memory leaks.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Reconstruct the same backbone and capacity as the released-gradient model.
        model = build_diffusion_model(model_config).to(self.device)
        model.load_state_dict(model_state_dict)

        grad = [g.to(self.device) for g in grad_cpu]
        known_latents = {k: v.to(self.device) for k, v in known_latents_cpu.items()}
        previous_latents = (
            {k: v.to(self.device) for k, v in previous_latents_cpu.items()}
            if previous_latents_cpu is not None else None
        )

        # set seed
        torch.manual_seed(torch.seed() + idx)
        result, loss, final_latents = self._single_inversion(
            grad,
            shape,
            model,
            n_cats_amounts,
            known_latents,
            simple=simple,
            performance_manager=performance_manager,
            previous_latents=previous_latents,
        )

        # Track memory allocated by this specific worker process
        vram_mb = 0.0
        if torch.cuda.is_available():
             vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        results_dict[idx] = (result, loss, vram_mb, final_latents)

        # Free GPU memory in this process before exit
        del result, grad, model, known_latents
        torch.cuda.empty_cache()
        return None

    def _run_inversion_worker_gidm(
        self,
        results_dict,
        idx,
        grad_cpu,
        shape,
        model_state_dict,
        model_pretrained_state_dict,
        model_config,
        model_pretrained_config,
        n_cats_amounts,
        known_latents_cpu,
        simple,
        performance_manager,
    ):
        """Worker function that runs in a separate process.
        Receives CPU tensors and model state_dicts to avoid CUDA IPC memory leaks.
        """
        model = build_diffusion_model(model_config).to(self.device)
        model.load_state_dict(model_state_dict)
        model_pretrained = build_diffusion_model(model_pretrained_config).to(self.device)
        model_pretrained.load_state_dict(model_pretrained_state_dict)

        grad = [g.to(self.device) for g in grad_cpu]
        known_latents = {k: v.to(self.device) for k, v in known_latents_cpu.items()}

        # set seed
        torch.manual_seed(self.args.seed + idx)
        result = self._single_inversion_gidm(
            grad,
            shape,
            model,
            model_pretrained,
            n_cats_amounts,
            known_latents,
            simple=simple,
            performance_manager=performance_manager,
        )

        results_dict[idx] = result.detach().cpu()

        # Free GPU memory in this process before exit
        del result, grad, model, model_pretrained, known_latents
        torch.cuda.empty_cache()
        return None

    def _single_inversion_fedavg(
        self, weight_delta, shape, model, n_cats_amounts, known_latents, K, local_lr, simple=True,
    ):
        """Invert a per-client FedAvg weight update.

        The client runs K local SGD steps on its data (starting from the known global weights theta0)
        and shares the resulting weight delta = theta_K - theta0. We recover the data by optimizing
        dummy data and K sets of dummy latents so that unrolling the same K local steps on the dummy
        data reproduces the observed delta (a MAML-style differentiable inner loop). Gaussian/masked
        inner loss is reused; the inner steps use plain SGD to match the client protocol.
        """
        from torch.func import functional_call

        param_names = [n for n, _ in model.named_parameters()]
        theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffers = {n: b.detach() for n, b in model.named_buffers()}
        delta_flat = torch.cat([d.detach().flatten() for d in weight_delta])
        # The weight delta is flattened parameter-wise exactly like a gradient, so the same
        # continuous/rest block split (and its precomputation) applies.
        self._prepare_double_cossim(delta_flat, model)
        self._prepare_tokenwise(delta_flat, model)

        strategy = self._resolve_init_strategy()
        dummy_data = self._initialise_dummy_data(shape, sum(n_cats_amounts), strategy=strategy)
        if len(known_latents) != K:
            raise ValueError(f"Expected {K} FedAvg latent releases, got {len(known_latents)}")

        dummy_latents_list, latent_opts = [], []
        latent_nan_masks, latent_starting_values = [], []
        for known_k in known_latents:
            dl, nan_masks, starting_values = self.diffusion_process.init_dummy_latents(
                known_k, self.args,
            )
            dummy_latents_list.append(dl)
            latent_nan_masks.append(nan_masks)
            latent_starting_values.append(starting_values)
            latent_opts.append(self.diffusion_process.build_latent_optimizers(dl, self.args, simple))
        optimizer_data = torch.optim.Adam([dummy_data], lr=self.args.attack_lr)

        softmax_temp = getattr(self.args, "softmax_temp_init", 1000.0)
        softmax_temp_min = getattr(self.args, "softmax_temp_min", 1e-6)
        decay = np.exp(np.log(softmax_temp_min / softmax_temp) / max(self.args.optim_iters - 1, 1))
        self._mse_scale = self._cossim_scale = None
        self._last_mse = self._last_direction = None
        self._last_mse_term_live = self._last_cossim_term_live = None

        last_loss = 0.0
        for _it in range(self.args.optim_iters):
            self.diffusion_process.set_attack_iteration(_it)
            optimizer_data.zero_grad()
            for opts in latent_opts:
                for o in opts.values():
                    if o is not None:
                        o.zero_grad()
            dummy_softmax = self.softmax_cat(dummy_data, n_cats_amounts, temperature=softmax_temp)
            softmax_temp = max(softmax_temp * decay, softmax_temp_min)

            # differentiable unroll of K local SGD steps
            p = {n: theta0[n].clone().requires_grad_(True) for n in param_names}
            p_start = p
            prior_loss = torch.zeros((), device=self.device)
            for k in range(K):
                latents_k = dummy_latents_list[k]
                x_noised = self.diffusion_process.forward_diffuse_attack(dummy_softmax, latents_k, shape[0])
                pk = p
                fmodel = lambda x, t, _p=pk: functional_call(model, (_p, buffers), (x, t))
                loss_k = self.diffusion_process.compute_attack_inner_loss(fmodel, x_noised, latents_k)
                prior_loss = prior_loss + self.diffusion_process.compute_latent_prior_loss(latents_k)
                g = torch.autograd.grad(loss_k, list(pk.values()), create_graph=True)
                p = {n: pk[n] - local_lr * gi for n, gi in zip(param_names, g)}

            pred_delta_flat = torch.cat([(p[n] - p_start[n]).flatten() for n in param_names])
            loss_match = self._compute_gradient_matching_loss(delta_flat, pred_delta_flat)
            total_loss = loss_match + prior_loss / K
            total_loss.backward()
            optimizer_data.step()
            for opts in latent_opts:
                for o in opts.values():
                    if o is not None:
                        o.step()
            for dl, nan_masks, starting_values in zip(
                dummy_latents_list, latent_nan_masks, latent_starting_values,
            ):
                self.diffusion_process.restore_known_latents(dl, nan_masks, starting_values)
            last_loss = float(loss_match.detach().item())
        final_latents = [
            {name: value.detach().cpu() for name, value in dl.items()}
            for dl in dummy_latents_list
        ]
        for dl in dummy_latents_list:
            self.diffusion_process.cleanup_dummy_latents(dl)
        return dummy_data.detach(), last_loss, final_latents

    def inversion_fedavg(
        self, weight_delta, shape, model, n_cats_amounts, known_latents, K, local_lr,
        simple=True, performance_manager=None,
    ):
        """Public entry for per-client FedAvg weight-update inversion (single reconstruction)."""
        t0 = time.time()
        recon, last_loss, latents = self._single_inversion_fedavg(
            weight_delta, shape, model, n_cats_amounts, known_latents, K, local_lr, simple,
        )
        final_recon = recon.detach().cpu()
        elapsed = time.time() - t0
        metrics = {
            "time_optimization_s": elapsed,
            "time_pooling_s": 0.0,
            "time_alignment_s": 0.0,
            "reconstruction_time": elapsed,
            "estimated_vram_mb": (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
            "final_latents": latents[0] if latents else None,
            "raw_reconstruction": final_recon.numpy(),
            "epoch_lr_used": None,
            "fedavg_local_steps": K,
            "final_loss": last_loss,
        }
        return final_recon, metrics

    def inversion(
        self,
        grad,
        shape,
        model,
        n_cats_amounts,
        known_latents,
        ground_truth=None,
        simple=True,
        performance_manager=None,
    ):
        threads = self.N
        results = [None] * self.N

        # Latent warm-start is independent of dummy-data init strategy:
        # latent warm-start helps multi-epoch ensembling, data warm-start poisons it.
        # Latent warm-start is per-slot; under order-invariant global ensembling (reshuffle) the slot
        # no longer maps to fixed rows, so disable it there to avoid a latent<->row mismatch.
        warm_latents = getattr(self.args, "warm_latents", True) and not getattr(self.args, "global_ensemble", False)
        current_ix = self.previous_ix if self.size > 0 else -1
        previous_latents = None
        if warm_latents and self.size > 0 and 0 <= current_ix < len(self.latest_latents):
            stored = self.latest_latents[current_ix]
            if stored is not None:
                previous_latents = {k: v.to(self.device) for k, v in stored.items()}

        # --- Optimization phase ---
        t_opt_start = time.time()
        vram = 0.0

        all_final_latents = []
        if threads == 1:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            result, loss, final_latents = self._single_inversion(
                grad,
                shape,
                model,
                n_cats_amounts,
                known_latents,
                simple=simple,
                performance_manager=performance_manager,
                previous_latents=previous_latents,
            )

            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

            results = [result]
            losses = [loss]
            all_final_latents = [final_latents]
        else:
            # Move tensors to CPU to avoid CUDA IPC memory leaks across processes
            grad_cpu = [g.detach().cpu() for g in grad]
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            model_config = diffusion_model_config(model)
            known_latents_cpu = {k: v.cpu() for k, v in known_latents.items()}
            previous_latents_cpu = (
                {k: v.cpu() for k, v in previous_latents.items()}
                if previous_latents is not None else None
            )

            manager = mp.Manager()
            results_dict = manager.dict()
            mp.spawn(
                self._run_inversion_worker,
                nprocs=threads,
                args=(
                    results_dict,
                    grad_cpu,
                    shape,
                    model_state_dict,
                    model_config,
                    n_cats_amounts,
                    known_latents_cpu,
                    simple,
                    performance_manager,
                    previous_latents_cpu,
                ),
            )
            results = [results_dict[i][0] for i in range(threads)]
            losses = [results_dict[i][1] for i in range(threads)]
            all_final_latents = [results_dict[i][3] for i in range(threads)]

            # Sum the peak memory from all spawned worker processes
            vram = sum([results_dict[i][2] for i in range(threads)])

            del results_dict, grad_cpu, model_state_dict, known_latents_cpu
            manager.shutdown()
            del manager
        t_opt_end = time.time()

        # sort results by losses
        sorted_indices = np.argsort(losses)
        results = [results[i] for i in sorted_indices]
        best_final_latents = all_final_latents[sorted_indices[0]]

        # --- Pooling phase ---
        t_pool_start = time.time()
        final_recon = self._ensemble_reconstructions(
            results, strategy="median", ground_truth=ground_truth if self.args.perfect_pooling else None
        )  # median pooling, as in TabLeak
        t_pool_end = time.time()
        del results
        # remove gradient tracking
        final_recon = final_recon.detach().cpu()
        raw_recon = final_recon.clone()  # pool-median pre-blend snapshot for diagnostic logging

        # --- Alignment phase ---
        t_align_start = time.time()
        if self.args.epochs > 1:
            if self.size < 0:
                self.latest_reconstructions.append(final_recon)  # add latest to list
                self.alignment_anchors.append(ground_truth.cpu() if self.args.perfect_pooling else final_recon)
                self.latest_latents.append(best_final_latents)
            else:
                # Capture batch slot before ensemble_batch increments previous_ix.
                latent_slot = self.previous_ix
                final_recon, aligned_latents = self.ensemble_batch(
                    final_recon, ground_truth=ground_truth, latents=best_final_latents,
                )
                if 0 <= latent_slot < len(self.latest_latents):
                    self.latest_latents[latent_slot] = aligned_latents
        t_align_end = time.time()

        time_optimization = t_opt_end - t_opt_start
        time_pooling = t_pool_end - t_pool_start
        time_alignment = t_align_end - t_align_start

        metrics = {
            "time_optimization_s": time_optimization,
            "time_pooling_s": time_pooling,
            "time_alignment_s": time_alignment,
            "reconstruction_time": time_optimization + time_pooling + time_alignment,
            "estimated_vram_mb": vram,
            "final_latents": best_final_latents,
            "raw_reconstruction": raw_recon.numpy(),
            # How well the attack ended up matching the observed gradient. Separates the two ways
            # an attack can fail: a near-zero residual with a bad reconstruction means the gradient
            # is non-identifiable (many inputs produce it), while a large residual means the
            # optimizer never matched the gradient at all.
            # The composite loss is NOT comparable across loss_fn values ("both" is EMA-normalized
            # and sits near 1.0 by construction), so record the raw terms too.
            "final_grad_loss": float(np.min(losses)),
            "final_cossim_dist": (
                float(self._last_direction) if self._last_direction is not None else None
            ),
            "final_grad_mse": float(self._last_mse) if self._last_mse is not None else None,
            "epoch_lr_used": (
                float(self.epoch_lr_schedule[self.epoch_ix]) if self.epoch_ix >= 0 else None
            ),
        }

        # cleanup
        del grad
        del shape
        del model
        del n_cats_amounts
        del known_latents
        del simple
        return final_recon.numpy(), metrics

    def inversion_gidm(
        self,
        grad,
        shape,
        model,
        model_pretrained,
        n_cats_amounts,
        known_latents,
        ground_truth=None,
        simple=True,
        performance_manager=None,
    ):
        threads = []
        results = [None] * self.N

        if self.N == 1:
            results = [
                self._single_inversion_gidm(
                    grad,
                    shape,
                    model,
                    model_pretrained,
                    n_cats_amounts,
                    known_latents,
                    simple=simple,
                    performance_manager=performance_manager,
                )
            ]
        else:
            # Move tensors to CPU to avoid CUDA IPC memory leaks across processes
            grad_cpu = [g.detach().cpu() for g in grad]
            model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            model_pretrained_state_dict = {k: v.cpu() for k, v in model_pretrained.state_dict().items()}
            model_config = diffusion_model_config(model)
            model_pretrained_config = diffusion_model_config(model_pretrained)
            known_latents_cpu = {k: v.cpu() for k, v in known_latents.items()}

            mp.set_start_method("spawn", force=True)
            manager = mp.Manager()
            results_dict = manager.dict()

            processes = []
            for i in range(self.N):
                p = mp.Process(
                    target=self._run_inversion_worker_gidm,
                    args=(
                        results_dict,
                        i,
                        grad_cpu,
                        shape,
                        model_state_dict,
                        model_pretrained_state_dict,
                        model_config,
                        model_pretrained_config,
                        n_cats_amounts,
                        known_latents_cpu,
                        simple,
                        performance_manager,
                    ),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                p.close()
                del p
            results = [results_dict[i] for i in range(self.N)]
            del results_dict, grad_cpu, model_state_dict, model_pretrained_state_dict
            del known_latents_cpu
            manager.shutdown()
            del manager

        final_recon = self._ensemble_reconstructions(results, strategy="median", ground_truth=ground_truth)
        # remove gradient tracking
        final_recon = final_recon.detach()

        # if no epoch has been completed yet, just add
        if self.size < 0:
            self.latest_reconstructions.append(final_recon)  # add latest to list
        else:
            final_recon = self.ensemble_batch(final_recon)
        return final_recon.numpy()

    def inversion_full_dataset(self, collected_batches, model, n_cats_amounts):
        """
        Joint optimization over all batches simultaneously.
        Each batch has its own dummy_data and dummy latent variables.
        A single shared Adam optimizer updates all parameters.
        Each batch uses its own model snapshot for gradient matching.
        Returns (list_of_reconstructions, metrics_dict).
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        all_dummy_data = []
        all_dummy_latents = []
        all_nan_masks = []
        all_starting_values = []
        all_optimizable = []
        model_snapshots = []
        all_analytic_xt = []

        strategy = self._resolve_init_strategy()
        model_config = diffusion_model_config(model)
        for i, batch_info in enumerate(collected_batches):
            shape = batch_info['shape']
            known_latents = batch_info['known_latents']

            previous_latents = None
            if strategy == "previous" and i < len(self.latest_latents) and self.latest_latents[i] is not None:
                previous_latents = {k: v.to(self.device) for k, v in self.latest_latents[i].items()}

            batch_strategy = (
                strategy
                if strategy != "previous" or i < len(self.latest_reconstructions)
                else "uniform"
            )
            dummy_data = self._initialise_dummy_data(
                shape, sum(n_cats_amounts), strategy=batch_strategy, previous_ix=i,
            )
            dummy_latents, nan_masks, starting_values = self.diffusion_process.init_dummy_latents(
                known_latents, self.args, previous_latents=previous_latents,
            )

            all_dummy_data.append(dummy_data)
            all_dummy_latents.append(dummy_latents)
            all_nan_masks.append(nan_masks)
            all_starting_values.append(starting_values)

            all_optimizable.append(dummy_data)
            all_optimizable.extend(self.diffusion_process.get_all_optimizable_params(dummy_latents))

            # Recreate the same backbone/capacity as the per-batch snapshot.
            snapshot_model = build_diffusion_model(model_config).to(self.device)
            snapshot_model.load_state_dict(batch_info['model_state'])
            model_snapshots.append(snapshot_model)
            analytic_xt = (
                self._analytic_tokenizer_xt(batch_info['grads'], snapshot_model, shape[0])
                if (self._ft_analytic_weight > 0 or self._ft_analytic_init)
                else None
            )
            all_analytic_xt.append(analytic_xt)
            if self._ft_analytic_init and batch_strategy != "previous":
                self._seed_dummy_from_analytic(
                    dummy_data.data,
                    analytic_xt,
                    dummy_latents,
                    shape[1] - sum(n_cats_amounts),
                )

        # Pre-flatten real gradients once
        all_grad_flats = [torch.cat([g.flatten() for g in b['grads']]) for b in collected_batches]

        # Single shared optimizer
        optimizer = torch.optim.Adam(all_optimizable, lr=self.args.attack_lr)
        softmax_temp = getattr(self.args, "softmax_temp_init", 1000.0)
        softmax_temp_min = getattr(self.args, "softmax_temp_min", 1e-6)
        if self.args.optim_iters > 1:
            softmax_temp_decay = np.exp(
                np.log(softmax_temp_min / softmax_temp) / (self.args.optim_iters - 1)
            )
        else:
            softmax_temp_decay = 1.0

        # Reset per-inversion scales for the "both" loss path.
        self._mse_scale = None
        self._cossim_scale = None
        self._last_mse = None
        self._last_direction = None
        self._last_mse_term_live = None
        self._last_cossim_term_live = None

        # --- Optimization phase ---
        t_opt_start = time.time()
        for iteration in range(self.args.optim_iters):
            self.diffusion_process.set_attack_iteration(iteration)
            optimizer.zero_grad()
            total_loss = 0.0

            for i, batch_info in enumerate(collected_batches):
                shape = batch_info['shape']
                grad_flat = all_grad_flats[i]
                self._prepare_double_cossim(grad_flat, model_snapshots[i])
                self._prepare_tokenwise(grad_flat, model_snapshots[i])

                dummy_data_softmax = self.softmax_cat(all_dummy_data[i], n_cats_amounts, temperature=softmax_temp)
                batch_noised = self.diffusion_process.forward_diffuse_attack(
                    dummy_data_softmax, all_dummy_latents[i], shape[0]
                )
                loss = self.diffusion_process.compute_attack_inner_loss(
                    model_snapshots[i], batch_noised, all_dummy_latents[i]
                )
                grads_predicted = torch.autograd.grad(loss, model_snapshots[i].parameters(), create_graph=True)

                pred_flat = torch.cat([g.flatten() for g in grads_predicted])
                batch_loss = self._compute_gradient_matching_loss(grad_flat, pred_flat)
                batch_loss = batch_loss + self.diffusion_process.compute_latent_prior_loss(
                    all_dummy_latents[i],
                )
                analytic_xt = all_analytic_xt[i]
                if analytic_xt is not None:
                    xt_target, xt_valid = analytic_xt
                    batch_loss = batch_loss + self._ft_analytic_weight * torch.mean(
                        ((batch_noised - xt_target) ** 2)[xt_valid]
                    )
                total_loss = total_loss + batch_loss

                # free per-batch tensors
                del dummy_data_softmax, batch_noised
                del loss, grads_predicted, pred_flat

            total_loss.backward()
            optimizer.step()
            softmax_temp = max(softmax_temp * softmax_temp_decay, softmax_temp_min)

            # Restore known values for all batches
            for i in range(len(collected_batches)):
                self.diffusion_process.restore_known_latents(
                    all_dummy_latents[i], all_nan_masks[i], all_starting_values[i]
                )

            del total_loss
        t_opt_end = time.time()

        # Capture exactly what PyTorch allocated during the run
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            vram = 0.0

        reconstructions = [dd.detach().cpu() for dd in all_dummy_data]
        all_final_latents = [{k: v.detach().cpu() for k, v in dl.items()} for dl in all_dummy_latents]

        # --- Alignment phase (cross-epoch, when epochs > 1) ---
        t_align_start = time.time()
        if self.args.epochs > 1:
            for i, recon in enumerate(reconstructions):
                if self.size < 0:
                    self.latest_reconstructions.append(recon)
                    gt = collected_batches[i]['ground_truth'].cpu()
                    self.alignment_anchors.append(gt if self.args.perfect_pooling else recon)
                    self.latest_latents.append(all_final_latents[i])
                else:
                    self.previous_ix = i
                    reconstructions[i], aligned_latents = self.ensemble_batch(
                        recon,
                        ground_truth=collected_batches[i]['ground_truth'],
                        latents=all_final_latents[i],
                    )
                    if i < len(self.latest_latents):
                        self.latest_latents[i] = aligned_latents
        t_align_end = time.time()

        metrics = {
            "time_optimization_s": t_opt_end - t_opt_start,
            "time_pooling_s": 0,
            "time_alignment_s": t_align_end - t_align_start,
            "reconstruction_time": (t_opt_end - t_opt_start) + (t_align_end - t_align_start),
            "estimated_vram_mb": vram,
            "all_final_latents": all_final_latents,
        }

        # Cleanup
        for dummy_latents in all_dummy_latents:
            self.diffusion_process.cleanup_dummy_latents(dummy_latents)
        del all_dummy_data, all_dummy_latents, model_snapshots, optimizer

        return [r.numpy() for r in reconstructions], metrics

    def _ensemble_reconstructions(self, reconstructions, strategy="median", ground_truth=None):
        # first one is always the best scoring one
        best_recon = ground_truth.cpu() if ground_truth is not None else reconstructions[0]
        best_recon_inverted = self.prepper.destandardize_np(best_recon)
        best_recon_inverted = self.prepper.decodeNpToDf(best_recon_inverted)
        if not self.args.no_align:
            for i in range(0 if ground_truth is not None else 1, len(reconstructions)):
                inverted = self.prepper.destandardize_np(reconstructions[i])
                inverted = self.prepper.decodeNpToDf(inverted)
                if self.args.baseline == "tableak" or self.args.no_fast_align:
                    _, new_indices = self.matcher.row_matching(best_recon_inverted, inverted)
                    reconstructions[i] = reconstructions[i][new_indices]
                else:
                    # fast align (pass pre-decoded DataFrames to avoid redundant decoding)
                    aligned_index_order, _ = self.fast_align(
                        best_recon, reconstructions[i],
                        existing_df=best_recon_inverted, new_one_df=inverted
                    )
                    reconstructions[i] = reconstructions[i][aligned_index_order]

        recon_stack = torch.stack(reconstructions)
        if strategy == "median":
            # torch.median picks the lower middle for even N (so N=2 → elementwise min).
            # quantile interpolates → true median for any N.
            final_recon = torch.quantile(recon_stack, 0.5, dim=0)
        elif strategy == "mean":
            final_recon = torch.mean(recon_stack, dim=0)
        else:
            final_recon = recon_stack[0]
        return final_recon
