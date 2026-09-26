from typing import Callable, Union
import torch
import torch.nn as nn
import torch.optim
from torch.nn.attention import SDPBackend, sdpa_kernel

ModuleType = Union[str, Callable[..., nn.Module]]


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=num_channels // 2, dtype=torch.float32)
        freqs = freqs / (num_channels // 2 - (1 if endpoint else 0))
        freqs = (1 / max_positions) ** freqs
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=1024, d_out=None):
        super().__init__()
        self.dim_t = dim_t
        self.d_out = d_in if d_out is None else d_out
        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, self.d_out),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)
        x = self.proj(x)
        x = x + emb
        return self.mlp(x)


class FTTransformerDiffusion(nn.Module):
    """FT-Transformer-style diffusion denoiser (alternative backbone to MLPDiffusion).

    Same interface as MLPDiffusion: forward(x, timesteps) -> (B, d_out). Each input scalar feature is
    tokenized to a d_model vector (FT-Transformer numerical feature tokenizer), a timestep embedding is
    added to every token, and a transformer encoder attends across the feature tokens. NO positional
    encoding is used -- attention is permutation-equivariant over columns, which is the appropriate
    inductive bias for tabular data (columns have no sequence order). A per-token linear head yields a
    scalar per input dimension; the first d_out are returned (for masked diffusion d_in = d_out + n_cats
    with the mask tokens appended last, so the leading d_out tokens are the data features).
    """

    def __init__(
        self,
        d_in,
        d_out=None,
        d_model=256,
        nhead=8,
        nlayers=4,
        dim_feedforward=512,
        feat_init_std=1.0,
        head_init_std=0.01,
    ):
        super().__init__()
        if d_model % 2:
            raise ValueError(f"d_model must be even for the sinusoidal timestep embedding, got {d_model}")
        if d_model % nhead:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_in = d_in
        self.d_out = d_in if d_out is None else d_out
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.dim_feedforward = dim_feedforward
        self.feat_init_std = feat_init_std
        self.head_init_std = head_init_std
        # feature tokenizer: token_i = x_i * W_i + b_i  (per-feature embedding vector).
        # feat_init_std must be large enough that the input-token contribution (x_i * W_i) is
        # comparable to the timestep embedding added to every token; too small (e.g. 0.02) drowns
        # continuous inputs under the timestep signal so the model ignores them and the attack cannot
        # invert continuous features.
        self.feat_weight = nn.Parameter(torch.randn(d_in, d_model) * feat_init_std)
        # The bias must be RANDOM per feature, not zeros. With a zero bias, token_i = x_i * W_i,
        # so every feature whose value is 0 yields the identical (zero) token and the encoder cannot
        # tell those features apart -- catastrophic on one-hot tabular data, which is mostly zeros
        # (an 8-way one-hot value alone creates 21 colliding zero-token pairs). The random bias is
        # the feature-identity embedding that prevents this degeneracy.
        self.feat_bias = nn.Parameter(torch.randn(d_in, d_model) * feat_init_std)
        # timestep embedding (same positional-embedding scheme as MLPDiffusion)
        self.map_noise = PositionalEmbedding(num_channels=d_model)
        self.time_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            activation="gelu", batch_first=True, dropout=0.0,  # deterministic: attack needs stable grads
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)
        # A near-zero noise prediction is a much better diffusion starting point than PyTorch's
        # default Linear initialization. Keep this nonzero so gradients reach the encoder and
        # feature tokenizer from the first update.
        nn.init.normal_(self.head.weight, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, timesteps):
        tokens = x.unsqueeze(-1) * self.feat_weight + self.feat_bias  # (B, d_in, d_model)
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)  # (B, d_model)
        tokens = tokens + emb.unsqueeze(1)  # broadcast timestep onto every feature token
        # Force the math SDPA backend: the gradient-inversion attack differentiates through the
        # model's own gradient (create_graph=True), and the flash/mem-efficient attention kernels
        # do not implement the required double backward (fails on CPU and CUDA flash kernels).
        with sdpa_kernel(SDPBackend.MATH):
            h = self.encoder(tokens)  # (B, d_in, d_model)
        out = self.head(h).squeeze(-1)  # (B, d_in)
        return out[:, : self.d_out]


def diffusion_model_config(model):
    """Return the constructor configuration needed to recreate a diffusion backbone."""
    if isinstance(model, FTTransformerDiffusion):
        return {
            "backbone": "ft-transformer",
            "d_in": model.d_in,
            "d_out": model.d_out,
            "d_model": model.d_model,
            "nhead": model.nhead,
            "nlayers": model.nlayers,
            "dim_feedforward": model.dim_feedforward,
            "feat_init_std": model.feat_init_std,
            "head_init_std": model.head_init_std,
        }
    if isinstance(model, MLPDiffusion):
        return {
            "backbone": "mlp",
            "d_in": model.proj.in_features,
            "d_out": model.d_out,
            "dim_t": model.dim_t,
        }
    raise TypeError(f"Unsupported diffusion model type: {type(model).__name__}")


def build_diffusion_model(config):
    """Recreate a diffusion backbone from :func:`diffusion_model_config`."""
    config = dict(config)
    backbone = config.pop("backbone")
    if backbone == "ft-transformer":
        return FTTransformerDiffusion(**config)
    if backbone == "mlp":
        return MLPDiffusion(**config)
    raise ValueError(f"Unknown diffusion backbone: {backbone}")
