# Adapted from https://github.com/zhuohangli/GGL and https://github.com/KaiyuanZh/CENSOR

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from tqdm import tqdm


def dp_sgd_gradient(args, model, data, latents, diffusion_process, device):
    """Certified DP-SGD gradient: per-example L2 clip + calibrated Gaussian noise.

    Unlike `gradient_clipping`/`additive_noise` (which clip/noise the ALREADY-AGGREGATED gradient and
    give no (eps,delta) guarantee), this computes true per-example gradients, clips each to L2 norm
    `args.dp_clip`, sums them, adds Gaussian noise with std `z * dp_clip` (z = args._dp_noise_multiplier,
    calibrated to the target (eps,delta) by src.dp_accountant), and returns the noisy mean gradient.

    Gaussian diffusion only (the loss is a plain model(x,t)->MSE, so per-example grads via torch.func are
    clean). Masked diffusion is not supported.
    """
    from .diffusion_process import GaussianDiffusionProcess
    if not isinstance(diffusion_process, GaussianDiffusionProcess):
        raise NotImplementedError(
            "DP-SGD defense currently supports Gaussian diffusion only "
            f"(got {type(diffusion_process).__name__})."
        )

    C = float(getattr(args, "dp_clip", 1.0))
    z = float(getattr(args, "_dp_noise_multiplier"))
    B = data.shape[0]

    x_noised = diffusion_process.forward_diffuse(data, latents).detach()
    t = latents["t"].float().detach()
    noise = latents["noise"].detach()

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    param_names = list(params.keys())

    def loss_i(p, b, x_i, t_i, noise_i):
        pred = functional_call(model, (p, b), (x_i.unsqueeze(0), t_i.unsqueeze(0)))
        return F.mse_loss(pred, noise_i.unsqueeze(0))

    grad_fn = vmap(grad(loss_i), in_dims=(None, None, 0, 0, 0))

    # Microbatch the per-example gradients: materializing (B, *param_shape) for every parameter at
    # once OOMs at large B with a wide MLP (256 x dim_t^2 tensors). Process the batch in chunks,
    # clip each example, and accumulate the clipped sum — memory is bounded by the chunk size, and
    # the result is identical to computing the whole batch at once.
    chunk = max(1, min(B, getattr(args, "dp_microbatch", 16)))
    summed = {n: torch.zeros_like(params[n]) for n in param_names}
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        ps = grad_fn(params, buffers, x_noised[s:e], t[s:e], noise[s:e])  # {n: (c, *shape)}
        c = e - s
        flat = torch.cat([ps[n].reshape(c, -1) for n in param_names], dim=1)  # (c, P)
        clip_factor = torch.clamp(C / (flat.norm(dim=1) + 1e-12), max=1.0)    # (c,)
        for n in param_names:
            g = ps[n]
            summed[n] += (g * clip_factor.reshape((c,) + (1,) * (g.dim() - 1))).sum(dim=0)
        del ps, flat

    noise_std = z * C
    privatized = []
    for n in param_names:
        out = summed[n]
        if noise_std > 0:
            out = out + torch.normal(0.0, noise_std, size=out.shape, device=device)
        privatized.append(out / B)
    # named_parameters() order matches parameters() order, which is what the attacker consumes
    return privatized


def additive_noise(model, input_gradient, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0  # np.inf selects max-norm clipping

    if norm_type == np.inf:
        norms = [g.detach().abs().max() for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in input_gradient]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    gradient = [g.mul_(clip_coef_clamped) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    gradient = []
    for g in input_gradient:
        thresh = torch.quantile(g.detach().abs().flatten().float(), percentage / 100.0)
        gradient.append(torch.where(g.abs() < thresh, torch.zeros_like(g), g))
    return gradient


def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device
    
    gt_data = ground_truth.clone()
    gt_data.requires_grad=True

    # register forward hook to get intermediate layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]
        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)
    
    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:,f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/((feature_graph.data[:,f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:,f] = 0
        
    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)
    
    handle.remove()
    
    return gradient

def generate_orthogonal_gradient(input_gradient):
    """
    Generate a new gradient that is orthogonal to the input_gradient.

    Args:
    - input_gradient: A list of gradients (tensors) for each layer of the model.

    Returns:
    - orthogonal_gradient: A list of gradients (tensors) that are orthogonal to the input_gradient.
    """
    orthogonal_gradient = []

    for grad in input_gradient:
        # Generate a random gradient with the same shape
        random_grad = torch.randn_like(grad)
        random_grad = random_grad / torch.norm(random_grad)

        # Flatten the gradients to treat them as vectors
        grad_flat = grad.flatten()
        random_grad_flat = random_grad.flatten()

        # Compute the projection of random_grad onto grad
        proj_scalar = torch.dot(random_grad_flat, grad_flat) / torch.norm(grad_flat)
        proj_vector = proj_scalar * grad_flat

        # Subtract the projection from random_grad to make it orthogonal
        orthogonal_grad_flat = random_grad_flat - proj_vector
        
        # Reshape back to the original shape and append to the list
        orthogonal_grad = orthogonal_grad_flat.view_as(grad)
        orthogonal_gradient.append(orthogonal_grad)

    return orthogonal_gradient

def normalize_orthogonal_gradient(ortho_gradient, original_gradient, fixed_norm=None):
    normalized_gradient = []
    for og, orig_g in zip(ortho_gradient, original_gradient):
        norm_og = torch.norm(og.flatten())
        norm_orig_g = fixed_norm if fixed_norm is not None else torch.norm(orig_g.flatten())
        normalized_g = (og / norm_og) * norm_orig_g if norm_og > 0 else og
        normalized_gradient.append(normalized_g)
    return normalized_gradient


def orthogonal_gradient(args, input_gradient, model, ground_truth, latents, diffusion_process, trials, device, epsilon=1e-5, best_loss=float('inf')):
    best_gradient = None
    learning_rate = 1e-4

    original_params = [param.clone() for param in model.parameters()]  # Save original parameters outside the loop

    for trial in range(trials):
        model.train()
        noisy_gradient = generate_orthogonal_gradient(input_gradient)
        noisy_gradient = normalize_orthogonal_gradient(noisy_gradient, input_gradient)

        with torch.no_grad():
            for param, original, noise_grad in zip(model.parameters(), original_params, noisy_gradient):
                param.data.copy_(original - noise_grad * learning_rate)  # Update and evaluate in one step

            model.eval()
            batch_noised = diffusion_process.forward_diffuse(ground_truth, latents)
            new_loss = diffusion_process.compute_training_loss(model, batch_noised, latents)

            # Keep the orthogonal gradient that yields the lowest training loss
            if new_loss < best_loss:
                best_loss = new_loss
                best_gradient = [g.detach().clone() for g in noisy_gradient]  # Only clone when necessary

    # Restore original parameters
    for param, original in zip(model.parameters(), original_params):
        param.data.copy_(original)

    if best_gradient is not None:
        input_gradient = [g.clone() for g in best_gradient]

    return input_gradient, best_loss