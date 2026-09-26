"""DP-SGD defense diagnostics: validate the mechanism, the accountant, and the
calibration of the clip norm C against actual per-example gradient norms.

Checks:
  A1  per-example gradient L2 norm distribution at the attack point (random-init model, epoch-1
      subset, training latent distribution) and after ~200 undefended Adam steps. Calibrates the
      clip norm C (standard practice: C ~= median per-example norm; see _dp_clip_cal in
      experiments/params.py).
  A2  released-gradient SNR per noise multiplier z: ||mean clipped grad|| vs ||noise||/B, empirical
      and analytic (sqrt(P)*z*C/B); reports the z* where SNR=1.
  A3  mechanism equivalence: dp_sgd_gradient(z=0, C=huge) must equal the plain autograd gradient of
      the batch-mean loss, and dp_sgd_gradient(z=0, C=1) must equal a naive per-example clip loop.
      Failure here = real mechanism bug; gates everything else.
  A4  accountant cross-check (--check-accountant, run via `uv run --with dp-accounting`): our RDP
      curve vs the reference dp_accounting library, exact analytic <= RDP, calibration round-trips.

Usage:
  uv run python scripts/dp_diagnostics.py --datasets adult credit abalone --seeds 3
  uv run --with dp-accounting python scripts/dp_diagnostics.py --check-accountant
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

sys.path.insert(0, ".")
from src.utils import Preprocessor
from src.model import MLPDiffusion
from src.diffusion_process import create_diffusion_process
from src.defense import dp_sgd_gradient
from src.dp_accountant import analytic_gaussian_epsilon, rdp_to_epsilon, get_noise_multiplier, get_epsilon

Z_GRID = [0.003, 0.01, 0.03, 0.07, 0.15, 0.3, 0.69]
REGIMES = [(1, 64), (256, 1024)]  # (batch_sz, subset): batch-1 attack regime and a batch-256 regime


def make_args(dataname, subset_size, dp_clip=1.0):
    """Minimal args namespace covering everything GaussianDiffusionProcess / dp_sgd_gradient read."""
    return SimpleNamespace(
        dataname=dataname, subset_size=subset_size, diffusion_type="gaussian",
        beta_1=0.0001, beta_T=0.02, timesteps=500,
        no_t=False, no_noise=False, known_t_amount=0, known_noise_amount=0, known_mask_amount=0,
        simple_attack=False, random_latents=None, optim_iters=2000, attack_lr=0.06,
        t_init="midpoint", t_lr_mult=1.0, lr=0.0001,
        dp_clip=dp_clip, dp_microbatch=16, _dp_noise_multiplier=0.0,
    )


def setup(dataname, subset_size, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = make_args(dataname, subset_size)
    prepper = Preprocessor(dataname, subset_size, random_state=seed)
    n_cats = [len(c) for c in prepper.OneHotEncoder.categories_]
    feat_info = {"cont": len(prepper.num_cols), "cat": n_cats}
    dp = create_diffusion_process(args, device, feat_info=feat_info)
    x = prepper.standardize_np(prepper.encodeDfToNp(prepper.df_train))
    x = torch.tensor(x, dtype=torch.float32, device=device)
    d_in, d_out = dp.get_model_dims(x.shape[1])
    model = MLPDiffusion(d_in=d_in, d_out=d_out).to(device)
    return args, dp, x, model


def per_example_grad_norms(model, x_noised, t, noise, chunk=16):
    """L2 norm of each example's full flattened parameter gradient (same loss as dp_sgd_gradient)."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_i(p, b, x_i, t_i, noise_i):
        pred = functional_call(model, (p, b), (x_i.unsqueeze(0), t_i.unsqueeze(0)))
        return F.mse_loss(pred, noise_i.unsqueeze(0))

    grad_fn = vmap(grad(loss_i), in_dims=(None, None, 0, 0, 0))
    norms = []
    B = x_noised.shape[0]
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        ps = grad_fn(params, buffers, x_noised[s:e], t[s:e], noise[s:e])
        flat = torch.cat([g.reshape(e - s, -1) for g in ps.values()], dim=1)
        norms.append(flat.norm(dim=1).cpu())
    return torch.cat(norms)


def flat_grad(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def check_a1_a2(datasets, seeds, device, csv_path):
    rows = []
    print("\n=== A1: per-example gradient norms (C calibration) & A2: released-gradient SNR ===")
    for dataname in datasets:
        for batch_sz, subset in REGIMES:
            for state in ("init", "trained"):
                all_norms, all_signal_bB = [], []
                P = None
                for seed in range(seeds):
                    args, dp, x, model = setup(dataname, subset, seed, device)
                    P = sum(p.numel() for p in model.parameters())
                    if state == "trained":
                        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
                        for _ in range(200):
                            idx = torch.randint(0, x.shape[0], (min(256, x.shape[0]),), device=device)
                            b = x[idx]
                            lat = dp.sample_latents(b, device)
                            loss = dp.compute_training_loss(model, dp.forward_diffuse(b, lat), lat)
                            opt.zero_grad(); loss.backward(); opt.step()
                    lat = dp.sample_latents(x, device)
                    xn = dp.forward_diffuse(x, lat).detach()
                    norms = per_example_grad_norms(model, xn, lat["t"].float(), lat["noise"])
                    all_norms.append(norms)
                    # A2 signal: norm of the released (clipped, averaged) gradient at C=1
                    clipped = norms.clamp(max=1.0)
                    if batch_sz == 1:
                        all_signal_bB.append(clipped.median().item())  # per-release signal, B=1
                    else:
                        # actual mean clipped gradient over batches of size batch_sz
                        params = {k: v.detach() for k, v in model.named_parameters()}
                        for s in range(0, x.shape[0], batch_sz):
                            bsl = slice(s, s + batch_sz)
                            if x[bsl].shape[0] < batch_sz:
                                break
                            args.dp_clip, args._dp_noise_multiplier = 1.0, 0.0
                            g = dp_sgd_gradient(args, model, x[bsl],
                                                {k: v[bsl] for k, v in lat.items()}, dp, device)
                            all_signal_bB.append(flat_grad(g).norm().item())
                norms = torch.cat(all_norms).numpy()
                med, mean = float(np.median(norms)), float(norms.mean())
                q = np.percentile(norms, [10, 25, 75, 90])
                frac_clipped = float((norms > 1.0).mean())
                signal = float(np.mean(all_signal_bB))
                print(f"\n{dataname} (batch {batch_sz}, subset {subset}, {state}, {seeds} seeds, P={P/1e6:.2f}M):")
                print(f"  norms: median={med:.4f} mean={mean:.4f} p10={q[0]:.4f} p25={q[1]:.4f} "
                      f"p75={q[2]:.4f} p90={q[3]:.4f}  frac(>C=1)={frac_clipped:.3f}")
                noise_norm_analytic = {z: np.sqrt(P) * z * 1.0 / batch_sz for z in Z_GRID}
                z_star = signal * batch_sz / np.sqrt(P)
                snr_str = "  ".join(f"z={z}:{signal / noise_norm_analytic[z]:.3f}" for z in Z_GRID)
                print(f"  signal(C=1)={signal:.4f}  SNR: {snr_str}")
                print(f"  z*(SNR=1)={z_star:.5f}  ->  eps(z*)={analytic_gaussian_epsilon(z_star, 1e-5):,.0f}"
                      if z_star > 0 else "")
                rows.append(dict(dataset=dataname, batch=batch_sz, subset=subset, state=state,
                                 n=len(norms), P=P, median=med, mean=mean, p10=q[0], p25=q[1],
                                 p75=q[2], p90=q[3], frac_clipped=frac_clipped,
                                 signal_c1=signal, z_star=z_star))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[A1/A2] wrote {csv_path}")

    # decision rule (batch-1 attack point, random init)
    meds = {r["dataset"]: r["median"] for r in rows
            if r["batch"] == 1 and r["state"] == "init"}
    print("\n[A1 decision] batch-1 random-init medians:", {k: round(v, 3) for k, v in meds.items()})
    if all(0.5 <= m <= 2.0 for m in meds.values()):
        print("[A1 decision] C=1.0 is within 2x of every median per-example norm -> C=1.0 is calibrated.")
    else:
        print("[A1 decision] C=1.0 is MISCALIBRATED for at least one dataset -> use a per-dataset C.")
    return rows


def check_a2_noise_empirical(device):
    """Empirical vs analytic noise norm (validates the sqrt(P)*z*C/B aggregation argument)."""
    print("\n=== A2b: empirical noise-norm check (P=1e6 reference tensor) ===")
    P = 1_000_000
    for z in (0.01, 0.3):
        draws = [torch.normal(0.0, z, size=(P,), device=device).norm().item() for _ in range(10)]
        emp, ana = float(np.mean(draws)), np.sqrt(P) * z
        rel = abs(emp - ana) / ana
        status = "OK" if rel < 0.02 else "FAIL"
        print(f"  z={z}: empirical={emp:.3f} analytic={ana:.3f} rel_err={rel:.4f} [{status}]")


def check_a3(device):
    print("\n=== A3: mechanism equivalence (z=0) ===")
    ok = True
    args, dp, x, model = setup("adult", 64, 0, device)
    b = x[:8]
    lat = dp.sample_latents(b, device)

    # (a) C=huge, z=0: must equal plain autograd grad of the batch-mean loss
    args.dp_clip, args._dp_noise_multiplier = 1e9, 0.0
    g_dp = flat_grad(dp_sgd_gradient(args, model, b, lat, dp, device))
    loss = dp.compute_training_loss(model, dp.forward_diffuse(b, lat), lat)
    g_ref = flat_grad(torch.autograd.grad(loss, list(model.parameters())))
    cos = F.cosine_similarity(g_dp, g_ref, dim=0).item()
    rel = ((g_dp - g_ref).norm() / g_ref.norm()).item()
    st = "OK" if (cos >= 0.99999 and rel <= 1e-4) else "FAIL"
    ok &= st == "OK"
    print(f"  (a) unclipped vs autograd: cos={cos:.7f} rel_l2={rel:.2e} [{st}]")

    # (b) C=1, z=0: must equal a naive per-example clip loop
    args.dp_clip = 1.0
    g_dp = flat_grad(dp_sgd_gradient(args, model, b, lat, dp, device))
    summed = None
    for i in range(b.shape[0]):
        lat_i = {k: v[i:i + 1] for k, v in lat.items()}
        loss_i = dp.compute_training_loss(model, dp.forward_diffuse(b[i:i + 1], lat_i), lat_i)
        gi = flat_grad(torch.autograd.grad(loss_i, list(model.parameters())))
        gi = gi * min(1.0, 1.0 / (gi.norm().item() + 1e-12))
        summed = gi if summed is None else summed + gi
    g_naive = summed / b.shape[0]
    cos = F.cosine_similarity(g_dp, g_naive, dim=0).item()
    rel = ((g_dp - g_naive).norm() / g_naive.norm()).item()
    st = "OK" if (cos >= 0.99999 and rel <= 1e-4) else "FAIL"
    ok &= st == "OK"
    print(f"  (b) clipped vs naive loop:  cos={cos:.7f} rel_l2={rel:.2e} [{st}]")
    print(f"[A3] {'PASS — released gradients are the true (clipped) data gradients; signal-bearing.' if ok else 'FAIL — mechanism bug, stop.'}")
    return ok


def check_a4():
    print("\n=== A4: accountant cross-check ===")
    delta = 1e-5
    # exact <= RDP everywhere; round-trips
    for z in (0.3, 0.69, 1.0, 2.0, 5.0, 10.0):
        assert analytic_gaussian_epsilon(z, delta) <= rdp_to_epsilon(delta, z, 1) + 1e-9
    for target in (0.5, 1.0, 8.0):
        for T in (1, 1200):
            zc = get_noise_multiplier(target, delta, T)
            back = get_epsilon(delta, zc, T)
            assert abs(back - target) < 1e-3, (target, T, back)
    print("  exact<=RDP on grid: OK;  calibration round-trips (T=1 exact, T=1200 RDP): OK")

    try:
        import dp_accounting
        from dp_accounting import dp_event
        from dp_accounting.rdp import rdp_privacy_accountant
    except ImportError:
        print("  dp-accounting not installed — run via `uv run --with dp-accounting` for the library cross-check.")
        return
    from src.dp_accountant import DEFAULT_ORDERS
    worst = 0.0
    for z in (0.3, 0.69, 1.0, 2.0, 5.0, 10.0):
        for steps in (1, 1200):
            acct = rdp_privacy_accountant.RdpAccountant(orders=list(DEFAULT_ORDERS))
            acct.compose(dp_event.GaussianDpEvent(z), steps)
            eps_lib = acct.get_epsilon(delta)
            eps_ours = rdp_to_epsilon(delta, z, steps)
            rel = abs(eps_ours - eps_lib) / max(eps_lib, 1e-12)
            worst = max(worst, rel)
            print(f"  z={z:5.2f} T={steps:5d}: ours={eps_ours:10.4f} dp_accounting={eps_lib:10.4f} rel={rel:.2%}")
    print(f"[A4] worst relative difference vs dp-accounting: {worst:.2%} "
          f"[{'OK' if worst <= 0.05 else 'FAIL'}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["adult", "credit", "abalone"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--check-accountant", action="store_true")
    ap.add_argument("--csv", default="results/diagnostic/dp_gradnorms.csv")
    a = ap.parse_args()

    if a.check_accountant:
        check_a4()
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if not check_a3(device):
        sys.exit(1)
    check_a2_noise_empirical(device)
    check_a1_a2(a.datasets, a.seeds, device, a.csv)


if __name__ == "__main__":
    main()
