"""Self-contained (eps, delta)-DP accountant for the Gaussian mechanism.

Used by the DP-SGD defense (`src.defense.dp_sgd_gradient`). Needs no opacus / dp-accounting install
(numpy + scipy.special only, both already project dependencies).

Model: each released gradient is a Gaussian mechanism applied to a sum of per-example gradients
that have been L2-clipped to `C`. The add/remove-one-example L2 sensitivity of that sum is `C`, and
we add Gaussian noise with std `z*C` (z = noise multiplier). The noise-to-sensitivity ratio is thus
`z`.

Two accountants, dispatched by `get_epsilon` / `get_noise_multiplier` on `steps`:
- steps == 1: EXACT analytic Gaussian mechanism (Balle & Wang, ICML 2018) -- the released gradient
  is a single Gaussian mechanism, so its (eps, delta) curve is known in closed form and no
  RDP-conversion slack is paid. This is what the attack-side sweeps report (one released gradient).
- steps > 1: Renyi-DP composition, RDP(alpha) = steps * alpha / (2 * z**2), converted to
  (eps, delta) by the standard minimization over orders. Used for the utility-side accounting of a
  full training run (many releases).

We deliberately do NOT claim privacy amplification by subsampling (i.e. we take sampling rate q = 1).
This is the conservative choice: the reported epsilon is an UPPER bound on the true privacy loss, so
the defense is, if anything, under-credited. If the client Poisson-subsamples its local data, the
true epsilon is smaller (stronger privacy).

References: Mironov, "Renyi Differential Privacy" (2017); Balle & Wang, "Improving the Gaussian
Mechanism for Differential Privacy" (ICML 2018), Eq. (6): for sensitivity/noise ratio mu = 1/z,
delta(eps) = Phi(mu/2 - eps/mu) - e^eps * Phi(-mu/2 - eps/mu).
"""
import numpy as np
from scipy.special import log_ndtr, ndtr

# RDP orders to search over (float orders > 1). Standard Opacus/TF-Privacy-style grid.
DEFAULT_ORDERS = np.concatenate([
    np.arange(1.1, 12.0, 0.1),
    np.arange(12.0, 64.0, 1.0),
    np.array([128.0, 256.0, 512.0]),
])


def gaussian_rdp(noise_multiplier, steps, orders=DEFAULT_ORDERS):
    """RDP epsilon at each order for `steps` compositions of the Gaussian mechanism (q=1)."""
    orders = np.asarray(orders, dtype=np.float64)
    return steps * orders / (2.0 * noise_multiplier ** 2)


def rdp_to_epsilon(delta, noise_multiplier, steps, orders=DEFAULT_ORDERS):
    """Convert RDP guarantee to (eps, delta)-DP via the improved conversion of Canonne, Kamath &
    Steinke (2020), Prop. 12 — the same one dp_accounting / opacus use:
        eps = min_alpha  RDP(alpha) + log((alpha-1)/alpha) - (log delta + log alpha) / (alpha - 1).
    Strictly tighter than the classic Mironov bound RDP + log(1/delta)/(alpha-1)."""
    orders = np.asarray(orders, dtype=np.float64)
    rdp = gaussian_rdp(noise_multiplier, steps, orders)
    eps = (
        rdp
        + np.log((orders - 1.0) / orders)
        - (np.log(delta) + np.log(orders)) / (orders - 1.0)
    )
    return float(max(0.0, np.min(eps)))


def _analytic_gaussian_delta(noise_multiplier, epsilon):
    """delta(eps) for ONE Gaussian mechanism release with noise/sensitivity ratio z (Balle-Wang Eq. 6).

    Computed in log-space: the e^eps factor overflows float64 for eps > ~709 (tiny z puts the
    relevant eps in the tens of thousands), so the second term is exp(eps + log Phi(b)), which is a
    well-conditioned difference of two moderate numbers.
    """
    z = float(noise_multiplier)
    a = 1.0 / (2.0 * z) - epsilon * z
    b = -1.0 / (2.0 * z) - epsilon * z
    s = epsilon + log_ndtr(b)
    term2 = np.exp(s) if s < 0 else 1.0  # mathematically e^eps*Phi(b) <= Phi(a) <= 1; cap for safety
    return float(ndtr(a) - term2)


def analytic_gaussian_epsilon(noise_multiplier, delta, iters=200):
    """EXACT epsilon for ONE Gaussian mechanism release at the given delta (Balle & Wang 2018).

    delta(eps) is continuous and strictly decreasing in eps, so binary-search eps. Strictly tighter
    than the RDP route; use for steps == 1 only.
    """
    z = float(noise_multiplier)
    if z <= 0:
        return float("inf")
    if _analytic_gaussian_delta(z, 0.0) <= delta:
        return 0.0
    hi = 1.0
    while _analytic_gaussian_delta(z, hi) > delta:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError(f"analytic epsilon > 1e9 for z={z}, delta={delta}")
    lo = hi / 2.0 if hi > 1.0 else 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _analytic_gaussian_delta(z, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def get_epsilon(delta, noise_multiplier, steps, orders=DEFAULT_ORDERS):
    """Achieved epsilon: exact analytic Gaussian mechanism for a single release, RDP composition
    otherwise. This is the value logged as `dp_achieved_epsilon`."""
    if steps == 1:
        return analytic_gaussian_epsilon(noise_multiplier, delta)
    return rdp_to_epsilon(delta, noise_multiplier, steps, orders)


def _rdp_noise_multiplier(target_epsilon, delta, steps, orders=DEFAULT_ORDERS,
                          z_lo=1e-3, z_hi=1e4, iters=80):
    """Smallest z achieving (target_epsilon, delta)-DP over `steps` steps under RDP composition.

    Monotone: larger z -> smaller epsilon, so binary-search z.
    """
    # ensure the bracket actually spans the target
    if rdp_to_epsilon(delta, z_hi, steps, orders) > target_epsilon:
        raise ValueError(f"z_hi={z_hi} still yields eps > {target_epsilon}; widen the search bound.")
    lo, hi = z_lo, z_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        eps = rdp_to_epsilon(delta, mid, steps, orders)
        if eps > target_epsilon:   # too much privacy loss -> need more noise
            lo = mid
        else:
            hi = mid
    return hi


def get_noise_multiplier(target_epsilon, delta, steps, orders=DEFAULT_ORDERS,
                         z_lo=1e-3, z_hi=1e4, iters=80):
    """Smallest noise multiplier z achieving (target_epsilon, delta)-DP over `steps` steps.

    Dispatches like `get_epsilon`: exact analytic calibration for a single release, RDP otherwise.
    """
    if steps == 1:
        if analytic_gaussian_epsilon(z_hi, delta) > target_epsilon:
            raise ValueError(f"z_hi={z_hi} still yields eps > {target_epsilon}; widen the search bound.")
        lo, hi = z_lo, z_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if analytic_gaussian_epsilon(mid, delta) > target_epsilon:
                lo = mid
            else:
                hi = mid
        return hi
    return _rdp_noise_multiplier(target_epsilon, delta, steps, orders, z_lo, z_hi, iters)


if __name__ == "__main__":
    # Sanity checks.
    # (1) Single-step Gaussian mechanism, sigma=1, delta=1e-5: RDP gives eps ~5.3, classical
    #     sqrt(2 ln(1.25/delta))/sigma ~= 4.84, exact analytic (Balle-Wang) ~3.9. The exact value
    #     must lower-bound both looser routes.
    e_rdp = rdp_to_epsilon(1e-5, noise_multiplier=1.0, steps=1)
    e_exact = analytic_gaussian_epsilon(1.0, 1e-5)
    classical = np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0
    print(f"[check] sigma=1, T=1, delta=1e-5: exact eps={e_exact:.3f}  RDP={e_rdp:.3f}  classical~={classical:.3f}")
    assert e_exact < e_rdp and e_exact < classical
    # (2) exact <= RDP across a z grid, including the tiny-z / huge-eps regime
    for z in (0.003, 0.01, 0.07, 0.3, 0.69, 2.0, 10.0):
        assert analytic_gaussian_epsilon(z, 1e-5) <= rdp_to_epsilon(1e-5, z, 1) + 1e-9, z
    print("[check] exact <= RDP on z grid OK")
    # (3) round-trip: target eps -> z -> eps (exact path at T=1, RDP path at T=1000)
    for target in (1.0, 8.0):
        for T in (1, 1000):
            z = get_noise_multiplier(target, 1e-5, T)
            back = get_epsilon(1e-5, z, T)
            assert abs(back - target) < 1e-3, (target, T, back)
            print(f"[check] target eps={target}, T={T}: z={z:.4f} -> eps={back:.4f}")
    # (4) monotonicity: more noise -> less epsilon (both accountants)
    assert rdp_to_epsilon(1e-5, 2.0, 1) < rdp_to_epsilon(1e-5, 1.0, 1)
    assert analytic_gaussian_epsilon(2.0, 1e-5) < analytic_gaussian_epsilon(1.0, 1e-5)
    print("[check] monotonicity OK")
