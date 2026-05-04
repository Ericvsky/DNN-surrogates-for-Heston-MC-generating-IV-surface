from __future__ import annotations

r"""
Enhanced Heston Monte Carlo surface generator + benchmark toolkit.

Main improvements over the original toolkit
-------------------------------------------
1) Cleaner documentation and naming.
2) Optional antithetic variates for lower Monte Carlo noise.
3) Optional price standard-error estimation on the full surface.
4) Stronger surface diagnostics:
   - no-arbitrage lower/upper bounds
   - monotonicity in strike
   - convexity in strike
   - monotonicity in maturity
5) Improved time-step convergence benchmark using common random numbers
   (coupled Brownian increments across different time-step levels), which
   makes the convergence table much less noisy.
6) Clean dataset generation with automatic resampling / retry until rows meet
   user-specified quality filters.
7) Optional price clipping to no-arbitrage bounds before implied-vol inversion,
   useful only for tiny Monte Carlo bound violations.

Dependencies
------------
numpy, pandas, scipy
"""

from dataclasses import asdict, dataclass
from math import lcm
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class HestonParams:
    v0: float
    rho: float
    kappa: float
    theta: float
    sigma_v: float


# ---------------------------------------------------------------------
# Black-Scholes utilities
# ---------------------------------------------------------------------
def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes price of a European call with zero dividends."""
    S0 = float(S0)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)

    if T <= 0.0:
        return max(S0 - K, 0.0)

    if sigma <= 0.0:
        return max(S0 - K * np.exp(-r * T), 0.0)

    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))



def call_price_bounds(S0: float, K_grid: ArrayLike, T_grid: ArrayLike, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return lower and upper no-arbitrage bounds for a call-price surface."""
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)
    lower = np.maximum(S0 - np.exp(-r * T_grid)[:, None] * K_grid[None, :], 0.0)
    upper = np.full((len(T_grid), len(K_grid)), float(S0), dtype=float)
    return lower, upper



def clip_price_surface_to_bounds(
    prices: np.ndarray,
    S0: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    r: float,
) -> Tuple[np.ndarray, int]:
    """
    Clip a price surface to the call no-arbitrage bounds.

    This is useful only for tiny Monte Carlo violations before implied-vol inversion.
    For research reporting, it is still better to record how often clipping happened.
    """
    lower, upper = call_price_bounds(S0, K_grid, T_grid, r)
    clipped = np.clip(np.asarray(prices, dtype=float), lower, upper)
    repaired = int(np.sum(np.abs(clipped - prices) > 1.0e-14))
    return clipped, repaired



def implied_vol_call_brent(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    vol_lower: float = 1.0e-8,
    vol_upper: float = 5.0,
) -> float:
    """
    Black-Scholes implied volatility from a call price using Brent's method.

    Returns np.nan if the price is outside no-arbitrage bounds or if root finding fails.
    """
    price = float(price)
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    upper_bound = float(S0)

    if price < intrinsic - 1.0e-10 or price > upper_bound + 1.0e-10:
        return np.nan

    if abs(price - intrinsic) <= 1.0e-10:
        return 0.0

    if abs(price - upper_bound) <= 1.0e-10:
        return vol_upper

    def objective(sigma: float) -> float:
        return bs_call_price(S0, K, T, r, sigma) - price

    try:
        return float(brentq(objective, vol_lower, vol_upper, maxiter=200))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------
def maturity_step_indices(T_grid: ArrayLike, n_steps: int) -> Tuple[np.ndarray, float]:
    """
    Convert maturity times into integer step indices on a common grid [0, T_max].

    Here n_steps is the total number of time steps between 0 and max(T_grid).
    With the default quarter-year grid up to 2 years, n_steps should be a multiple of 8.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    T_max = float(np.max(T_grid))
    raw = (T_grid / T_max) * n_steps
    rounded = np.round(raw).astype(int)

    if not np.allclose(raw, rounded, atol=1.0e-12):
        raise ValueError(
            f"n_steps={n_steps} does not align with T_grid up to T_max={T_max}. "
            "Choose n_steps so every maturity lies on the common simulation grid. "
            "For T_grid = [0.25, 0.50, ..., 2.0], use multiples of 8 such as "
            "80, 160, 400, or 800."
        )
    return rounded, T_max



def surface_to_vector(surface: np.ndarray) -> np.ndarray:
    """Flatten a surface as [T1(K1..Km), T2(K1..Km), ..., Tn(K1..Km)]."""
    return np.asarray(surface, dtype=float).reshape(-1)



def iv_column_names(K_grid: ArrayLike, T_grid: ArrayLike) -> List[str]:
    names: List[str] = []
    for T in T_grid:
        for K in K_grid:
            names.append(f"iv_T{float(T):.2f}_K{float(K):.2f}")
    return names


# ---------------------------------------------------------------------
# Random-number helpers
# ---------------------------------------------------------------------
def _generate_correlated_normals(
    M: int,
    n_steps: int,
    rho: float,
    seed: Optional[int] = None,
    antithetic: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated standard normal shocks on shape (n_steps, M).

    If antithetic=True, half the shocks are generated and then mirrored.
    The returned arrays always have exactly M columns.
    """
    rng = np.random.default_rng(seed)
    rho_perp_scale = np.sqrt(max(1.0 - rho * rho, 0.0))

    if antithetic:
        half = (M + 1) // 2
        z1_half = rng.standard_normal((n_steps, half))
        z_perp_half = rng.standard_normal((n_steps, half))
        z1_full = np.empty((n_steps, 2 * half), dtype=float)
        z_perp_full = np.empty((n_steps, 2 * half), dtype=float)
        z1_full[:, 0::2] = z1_half
        z1_full[:, 1::2] = -z1_half
        z_perp_full[:, 0::2] = z_perp_half
        z_perp_full[:, 1::2] = -z_perp_half
        z1 = z1_full[:, :M]
        z_perp = z_perp_full[:, :M]
    else:
        z1 = rng.standard_normal((n_steps, M))
        z_perp = rng.standard_normal((n_steps, M))

    z2 = rho * z1 + rho_perp_scale * z_perp
    return z1, z2



def _aggregate_normals(z: np.ndarray, coarse_n_steps: int) -> np.ndarray:
    """
    Aggregate fine-grid Brownian shocks to a coarser grid using block sums.

    If z has shape (N_fine, M) and coarse_n_steps divides N_fine, then each block of
    length N_fine / coarse_n_steps is summed and scaled by sqrt(block_size) so that the
    coarse shocks remain standard normal.
    """
    fine_n_steps, M = z.shape
    if fine_n_steps % coarse_n_steps != 0:
        raise ValueError(
            f"coarse_n_steps={coarse_n_steps} must divide fine_n_steps={fine_n_steps}"
        )
    block = fine_n_steps // coarse_n_steps
    z_reshaped = z.reshape(coarse_n_steps, block, M)
    return z_reshaped.sum(axis=1) / np.sqrt(block)


# ---------------------------------------------------------------------
# Heston Monte Carlo
# ---------------------------------------------------------------------
def simulate_heston_terminals_from_normals(
    params: HestonParams,
    S0: float,
    r: float,
    T_grid: ArrayLike,
    z1: np.ndarray,
    z2: np.ndarray,
    scheme: str = "full_truncation",
) -> List[np.ndarray]:
    r"""
    Simulate Heston paths from pre-generated correlated normal shocks.

    The stock update is log-Euler with the positive-part variance v_n^+ = max(v_n, 0):
        S_{n+1} = S_n exp((r - 0.5 v_n^+) dt + sqrt(v_n^+) sqrt(dt) Z_{n,1}).

    Variance update choices:
    - truncated_euler:
        v_{n+1} = max(v_n + kappa(theta - v_n)dt + sigma_v sqrt(v_n^+) sqrt(dt) Z_{n,2}, 0)
    - full_truncation:
        v_{n+1} = v_n + kappa(theta - v_n^+)dt + sigma_v sqrt(v_n^+) sqrt(dt) Z_{n,2}

    Returns terminal stock values at each maturity in T_grid.
    """
    n_steps, M = z1.shape
    if z2.shape != (n_steps, M):
        raise ValueError("z1 and z2 must have the same shape (n_steps, M)")

    maturity_indices, T_max = maturity_step_indices(T_grid, n_steps)
    dt = T_max / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.full(M, float(S0), dtype=float)
    v = np.full(M, float(params.v0), dtype=float)

    terminals: List[np.ndarray] = []
    maturity_ptr = 0
    targets = maturity_indices.tolist()

    for step in range(1, n_steps + 1):
        z1_step = z1[step - 1]
        z2_step = z2[step - 1]
        v_pos = np.maximum(v, 0.0)

        S *= np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1_step)

        if scheme == "truncated_euler":
            v = np.maximum(
                v + params.kappa * (params.theta - v) * dt
                + params.sigma_v * np.sqrt(v_pos) * sqrt_dt * z2_step,
                0.0,
            )
        elif scheme == "full_truncation":
            v = (
                v
                + params.kappa * (params.theta - v_pos) * dt
                + params.sigma_v * np.sqrt(v_pos) * sqrt_dt * z2_step
            )
        else:
            raise ValueError("scheme must be 'full_truncation' or 'truncated_euler'")

        while maturity_ptr < len(targets) and step == targets[maturity_ptr]:
            terminals.append(S.copy())
            maturity_ptr += 1

    return terminals



def simulate_heston_terminals(
    params: HestonParams,
    S0: float,
    r: float,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    seed: Optional[int] = None,
    scheme: str = "full_truncation",
    antithetic: bool = False,
) -> List[np.ndarray]:
    """
    Simulate Heston paths and return terminal stock values at each maturity in T_grid.
    """
    z1, z2 = _generate_correlated_normals(
        M=M,
        n_steps=n_steps,
        rho=params.rho,
        seed=seed,
        antithetic=antithetic,
    )
    return simulate_heston_terminals_from_normals(params, S0, r, T_grid, z1, z2, scheme=scheme)



def price_surface_from_terminals(
    terminals: List[np.ndarray],
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    r: float,
    return_se: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute the full call-price surface from terminal stock values.

    If return_se=True, also return a price standard-error surface.
    """
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    prices = np.empty((len(T_grid), len(K_grid)), dtype=float)
    se = np.empty_like(prices) if return_se else None

    for j, (ST, T) in enumerate(zip(terminals, T_grid)):
        disc = np.exp(-r * T)
        discounted_payoffs = disc * np.maximum(ST[:, None] - K_grid[None, :], 0.0)
        prices[j, :] = discounted_payoffs.mean(axis=0)
        if return_se:
            if discounted_payoffs.shape[0] > 1:
                se[j, :] = discounted_payoffs.std(axis=0, ddof=1) / np.sqrt(discounted_payoffs.shape[0])
            else:
                se[j, :] = np.nan

    return prices, se



def iv_surface_from_prices(
    prices: np.ndarray,
    S0: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    r: float,
) -> Tuple[np.ndarray, int]:
    """Convert a price surface into an implied-volatility surface."""
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    ivs = np.empty_like(prices, dtype=float)
    failures = 0
    for j, T in enumerate(T_grid):
        for i, K in enumerate(K_grid):
            iv = implied_vol_call_brent(prices[j, i], S0, K, T, r)
            if np.isnan(iv):
                failures += 1
            ivs[j, i] = iv
    return ivs, failures



def mc_iv_surface(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    seed: Optional[int] = None,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    return_price_se: bool = True,
    clip_prices_for_iv: bool = False,
) -> Dict[str, Any]:
    """
    Full Monte Carlo -> call-price surface -> IV surface pipeline for one parameter vector.

    Parameters
    ----------
    clip_prices_for_iv : bool
        If True, clip prices to no-arbitrage bounds before IV inversion and record how
        many nodes were adjusted. This is only recommended to repair tiny Monte Carlo
        bound violations.
    """
    t0 = perf_counter()
    terminals = simulate_heston_terminals(
        params=params,
        S0=S0,
        r=r,
        T_grid=T_grid,
        M=M,
        n_steps=n_steps,
        seed=seed,
        scheme=scheme,
        antithetic=antithetic,
    )
    simulation_time = perf_counter() - t0

    t1 = perf_counter()
    prices, price_se = price_surface_from_terminals(
        terminals,
        K_grid,
        T_grid,
        r,
        return_se=return_price_se,
    )
    sanity = sanity_checks_surface(prices, S0, K_grid, T_grid, r)

    prices_for_iv = np.asarray(prices, dtype=float)
    repaired_prices_for_iv = 0
    if clip_prices_for_iv:
        prices_for_iv, repaired_prices_for_iv = clip_price_surface_to_bounds(
            prices_for_iv, S0, K_grid, T_grid, r
        )

    ivs, iv_failures = iv_surface_from_prices(prices_for_iv, S0, K_grid, T_grid, r)
    inversion_time = perf_counter() - t1

    out: Dict[str, Any] = {
        "params": params,
        "prices": prices,
        "prices_for_iv": prices_for_iv,
        "ivs": ivs,
        "price_vector": surface_to_vector(prices),
        "iv_vector": surface_to_vector(ivs),
        "terminals": terminals,
        "iv_failures": int(iv_failures),
        "runtime_simulation_s": float(simulation_time),
        "runtime_iv_inversion_s": float(inversion_time),
        "runtime_total_s": float(simulation_time + inversion_time),
        "sanity_checks": sanity,
        "repaired_prices_for_iv": int(repaired_prices_for_iv),
        "M": int(M),
        "n_steps": int(n_steps),
        "scheme": scheme,
        "antithetic": bool(antithetic),
    }
    if price_se is not None:
        out["price_se"] = price_se
        out["price_se_vector"] = surface_to_vector(price_se)
    return out



def sanity_checks_surface(
    prices: np.ndarray,
    S0: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    r: float,
    tol: float = 1.0e-10,
) -> Dict[str, float]:
    """
    Price-surface diagnostics.

    Checks:
    - lower and upper no-arbitrage bounds
    - monotonicity in strike (call price should be non-increasing in K)
    - convexity in strike (discrete slope should be non-decreasing in K)
    - monotonicity in maturity (call price should be non-decreasing in T for q=0, r>=0)
    """
    prices = np.asarray(prices, dtype=float)
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    lower_bound, upper_bound = call_price_bounds(S0, K_grid, T_grid, r)
    monotonicity_violations = int(np.sum(np.diff(prices, axis=1) > tol))
    maturity_violations = int(np.sum(np.diff(prices, axis=0) < -tol))

    convexity_violations = 0
    if len(K_grid) >= 3:
        dK = np.diff(K_grid)
        for row in prices:
            slopes = np.diff(row) / dK
            convexity_violations += int(np.sum(np.diff(slopes) < -tol))

    return {
        "min_price": float(np.nanmin(prices)),
        "max_price": float(np.nanmax(prices)),
        "viol_lower_bound": int(np.sum(prices < lower_bound - tol)),
        "viol_upper_bound": int(np.sum(prices > upper_bound + tol)),
        "monotonicity_violations_across_strike": monotonicity_violations,
        "convexity_violations_across_strike": int(convexity_violations),
        "monotonicity_violations_across_maturity": maturity_violations,
    }



def surface_quality_ok(
    out: Dict[str, Any],
    require_zero_iv_failures: bool = True,
    require_no_bound_violations: bool = True,
    require_strike_monotonicity: bool = True,
    require_strike_convexity: bool = False,
    allow_repaired_prices_for_iv: bool = False,
) -> bool:
    """Check whether a surface output dictionary passes quality filters."""
    sanity = out["sanity_checks"]

    if require_zero_iv_failures and int(out["iv_failures"]) != 0:
        return False
    if require_no_bound_violations and (
        int(sanity["viol_lower_bound"]) != 0 or int(sanity["viol_upper_bound"]) != 0
    ):
        return False
    if require_strike_monotonicity and int(sanity["monotonicity_violations_across_strike"]) != 0:
        return False
    if require_strike_convexity and int(sanity["convexity_violations_across_strike"]) != 0:
        return False
    if (not allow_repaired_prices_for_iv) and int(out["repaired_prices_for_iv"]) != 0:
        return False
    return True


# ---------------------------------------------------------------------
# Heston semi-closed-form reference for European calls
# ---------------------------------------------------------------------
def _heston_cf(u: complex, T: float, S0: float, r: float, params: HestonParams) -> complex:
    """
    Characteristic function of log(S_T) under the Heston model.

    This uses a stable exponential-trap-style parametrisation.
    """
    i = 1j
    x0 = np.log(S0)
    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma_v
    rho = params.rho
    v0 = params.v0

    b = kappa - rho * sigma * i * u
    d = np.sqrt(b * b + sigma * sigma * (u * u + i * u))
    if np.real(d) < 0.0:
        d = -d

    g = (b - d) / (b + d)
    exp_neg_dT = np.exp(-d * T)

    C = (
        r * i * u * T
        + (kappa * theta / (sigma * sigma))
        * ((b - d) * T - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g)))
    )
    D = ((b - d) / (sigma * sigma)) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))
    return np.exp(C + D * v0 + i * u * x0)



def heston_call_price_cf(
    S0: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    integration_limit: float = 200.0,
    quad_limit: int = 250,
) -> float:
    """Semi-closed-form Heston European call price via numerical integration."""
    if T <= 0.0:
        return max(S0 - K, 0.0)

    logK = np.log(K)
    phi_minus_i = _heston_cf(-1j, T, S0, r, params)

    def integrand_p1(u: float) -> float:
        val = (
            np.exp(-1j * u * logK)
            * _heston_cf(u - 1j, T, S0, r, params)
            / (1j * u * phi_minus_i)
        )
        return float(np.real(val))

    def integrand_p2(u: float) -> float:
        val = np.exp(-1j * u * logK) * _heston_cf(u, T, S0, r, params) / (1j * u)
        return float(np.real(val))

    p1 = 0.5 + (1.0 / np.pi) * quad(
        integrand_p1,
        1.0e-9,
        integration_limit,
        limit=quad_limit,
        epsabs=1.0e-8,
        epsrel=1.0e-8,
    )[0]
    p2 = 0.5 + (1.0 / np.pi) * quad(
        integrand_p2,
        1.0e-9,
        integration_limit,
        limit=quad_limit,
        epsabs=1.0e-8,
        epsrel=1.0e-8,
    )[0]

    price = S0 * p1 - K * np.exp(-r * T) * p2
    lower_bound = max(S0 - K * np.exp(-r * T), 0.0)
    upper_bound = float(S0)
    return float(np.clip(price, lower_bound, upper_bound))



def heston_price_surface_cf(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
) -> np.ndarray:
    """Semi-closed-form Heston reference surface for European calls."""
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)
    prices = np.empty((len(T_grid), len(K_grid)), dtype=float)
    for j, T in enumerate(T_grid):
        for i, K in enumerate(K_grid):
            prices[j, i] = heston_call_price_cf(S0, K, T, r, params)
    return prices


# ---------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------
def benchmark_seed_stability(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    seeds: Iterable[int],
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Same theta, same M, same n_steps, different seeds.
    Measures Monte Carlo noise across reruns.
    """
    price_list: List[np.ndarray] = []
    iv_list: List[np.ndarray] = []
    price_se_list: List[np.ndarray] = []
    runtimes: List[float] = []
    failures: List[int] = []

    seeds = list(seeds)
    for seed in seeds:
        out = mc_iv_surface(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M,
            n_steps,
            seed=seed,
            scheme=scheme,
            antithetic=antithetic,
            return_price_se=True,
            clip_prices_for_iv=clip_prices_for_iv,
        )
        price_list.append(out["prices"])
        iv_list.append(out["ivs"])
        price_se_list.append(out["price_se"])
        runtimes.append(float(out["runtime_total_s"]))
        failures.append(int(out["iv_failures"]))

    prices = np.stack(price_list, axis=0)
    ivs = np.stack(iv_list, axis=0)
    price_se = np.stack(price_se_list, axis=0)

    summary = pd.DataFrame(
        [
            {
                "n_seeds": len(seeds),
                "M": int(M),
                "n_steps": int(n_steps),
                "mean_price_std": float(np.nanmean(np.std(prices, axis=0, ddof=1))),
                "max_price_std": float(np.nanmax(np.std(prices, axis=0, ddof=1))),
                "mean_iv_std": float(np.nanmean(np.std(ivs, axis=0, ddof=1))),
                "max_iv_std": float(np.nanmax(np.std(ivs, axis=0, ddof=1))),
                "mean_mc_price_se": float(np.nanmean(price_se)),
                "mean_runtime_s": float(np.mean(runtimes)),
                "avg_iv_failures": float(np.mean(failures)),
            }
        ]
    )

    price_std_surface = pd.DataFrame(
        np.std(prices, axis=0, ddof=1),
        index=np.asarray(T_grid, dtype=float),
        columns=np.asarray(K_grid, dtype=float),
    )
    iv_std_surface = pd.DataFrame(
        np.std(ivs, axis=0, ddof=1),
        index=np.asarray(T_grid, dtype=float),
        columns=np.asarray(K_grid, dtype=float),
    )
    mean_price_se_surface = pd.DataFrame(
        np.nanmean(price_se, axis=0),
        index=np.asarray(T_grid, dtype=float),
        columns=np.asarray(K_grid, dtype=float),
    )

    return {
        "summary": summary,
        "price_std_surface": price_std_surface,
        "iv_std_surface": iv_std_surface,
        "mean_price_se_surface": mean_price_se_surface,
    }



def benchmark_path_convergence(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M_values: Iterable[int],
    n_steps: int,
    seed: int = 123,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> pd.DataFrame:
    """
    Path-count convergence benchmark using nested subsets of one max-M simulation.
    """
    M_values = sorted({int(m) for m in M_values})
    max_M = max(M_values)

    terminals = simulate_heston_terminals(
        params,
        S0,
        r,
        T_grid,
        max_M,
        n_steps,
        seed=seed,
        scheme=scheme,
        antithetic=antithetic,
    )
    ref_prices, _ = price_surface_from_terminals([x[:max_M] for x in terminals], K_grid, T_grid, r, return_se=False)
    ref_prices_for_iv = ref_prices
    if clip_prices_for_iv:
        ref_prices_for_iv, _ = clip_price_surface_to_bounds(ref_prices_for_iv, S0, K_grid, T_grid, r)
    ref_ivs, _ = iv_surface_from_prices(ref_prices_for_iv, S0, K_grid, T_grid, r)

    rows: List[Dict[str, float]] = []
    for M_i in M_values:
        prices, price_se = price_surface_from_terminals([x[:M_i] for x in terminals], K_grid, T_grid, r, return_se=True)
        prices_for_iv = prices
        repaired = 0
        if clip_prices_for_iv:
            prices_for_iv, repaired = clip_price_surface_to_bounds(prices_for_iv, S0, K_grid, T_grid, r)
        ivs, fails = iv_surface_from_prices(prices_for_iv, S0, K_grid, T_grid, r)

        rows.append(
            {
                "M": int(M_i),
                "mean_abs_price_diff_vs_maxM": float(np.nanmean(np.abs(prices - ref_prices))),
                "max_abs_price_diff_vs_maxM": float(np.nanmax(np.abs(prices - ref_prices))),
                "mean_abs_iv_diff_vs_maxM": float(np.nanmean(np.abs(ivs - ref_ivs))),
                "max_abs_iv_diff_vs_maxM": float(np.nanmax(np.abs(ivs - ref_ivs))),
                "mean_mc_price_se": float(np.nanmean(price_se)) if price_se is not None else np.nan,
                "iv_failures": int(fails),
                "repaired_prices_for_iv": int(repaired),
            }
        )

    return pd.DataFrame(rows)



def benchmark_timestep_convergence(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps_values: Iterable[int],
    seeds: Iterable[int] | int = 123,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
    coupled: bool = True,
) -> pd.DataFrame:
    """
    Time-step convergence benchmark.

    Default behavior uses common random numbers across all time-step levels by:
    1) generating shocks on the finest grid,
    2) aggregating them to coarser grids.

    This makes the comparison much less noisy than independently simulating each level.

    Parameters
    ----------
    seeds : iterable of ints or one int
        If several seeds are supplied, the reported metrics are averaged across seeds.
    coupled : bool
        If True, use common-random-number coupling across time-step levels.
    """
    n_steps_values = sorted({int(n) for n in n_steps_values})
    ref_n = max(n_steps_values)
    fine_n = ref_n if not coupled else lcm(*n_steps_values)

    if isinstance(seeds, int):
        seed_list = [int(seeds)]
    else:
        seed_list = [int(s) for s in seeds]

    rows: List[Dict[str, float]] = []
    for seed in seed_list:
        if coupled:
            z1_fine, z2_fine = _generate_correlated_normals(
                M=M,
                n_steps=fine_n,
                rho=params.rho,
                seed=seed,
                antithetic=antithetic,
            )
            ref_t0 = perf_counter()
            if ref_n == fine_n:
                ref_z1, ref_z2 = z1_fine, z2_fine
            else:
                ref_z1 = _aggregate_normals(z1_fine, ref_n)
                ref_z2 = _aggregate_normals(z2_fine, ref_n)
            ref_terminals = simulate_heston_terminals_from_normals(
                params, S0, r, T_grid, ref_z1, ref_z2, scheme=scheme
            )
            ref_sim_time = perf_counter() - ref_t0
            ref_prices, _ = price_surface_from_terminals(ref_terminals, K_grid, T_grid, r, return_se=False)
            ref_prices_for_iv = ref_prices
            ref_repaired = 0
            if clip_prices_for_iv:
                ref_prices_for_iv, ref_repaired = clip_price_surface_to_bounds(ref_prices_for_iv, S0, K_grid, T_grid, r)
            ref_ivs, ref_fails = iv_surface_from_prices(ref_prices_for_iv, S0, K_grid, T_grid, r)
            _ = (ref_repaired, ref_fails)
        else:
            ref = mc_iv_surface(
                params,
                S0,
                r,
                K_grid,
                T_grid,
                M,
                ref_n,
                seed=seed,
                scheme=scheme,
                antithetic=antithetic,
                return_price_se=False,
                clip_prices_for_iv=clip_prices_for_iv,
            )
            ref_prices = ref["prices"]
            ref_ivs = ref["ivs"]
            ref_sim_time = float(ref["runtime_simulation_s"])

        for n_steps in n_steps_values:
            if coupled:
                if n_steps == ref_n:
                    prices = ref_prices
                    ivs = ref_ivs
                    runtime_sim = ref_sim_time
                    repaired = 0
                    fails = 0
                else:
                    z1 = _aggregate_normals(z1_fine, n_steps)
                    z2 = _aggregate_normals(z2_fine, n_steps)
                    t0 = perf_counter()
                    terminals = simulate_heston_terminals_from_normals(
                        params, S0, r, T_grid, z1, z2, scheme=scheme
                    )
                    runtime_sim = perf_counter() - t0
                    prices, _ = price_surface_from_terminals(terminals, K_grid, T_grid, r, return_se=False)
                    prices_for_iv = prices
                    repaired = 0
                    if clip_prices_for_iv:
                        prices_for_iv, repaired = clip_price_surface_to_bounds(
                            prices_for_iv, S0, K_grid, T_grid, r
                        )
                    ivs, fails = iv_surface_from_prices(prices_for_iv, S0, K_grid, T_grid, r)
            else:
                out = mc_iv_surface(
                    params,
                    S0,
                    r,
                    K_grid,
                    T_grid,
                    M,
                    n_steps,
                    seed=seed,
                    scheme=scheme,
                    antithetic=antithetic,
                    return_price_se=False,
                    clip_prices_for_iv=clip_prices_for_iv,
                )
                prices = out["prices"]
                ivs = out["ivs"]
                runtime_sim = float(out["runtime_simulation_s"])
                repaired = int(out["repaired_prices_for_iv"])
                fails = int(out["iv_failures"])

            rows.append(
                {
                    "seed": int(seed),
                    "n_steps": int(n_steps),
                    "mean_abs_price_diff_vs_maxN": float(np.nanmean(np.abs(prices - ref_prices))),
                    "max_abs_price_diff_vs_maxN": float(np.nanmax(np.abs(prices - ref_prices))),
                    "mean_abs_iv_diff_vs_maxN": float(np.nanmean(np.abs(ivs - ref_ivs))),
                    "max_abs_iv_diff_vs_maxN": float(np.nanmax(np.abs(ivs - ref_ivs))),
                    "runtime_simulation_s": float(runtime_sim),
                    "iv_failures": int(fails),
                    "repaired_prices_for_iv": int(repaired),
                }
            )

    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby("n_steps", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_abs_price_diff_vs_maxN=("mean_abs_price_diff_vs_maxN", "mean"),
            std_abs_price_diff_vs_maxN=("mean_abs_price_diff_vs_maxN", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            max_abs_price_diff_vs_maxN=("max_abs_price_diff_vs_maxN", "mean"),
            mean_abs_iv_diff_vs_maxN=("mean_abs_iv_diff_vs_maxN", "mean"),
            std_abs_iv_diff_vs_maxN=("mean_abs_iv_diff_vs_maxN", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            max_abs_iv_diff_vs_maxN=("max_abs_iv_diff_vs_maxN", "mean"),
            mean_runtime_simulation_s=("runtime_simulation_s", "mean"),
            avg_iv_failures=("iv_failures", "mean"),
            avg_repaired_prices_for_iv=("repaired_prices_for_iv", "mean"),
        )
        .sort_values("n_steps")
        .reset_index(drop=True)
    )
    return agg



def benchmark_reference_accuracy(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    seed: int = 123,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Compare the MC surface to the semi-closed-form Heston European call surface."""
    mc = mc_iv_surface(
        params,
        S0,
        r,
        K_grid,
        T_grid,
        M,
        n_steps,
        seed=seed,
        scheme=scheme,
        antithetic=antithetic,
        return_price_se=True,
        clip_prices_for_iv=clip_prices_for_iv,
    )
    cf_prices = heston_price_surface_cf(params, S0, r, K_grid, T_grid)
    cf_ivs, _ = iv_surface_from_prices(cf_prices, S0, K_grid, T_grid, r)

    rows: List[Dict[str, float]] = []
    K_arr = np.asarray(K_grid, dtype=float)
    T_arr = np.asarray(T_grid, dtype=float)
    price_se = mc.get("price_se")

    for j, T in enumerate(T_arr):
        for i, K in enumerate(K_arr):
            rows.append(
                {
                    "T": float(T),
                    "K": float(K),
                    "mc_price": float(mc["prices"][j, i]),
                    "cf_price": float(cf_prices[j, i]),
                    "abs_price_error": float(abs(mc["prices"][j, i] - cf_prices[j, i])),
                    "mc_iv": float(mc["ivs"][j, i]),
                    "cf_iv": float(cf_ivs[j, i]),
                    "abs_iv_error": float(abs(mc["ivs"][j, i] - cf_ivs[j, i])),
                    "mc_price_se": float(price_se[j, i]) if price_se is not None else np.nan,
                }
            )

    grid = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "M": int(M),
                "n_steps": int(n_steps),
                "mean_abs_price_error": float(grid["abs_price_error"].mean()),
                "max_abs_price_error": float(grid["abs_price_error"].max()),
                "rmse_price": float(np.sqrt(np.mean(grid["abs_price_error"] ** 2))),
                "mean_abs_iv_error": float(grid["abs_iv_error"].mean()),
                "max_abs_iv_error": float(grid["abs_iv_error"].max()),
                "rmse_iv": float(np.sqrt(np.mean(grid["abs_iv_error"] ** 2))),
                "mean_mc_price_se": float(grid["mc_price_se"].mean()),
                "iv_failures": int(mc["iv_failures"]),
                "repaired_prices_for_iv": int(mc["repaired_prices_for_iv"]),
                "runtime_total_s": float(mc["runtime_total_s"]),
            }
        ]
    )
    return {"summary": summary, "grid": grid}



def benchmark_runtime(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    repeats: int = 3,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> pd.DataFrame:
    """Measure wall-clock runtime for one full MC -> IV surface."""
    runtimes: List[float] = []
    sim_times: List[float] = []
    iv_times: List[float] = []
    for rep in range(repeats):
        out = mc_iv_surface(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M,
            n_steps,
            seed=12345 + rep,
            scheme=scheme,
            antithetic=antithetic,
            return_price_se=False,
            clip_prices_for_iv=clip_prices_for_iv,
        )
        runtimes.append(float(out["runtime_total_s"]))
        sim_times.append(float(out["runtime_simulation_s"]))
        iv_times.append(float(out["runtime_iv_inversion_s"]))

    std = float(np.std(runtimes, ddof=1)) if repeats > 1 else 0.0
    return pd.DataFrame(
        [
            {
                "repeats": int(repeats),
                "M": int(M),
                "n_steps": int(n_steps),
                "antithetic": bool(antithetic),
                "mean_runtime_s": float(np.mean(runtimes)),
                "std_runtime_s": std,
                "mean_simulation_s": float(np.mean(sim_times)),
                "mean_iv_inversion_s": float(np.mean(iv_times)),
                "min_runtime_s": float(np.min(runtimes)),
                "max_runtime_s": float(np.max(runtimes)),
            }
        ]
    )



def benchmark_suite(
    params: HestonParams,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M_main: int = 10_000,
    M_stability: int = 5_000,
    n_steps: int = 400,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Run the standard five-benchmark suite in one call."""
    return {
        "seed_stability": benchmark_seed_stability(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M=M_stability,
            n_steps=n_steps,
            seeds=[1, 2, 3, 4, 5],
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
        )["summary"],
        "path_convergence": benchmark_path_convergence(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M_values=[1_000, 2_000, 5_000, M_main],
            n_steps=n_steps,
            seed=123,
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
        ),
        "timestep_convergence": benchmark_timestep_convergence(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M=M_stability,
            n_steps_values=[80, 160, 400, 800],
            seeds=[11, 22, 33],
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
            coupled=True,
        ),
        "reference_accuracy": benchmark_reference_accuracy(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M=M_main,
            n_steps=n_steps,
            seed=123,
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
        )["summary"],
        "runtime": benchmark_runtime(
            params,
            S0,
            r,
            K_grid,
            T_grid,
            M=M_main,
            n_steps=n_steps,
            repeats=3,
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
        ),
    }


# ---------------------------------------------------------------------
# Parameter sampling and dataset generation
# ---------------------------------------------------------------------
def sample_heston_parameters(
    n_samples: int,
    seed: int = 123,
    rho_range: Tuple[float, float] = (-0.95, 0.0),
    kappa_range: Tuple[float, float] = (0.5, 6.0),
    sigma_v_range: Tuple[float, float] = (0.1, 1.5),
    v0_range: Tuple[float, float] = (0.0025, 0.36),
    theta_range: Tuple[float, float] = (0.0025, 0.36),
    log_sample_positive: bool = True,
    enforce_feller: bool = False,
    max_draws_multiplier: int = 20,
) -> List[HestonParams]:
    """Simple random parameter sampler for dataset generation."""
    rng = np.random.default_rng(seed)
    out: List[HestonParams] = []

    def draw_positive(lo: float, hi: float, size: int) -> np.ndarray:
        if log_sample_positive:
            return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))
        return rng.uniform(lo, hi, size=size)

    max_draws = max_draws_multiplier * n_samples
    draws = 0
    while len(out) < n_samples and draws < max_draws:
        draws += 1
        v0 = float(draw_positive(*v0_range, size=1)[0])
        rho = float(rng.uniform(*rho_range))
        kappa = float(draw_positive(*kappa_range, size=1)[0])
        theta = float(draw_positive(*theta_range, size=1)[0])
        sigma_v = float(draw_positive(*sigma_v_range, size=1)[0])

        if enforce_feller and (2.0 * kappa * theta / (sigma_v * sigma_v) < 1.0):
            continue
        out.append(HestonParams(v0=v0, rho=rho, kappa=kappa, theta=theta, sigma_v=sigma_v))

    if len(out) < n_samples:
        raise RuntimeError(
            f"Could only sample {len(out)} parameter vectors out of {n_samples}. "
            "Relax constraints or increase max_draws_multiplier."
        )
    return out



def _row_from_surface_output(
    params: HestonParams,
    out: Dict[str, Any],
    K_grid: ArrayLike,
    T_grid: ArrayLike,
) -> Dict[str, float]:
    """Convert one surface output dictionary into a flat dataframe row."""
    row: Dict[str, float] = asdict(params)
    sanity = out["sanity_checks"]
    row.update(
        {
            "iv_failures": int(out["iv_failures"]),
            "repaired_prices_for_iv": int(out["repaired_prices_for_iv"]),
            "runtime_total_s": float(out["runtime_total_s"]),
            "runtime_simulation_s": float(out["runtime_simulation_s"]),
            "runtime_iv_inversion_s": float(out["runtime_iv_inversion_s"]),
            "viol_lower_bound": int(sanity["viol_lower_bound"]),
            "viol_upper_bound": int(sanity["viol_upper_bound"]),
            "monotonicity_violations_across_strike": int(sanity["monotonicity_violations_across_strike"]),
            "convexity_violations_across_strike": int(sanity["convexity_violations_across_strike"]),
            "monotonicity_violations_across_maturity": int(sanity["monotonicity_violations_across_maturity"]),
        }
    )
    iv_cols = iv_column_names(K_grid, T_grid)
    for col, val in zip(iv_cols, np.asarray(out["iv_vector"], dtype=float)):
        row[col] = float(val)
    return row



def generate_mc_dataset(
    params_list: Sequence[HestonParams],
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    base_seed: int = 1000,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
) -> pd.DataFrame:
    """
    Generate a dataframe of Monte Carlo IV surfaces for DNN training.

    This function keeps every row, including potentially noisy rows, but stores
    diagnostic columns so you can filter later.
    """
    rows: List[Dict[str, float]] = []
    for idx, params in enumerate(params_list):
        out = mc_iv_surface(
            params=params,
            S0=S0,
            r=r,
            K_grid=K_grid,
            T_grid=T_grid,
            M=M,
            n_steps=n_steps,
            seed=base_seed + idx,
            scheme=scheme,
            antithetic=antithetic,
            return_price_se=False,
            clip_prices_for_iv=clip_prices_for_iv,
        )
        rows.append(_row_from_surface_output(params, out, K_grid, T_grid))
    return pd.DataFrame(rows)



def generate_clean_mc_dataset(
    n_samples: int,
    S0: float,
    r: float,
    K_grid: ArrayLike,
    T_grid: ArrayLike,
    M: int,
    n_steps: int,
    sampler_seed: int = 2026,
    base_seed: int = 5000,
    scheme: str = "full_truncation",
    antithetic: bool = False,
    clip_prices_for_iv: bool = False,
    require_zero_iv_failures: bool = True,
    require_no_bound_violations: bool = True,
    require_strike_monotonicity: bool = True,
    require_strike_convexity: bool = False,
    allow_repaired_prices_for_iv: bool = False,
    max_attempts_multiplier: int = 20,
    param_sampling_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generate a clean Monte Carlo IV dataset by automatic resampling/retry.

    New parameter vectors are sampled one at a time until exactly n_samples rows pass the
    requested quality filters.
    """
    param_sampling_kwargs = dict(param_sampling_kwargs or {})
    accepted_rows: List[Dict[str, float]] = []
    attempts = 0
    max_attempts = max_attempts_multiplier * n_samples

    while len(accepted_rows) < n_samples and attempts < max_attempts:
        params = sample_heston_parameters(
            n_samples=1,
            seed=sampler_seed + attempts,
            **param_sampling_kwargs,
        )[0]
        out = mc_iv_surface(
            params=params,
            S0=S0,
            r=r,
            K_grid=K_grid,
            T_grid=T_grid,
            M=M,
            n_steps=n_steps,
            seed=base_seed + attempts,
            scheme=scheme,
            antithetic=antithetic,
            return_price_se=False,
            clip_prices_for_iv=clip_prices_for_iv,
        )
        attempts += 1

        if surface_quality_ok(
            out,
            require_zero_iv_failures=require_zero_iv_failures,
            require_no_bound_violations=require_no_bound_violations,
            require_strike_monotonicity=require_strike_monotonicity,
            require_strike_convexity=require_strike_convexity,
            allow_repaired_prices_for_iv=allow_repaired_prices_for_iv,
        ):
            row = _row_from_surface_output(params, out, K_grid, T_grid)
            row["accepted_attempt"] = attempts
            accepted_rows.append(row)

    if len(accepted_rows) < n_samples:
        raise RuntimeError(
            f"Only accepted {len(accepted_rows)} clean rows out of the required {n_samples} "
            f"after {attempts} attempts. Increase M, allow clipping, relax quality filters, "
            "or increase max_attempts_multiplier."
        )

    return pd.DataFrame(accepted_rows)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    S0 = 100.0
    r = 0.02

    T_grid = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
    K_grid = np.arange(90.0, 111.0, 1.0)  # 90, 91, ..., 110  -> 21 strikes

    theta_example = HestonParams(v0=0.04, rho=-0.70, kappa=2.0, theta=0.04, sigma_v=0.50)

    out = mc_iv_surface(
        params=theta_example,
        S0=S0,
        r=r,
        K_grid=K_grid,
        T_grid=T_grid,
        M=10_000,
        n_steps=400,
        seed=123,
        scheme="full_truncation",
        antithetic=False,
        return_price_se=True,
        clip_prices_for_iv=False,
    )

    print("\n=== One Monte Carlo IV surface ===")
    print("IV failures:", out["iv_failures"])
    print("Repaired prices for IV:", out["repaired_prices_for_iv"])
    print("Total runtime (s):", round(float(out["runtime_total_s"]), 4))
    print("Sanity checks:", out["sanity_checks"])

    print("\n=== Improved time-step convergence (coupled, averaged over seeds) ===")
    time_conv = benchmark_timestep_convergence(
        params=theta_example,
        S0=S0,
        r=r,
        K_grid=K_grid,
        T_grid=T_grid,
        M=5_000,
        n_steps_values=[80, 160, 400, 800],
        seeds=[11, 22, 33],
        scheme="full_truncation",
        coupled=True,
    )
    print(time_conv)

    print("\n=== Clean pilot dataset (automatic retry) ===")
    pilot_clean = generate_clean_mc_dataset(
        n_samples=5,
        S0=S0,
        r=r,
        K_grid=K_grid,
        T_grid=T_grid,
        M=5_000,
        n_steps=400,
        sampler_seed=2026,
        base_seed=5000,
        scheme="full_truncation",
        antithetic=False,
        clip_prices_for_iv=False,
    )
    print(pilot_clean.head())
