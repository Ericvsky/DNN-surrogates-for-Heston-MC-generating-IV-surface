from __future__ import annotations

"""
Monte Carlo baseline diagnostics for the restarted 8 x 15 Heston IV grid.

This script generates the Section-2-style MC checks for the new grid:
  - one MC price/IV surface
  - IV smiles by maturity
  - seed-stability heatmap
  - path-count convergence plot
  - time-step convergence plot
  - semi-closed-form Heston reference comparison

The production setting remains M=5000, n_steps=400, full truncation, antithetic=True.
Outputs are saved to mc_benchmark_8x15_output/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from heston_mc_benchmark_toolkit import (
    HestonParams,
    mc_iv_surface,
    benchmark_seed_stability,
    benchmark_path_convergence,
    benchmark_timestep_convergence,
    benchmark_reference_accuracy,
)

S0 = 100.0
R = 0.02
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)
assert len(K_GRID) == 15 and abs(K_GRID[0] - 90.0) < 1e-12 and abs(K_GRID[-1] - 118.0) < 1e-12
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)

M = 5_000
N_STEPS = 400
SCHEME = "full_truncation"
ANTITHETIC = True
CLIP_PRICES_FOR_IV = False

# A representative Heston parameter set for MC diagnostics.
PARAMS = HestonParams(v0=0.04, rho=-0.70, kappa=2.00, theta=0.04, sigma_v=0.40)


def save_heatmap(data: np.ndarray, title: str, out_path: Path, colorbar_label: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.imshow(data, aspect="auto", origin="lower")
    plt.xticks(range(len(K_GRID)), [f"{k:.1f}" for k in K_GRID], rotation=45)
    plt.yticks(range(len(T_GRID)), [f"{t:.2f}" for t in T_GRID])
    plt.xlabel("Strike K")
    plt.ylabel("Maturity T")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_iv_smiles(ivs: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for j, T in enumerate(T_GRID):
        plt.plot(K_GRID, ivs[j], marker="o", label=f"T={T:.2f}")
    plt.xlabel("Strike K")
    plt.ylabel("Implied volatility")
    plt.title("8x15 MC implied-volatility smiles by maturity")
    plt.grid(True, alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_iv_surface_3d(ivs: np.ndarray, out_path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    kk, tt = np.meshgrid(K_GRID, T_GRID)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(kk, tt, ivs, linewidth=0.4, antialiased=True, alpha=0.9)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Monte Carlo Heston IV surface, 8x15 grid")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_convergence_plot(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, out_path: Path, log_x: bool = False) -> None:
    plt.figure(figsize=(9, 5))
    for col in y_cols:
        plt.plot(df[x_col], df[col], marker="o", label=col)
    if log_x:
        plt.xscale("log")
    plt.xlabel(x_col)
    plt.ylabel("absolute IV difference vs reference level")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    out_dir = Path("mc_benchmark_8x15_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running 8x15 MC benchmark diagnostics")
    print(f"Output folder: {out_dir.resolve()}")
    print(f"Grid: {len(T_GRID)} x {len(K_GRID)} = {len(T_GRID) * len(K_GRID)}")
    print(f"M={M}, n_steps={N_STEPS}, scheme={SCHEME}, antithetic={ANTITHETIC}")

    one = mc_iv_surface(
        params=PARAMS,
        S0=S0,
        r=R,
        K_grid=K_GRID,
        T_grid=T_GRID,
        M=M,
        n_steps=N_STEPS,
        seed=123,
        scheme=SCHEME,
        antithetic=ANTITHETIC,
        return_price_se=True,
        clip_prices_for_iv=CLIP_PRICES_FOR_IV,
    )
    print("\nOne MC surface")
    print(f"IV failures: {one['iv_failures']}")
    print(f"Repaired prices for IV: {one['repaired_prices_for_iv']}")
    print(f"Runtime total (s): {one['runtime_total_s']:.6f}")
    print(f"Sanity checks: {one['sanity_checks']}")

    save_heatmap(one["prices"], "Monte Carlo Heston call-price surface, 8x15", out_dir / "mc_price_heatmap_8x15.png", "Call price")
    save_heatmap(one["ivs"], "Monte Carlo Heston implied-volatility surface, 8x15", out_dir / "mc_iv_heatmap_8x15.png", "Implied volatility")
    if one.get("price_se") is not None:
        save_heatmap(one["price_se"], "Monte Carlo price standard-error surface, 8x15", out_dir / "mc_price_se_heatmap_8x15.png", "Price standard error")
    save_iv_smiles(one["ivs"], out_dir / "iv_smiles_by_maturity_8x15.png")
    save_iv_surface_3d(one["ivs"], out_dir / "mc_iv_surface_3d_8x15.png")

    seed = benchmark_seed_stability(
        PARAMS, S0, R, K_GRID, T_GRID, M=M, n_steps=N_STEPS, seeds=[1, 2, 3, 4, 5],
        scheme=SCHEME, antithetic=ANTITHETIC, clip_prices_for_iv=CLIP_PRICES_FOR_IV,
    )
    seed["summary"].to_csv(out_dir / "seed_stability_summary_8x15.csv", index=False)
    seed["iv_std_surface"].to_csv(out_dir / "seed_iv_std_surface_8x15.csv")
    save_heatmap(seed["iv_std_surface"].to_numpy(), "Seed-stability: IV standard deviation, 8x15", out_dir / "seed_iv_std_heatmap_8x15.png", "IV standard deviation")

    path_df = benchmark_path_convergence(
        PARAMS, S0, R, K_GRID, T_GRID, M_values=[1000, 2000, 5000, 10000], n_steps=N_STEPS,
        seed=123, scheme=SCHEME, antithetic=ANTITHETIC, clip_prices_for_iv=CLIP_PRICES_FOR_IV,
    )
    path_df.to_csv(out_dir / "path_convergence_8x15.csv", index=False)
    save_convergence_plot(
        path_df,
        "M",
        ["mean_abs_iv_diff_vs_maxM", "max_abs_iv_diff_vs_maxM"],
        "Path-count convergence, 8x15",
        out_dir / "path_convergence_iv_8x15.png",
        log_x=True,
    )

    step_df = benchmark_timestep_convergence(
        PARAMS, S0, R, K_GRID, T_GRID, M=M, n_steps_values=[80, 160, 400, 800], seeds=[11, 22, 33],
        scheme=SCHEME, antithetic=ANTITHETIC, clip_prices_for_iv=CLIP_PRICES_FOR_IV, coupled=True,
    )
    step_df.to_csv(out_dir / "timestep_convergence_8x15.csv", index=False)
    save_convergence_plot(
        step_df,
        "n_steps",
        ["mean_abs_iv_diff_vs_maxN", "max_abs_iv_diff_vs_maxN"],
        "Time-step convergence, 8x15",
        out_dir / "timestep_convergence_iv_8x15.png",
        log_x=False,
    )

    ref = benchmark_reference_accuracy(
        PARAMS, S0, R, K_GRID, T_GRID, M=M, n_steps=N_STEPS, seed=123,
        scheme=SCHEME, antithetic=ANTITHETIC, clip_prices_for_iv=CLIP_PRICES_FOR_IV,
    )
    ref["summary"].to_csv(out_dir / "reference_accuracy_summary_8x15.csv", index=False)
    ref["grid"].to_csv(out_dir / "reference_accuracy_grid_8x15.csv", index=False)
    abs_iv_grid = ref["grid"].pivot(index="T", columns="K", values="abs_iv_error").loc[T_GRID, K_GRID].to_numpy()
    save_heatmap(abs_iv_grid, "Absolute IV error vs Heston reference, 8x15", out_dir / "reference_abs_iv_error_heatmap_8x15.png", "Absolute IV error")

    with open(out_dir / "mc_benchmark_summary_8x15.txt", "w", encoding="utf-8") as f:
        f.write("Monte Carlo benchmark summary: 8x15 restart\n")
        f.write("=" * 55 + "\n")
        f.write(f"Grid: {len(T_GRID)} x {len(K_GRID)} = {len(T_GRID) * len(K_GRID)}\n")
        f.write(f"K_grid={K_GRID.tolist()}\n")
        f.write(f"T_grid={T_GRID.tolist()}\n")
        f.write(f"M={M}\n")
        f.write(f"n_steps={N_STEPS}\n")
        f.write(f"scheme={SCHEME}\n")
        f.write(f"antithetic={ANTITHETIC}\n")
        f.write(f"one_surface_runtime_s={one['runtime_total_s']:.8f}\n")
        f.write(f"one_surface_iv_failures={one['iv_failures']}\n")
        f.write(f"one_surface_repaired_prices_for_iv={one['repaired_prices_for_iv']}\n")
        f.write("\nSeed stability summary\n")
        f.write(seed["summary"].to_string(index=False))
        f.write("\n\nReference accuracy summary\n")
        f.write(ref["summary"].to_string(index=False))
        f.write("\n")

    print("\nSaved MC benchmark outputs in:", out_dir)
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p)


if __name__ == "__main__":
    main()
