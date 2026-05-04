from __future__ import annotations

"""
Prepare derived total-variance and Black--Scholes price labels for the 8 x 15 Heston dataset.

Version: v4.1, 8x15 K90--118 restart
--------------------------------------
This script does NOT regenerate Monte Carlo paths. It reads the existing implied-volatility
training dataset and derives two finance-aware label sets:

    1. Total implied variance:
           w(T,K) = sigma_IV(T,K)^2 * T

    2. Black--Scholes call prices implied by the IV labels:
           C(T,K) = BS(S0, K, T, r, sigma_IV(T,K))

The derived files are saved as CSV "spreadsheet" files for documentation and optional reuse.
The v4.1 training script can also compute these labels in memory, so this script is mainly
for traceability and for reporting the redesigned training target.

Expected source dataset shape:
    10000 rows x 125 columns = 5 Heston parameters + 120 IV targets

Expected outputs:
    derived_dataset_10000_8x15_v4_1/heston_dataset_10000_8x15_total_variance_train_only_v4.1.csv
    derived_dataset_10000_8x15_v4_1/heston_dataset_10000_8x15_bs_price_train_only_v4.1.csv
    derived_dataset_10000_8x15_v4_1/heston_dataset_10000_8x15_iv_totalvar_price_combined_v4.1.csv
    derived_dataset_10000_8x15_v4_1/derived_dataset_metadata_v4.1.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

FEATURE_COLS = ["v0", "rho", "kappa", "theta", "sigma_v"]
S0 = 100.0
R = 0.02
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)
SURFACE_DIM = len(T_GRID) * len(K_GRID)


def expected_iv_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"iv_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def total_variance_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"w_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def price_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"price_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def bs_call_prices_from_iv_matrix(iv_matrix: np.ndarray, s0: float, r: float, k_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Vectorised Black--Scholes call prices for an (N, 8*15) IV matrix."""
    iv = np.asarray(iv_matrix, dtype=np.float64)
    k_flat = np.tile(k_grid, len(t_grid)).astype(np.float64)
    t_flat = np.repeat(t_grid, len(k_grid)).astype(np.float64)

    sigma = np.maximum(iv, 1e-12)
    sqrt_t = np.sqrt(t_flat)[None, :]
    denom = sigma * sqrt_t
    d1 = (np.log(s0 / k_flat)[None, :] + (r + 0.5 * sigma ** 2) * t_flat[None, :]) / denom
    d2 = d1 - denom
    prices = s0 * ndtr(d1) - k_flat[None, :] * np.exp(-r * t_flat)[None, :] * ndtr(d2)
    return prices.astype(np.float32)


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / "heston_dataset_10000_8x15_output" / "heston_dataset_10000_8x15_clean_train_only.csv"
    out_dir = base / "derived_dataset_10000_8x15_v4_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Source IV training CSV not found: {data_path}")

    print(f"Loading source IV dataset from: {data_path}")
    print(f"Saving derived datasets to: {out_dir}")
    print(f"Grid: {len(T_GRID)} maturities x {len(K_GRID)} strikes = {SURFACE_DIM} outputs")
    print(f"Strike grid: {K_GRID.tolist()}")

    df = pd.read_csv(data_path)
    iv_cols = [c for c in df.columns if c.startswith("iv_T")]
    expected_cols = expected_iv_columns(K_GRID, T_GRID)

    if iv_cols != expected_cols:
        raise ValueError(
            "IV target columns do not match the expected 8x15 K=90,92,...,118 grid.\n"
            f"First expected: {expected_cols[:3]} ... last expected: {expected_cols[-3:]}\n"
            f"First found:    {iv_cols[:3]} ... last found:    {iv_cols[-3:]}"
        )
    if len(iv_cols) != SURFACE_DIM:
        raise ValueError(f"Expected {SURFACE_DIM} IV columns, found {len(iv_cols)}")
    if df[FEATURE_COLS + iv_cols].isna().any().any():
        raise ValueError("NaNs found in feature or IV columns.")

    iv = df[iv_cols].to_numpy(dtype=np.float32)
    t_flat = np.repeat(T_GRID, len(K_GRID)).astype(np.float32)

    total_var = (iv ** 2) * t_flat[None, :]
    prices = bs_call_prices_from_iv_matrix(iv, S0, R, K_GRID, T_GRID)

    w_cols = total_variance_columns(K_GRID, T_GRID)
    c_cols = price_columns(K_GRID, T_GRID)

    df_w = pd.concat([df[FEATURE_COLS].reset_index(drop=True), pd.DataFrame(total_var, columns=w_cols)], axis=1)
    df_c = pd.concat([df[FEATURE_COLS].reset_index(drop=True), pd.DataFrame(prices, columns=c_cols)], axis=1)
    df_combined = pd.concat(
        [
            df[FEATURE_COLS].reset_index(drop=True),
            df[iv_cols].reset_index(drop=True),
            pd.DataFrame(total_var, columns=w_cols),
            pd.DataFrame(prices, columns=c_cols),
        ],
        axis=1,
    )

    w_path = out_dir / "heston_dataset_10000_8x15_total_variance_train_only_v4.1.csv"
    c_path = out_dir / "heston_dataset_10000_8x15_bs_price_train_only_v4.1.csv"
    combined_path = out_dir / "heston_dataset_10000_8x15_iv_totalvar_price_combined_v4.1.csv"
    metadata_path = out_dir / "derived_dataset_metadata_v4.1.txt"

    df_w.to_csv(w_path, index=False)
    df_c.to_csv(c_path, index=False)
    df_combined.to_csv(combined_path, index=False)

    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Derived dataset metadata for v4.1 total-variance DNN\n")
        f.write("=" * 60 + "\n")
        f.write(f"source_iv_dataset={data_path}\n")
        f.write(f"S0={S0}\n")
        f.write(f"r={R}\n")
        f.write(f"T_grid={T_GRID.tolist()}\n")
        f.write(f"K_grid={K_GRID.tolist()}\n")
        f.write(f"surface_dim={SURFACE_DIM}\n")
        f.write(f"rows={len(df)}\n")
        f.write(f"total_variance_columns={len(w_cols)}\n")
        f.write(f"price_columns={len(c_cols)}\n")
        f.write("total_variance_formula=w(T,K)=IV(T,K)^2*T\n")
        f.write("price_formula=Black-Scholes call price implied by IV label\n")
        f.write(f"total_variance_csv={w_path.name}\n")
        f.write(f"price_csv={c_path.name}\n")
        f.write(f"combined_csv={combined_path.name}\n")

    print("\nDerived dataset files saved:")
    print(f" - {w_path}")
    print(f" - {c_path}")
    print(f" - {combined_path}")
    print(f" - {metadata_path}")
    print("\nShapes:")
    print(f"Total-variance train-only: {df_w.shape}")
    print(f"BS-price train-only:       {df_c.shape}")
    print(f"Combined audit dataset:    {df_combined.shape}")
    print("\nExpected:")
    print("Total-variance train-only should be (10000, 125).")
    print("BS-price train-only should be (10000, 125).")
    print("Combined audit dataset should be (10000, 365).")


if __name__ == "__main__":
    main()
