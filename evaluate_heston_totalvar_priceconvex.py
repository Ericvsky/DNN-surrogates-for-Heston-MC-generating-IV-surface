from __future__ import annotations

"""
Evaluate v4.1: Total-Variance DNN with Price-Convexity Loss on the 8 x 15 Heston dataset.

This evaluator reloads the v4.1 checkpoint and computes:
    - IV MAE/RMSE;
    - total-variance MAE/RMSE;
    - Black--Scholes price MAE/RMSE;
    - hard surface-quality checks on predicted and true BS price surfaces;
    - best/median/worst IV surface plots and mean absolute-error heatmap.

The model pipeline is:

    Theta -> positive total variance w(T,K) -> IV -> BS call price.

The hard strike-convexity check is still applied in price space:

    C(K_{i+1}) - 2*C(K_i) + C(K_{i-1}) >= 0.
"""

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import ndtr
from torch import nn

FEATURE_COLS = ["v0", "rho", "kappa", "theta", "sigma_v"]
LOG_FEATURE_COLS = ["v0", "kappa", "theta", "sigma_v"]
INPUT_DIM = 5
LATENT_DIM = 32
COORD_DIM = 5
S0 = 100.0
R = 0.02
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)
OUTPUT_DIM = len(T_GRID) * len(K_GRID)
EPS_TOTAL_VARIANCE = 1e-8
CONVEXITY_TOL = 1e-10


class SurfaceCoordinateTotalVariancePriceConvexDNN(nn.Module):
    def __init__(self, coord_features_scaled: np.ndarray, k_grid: np.ndarray, t_grid: np.ndarray):
        super().__init__()
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.output_dim = len(self.k_grid) * len(self.t_grid)

        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, LATENT_DIM),
            nn.Softplus(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + COORD_DIM, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

        self.register_buffer("coord_features", torch.tensor(coord_features_scaled, dtype=torch.float32))
        t_flat = np.repeat(self.t_grid, len(self.k_grid)).astype(np.float32)
        self.register_buffer("t_flat", torch.tensor(t_flat, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        z = self.encoder(x)
        z_exp = z[:, None, :].expand(batch, self.output_dim, LATENT_DIM)
        c_exp = self.coord_features[None, :, :].expand(batch, self.output_dim, COORD_DIM)
        dec_in = torch.cat([z_exp, c_exp], dim=-1)
        raw_w = self.decoder(dec_in.reshape(batch * self.output_dim, LATENT_DIM + COORD_DIM)).reshape(batch, self.output_dim)
        w_pred = F.softplus(raw_w) + EPS_TOTAL_VARIANCE
        iv_pred = torch.sqrt(w_pred / self.t_flat[None, :])
        return iv_pred, w_pred


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def expected_iv_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"iv_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def standardize_np(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe = np.where(scale == 0.0, 1.0, scale)
    return ((x - mean) / safe).astype(np.float32)


def total_variance_from_iv(iv: np.ndarray, k_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_flat = np.repeat(t_grid, len(k_grid)).astype(np.float32)
    return (iv.astype(np.float32) ** 2) * t_flat[None, :]


def normal_cdf_np(x: np.ndarray) -> np.ndarray:
    return ndtr(x)


def bs_call_price_np(iv: np.ndarray, s0: float, r: float, k_grid: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    iv = np.asarray(iv, dtype=np.float64)
    k_flat = np.tile(k_grid, len(t_grid)).astype(np.float64)
    t_flat = np.repeat(t_grid, len(k_grid)).astype(np.float64)
    sigma = np.maximum(iv, 1e-12)
    sqrt_t = np.sqrt(t_flat)[None, :]
    vol_sqrt_t = sigma * sqrt_t
    d1 = (np.log(s0 / k_flat)[None, :] + (r + 0.5 * sigma ** 2) * t_flat[None, :]) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    price = s0 * normal_cdf_np(d1) - k_flat[None, :] * np.exp(-r * t_flat)[None, :] * normal_cdf_np(d2)
    return price.astype(np.float32)


def surface_sanity_checks(prices: np.ndarray, s0: float, r: float, k_grid: np.ndarray, t_grid: np.ndarray, invalid_nodes: int = 0) -> dict[str, int | float]:
    prices = np.asarray(prices, dtype=float)
    lower = np.zeros_like(prices)
    upper = np.full_like(prices, s0)
    for i, t in enumerate(t_grid):
        lower[i, :] = np.maximum(s0 - k_grid * np.exp(-r * t), 0.0)

    diff_k = prices[:, 1:] - prices[:, :-1]
    second_k = prices[:, 2:] - 2.0 * prices[:, 1:-1] + prices[:, :-2]
    diff_t = prices[1:, :] - prices[:-1, :]

    return {
        "min_price": float(np.nanmin(prices)),
        "max_price": float(np.nanmax(prices)),
        "invalid_or_negative_iv_nodes": int(invalid_nodes),
        "viol_lower_bound": int(np.sum(prices < lower - CONVEXITY_TOL)),
        "viol_upper_bound": int(np.sum(prices > upper + CONVEXITY_TOL)),
        "monotonicity_violations_across_strike": int(np.sum(diff_k > CONVEXITY_TOL)),
        "convexity_violations_across_strike": int(np.sum(second_k < -CONVEXITY_TOL)),
        "monotonicity_violations_across_maturity": int(np.sum(diff_t < -CONVEXITY_TOL)),
    }


def iv_surface_to_price_surface(iv_surface: np.ndarray, s0: float, r: float, k_grid: np.ndarray, t_grid: np.ndarray) -> tuple[np.ndarray, int]:
    invalid = int(np.sum(~np.isfinite(iv_surface) | (iv_surface < 0.0)))
    iv_safe = np.where(np.isfinite(iv_surface), np.maximum(iv_surface, 0.0), 0.0).reshape(1, -1)
    prices = bs_call_price_np(iv_safe, s0, r, k_grid, t_grid).reshape(len(t_grid), len(k_grid))
    return prices, invalid


def summarize_quality(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    invalid = f"{prefix}_invalid_or_negative_iv_nodes"
    lb = f"{prefix}_viol_lower_bound"
    ub = f"{prefix}_viol_upper_bound"
    mono_k = f"{prefix}_monotonicity_violations_across_strike"
    conv_k = f"{prefix}_convexity_violations_across_strike"
    mono_t = f"{prefix}_monotonicity_violations_across_maturity"
    summary = {
        f"pct_{prefix}_all_finite_nonnegative_iv": float(np.mean(df[invalid] == 0) * 100.0),
        f"pct_{prefix}_no_lower_bound_viol": float(np.mean(df[lb] == 0) * 100.0),
        f"pct_{prefix}_no_upper_bound_viol": float(np.mean(df[ub] == 0) * 100.0),
        f"pct_{prefix}_no_strike_monotonicity_viol": float(np.mean(df[mono_k] == 0) * 100.0),
        f"pct_{prefix}_no_strike_convexity_viol": float(np.mean(df[conv_k] == 0) * 100.0),
        f"pct_{prefix}_no_maturity_monotonicity_viol": float(np.mean(df[mono_t] == 0) * 100.0),
        f"mean_{prefix}_strike_monotonicity_viol": float(df[mono_k].mean()),
        f"mean_{prefix}_strike_convexity_viol": float(df[conv_k].mean()),
        f"mean_{prefix}_maturity_monotonicity_viol": float(df[mono_t].mean()),
    }
    summary[f"pct_{prefix}_full_price_surface_ok"] = float(
        np.mean(
            (df[invalid] == 0)
            & (df[lb] == 0)
            & (df[ub] == 0)
            & (df[mono_k] == 0)
            & (df[conv_k] == 0)
            & (df[mono_t] == 0)
        ) * 100.0
    )
    return summary


def save_heatmap(data: np.ndarray, title: str, out_path: Path, k_grid: np.ndarray, t_grid: np.ndarray) -> None:
    plt.figure(figsize=(9, 5))
    plt.imshow(data, aspect="auto", origin="lower")
    plt.xticks(range(len(k_grid)), [f"{k:.0f}" for k in k_grid], rotation=45)
    plt.yticks(range(len(t_grid)), [f"{t:.2f}" for t in t_grid])
    plt.xlabel("Strike")
    plt.ylabel("Maturity")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def format_params(row: pd.Series) -> str:
    return (
        f"v0={row['v0']:.3f}, rho={row['rho']:.3f}, "
        f"kappa={row['kappa']:.3f}, theta={row['theta']:.3f}, "
        f"sigma_v={row['sigma_v']:.3f}"
    )


def main() -> None:
    base = Path(__file__).resolve().parent
    data_path = base / "heston_dataset_10000_8x15_output" / "heston_dataset_10000_8x15_clean_train_only.csv"
    checkpoint_path = base / "training_output_10000_8x15_totalvar_priceconvex_v4_1" / "heston_totalvar_priceconvex_10000_8x15_v4.1.pt"
    out_dir = base / "training_output_10000_8x15_totalvar_priceconvex_v4_1" / "evaluation_10000_8x15_v4_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading dataset from: {data_path}")
    print(f"Loading v4.1 checkpoint from: {checkpoint_path}")
    print(f"Saving evaluation outputs to: {out_dir}")

    checkpoint = load_checkpoint(checkpoint_path)
    ckpt_feature_cols = list(checkpoint.get("feature_cols", FEATURE_COLS))
    ckpt_log_feature_cols = list(checkpoint.get("log_feature_cols", LOG_FEATURE_COLS))
    ckpt_target_cols = list(checkpoint["target_cols"])
    k_grid = np.asarray(checkpoint.get("k_grid", K_GRID), dtype=float)
    t_grid = np.asarray(checkpoint.get("t_grid", T_GRID), dtype=float)
    s0 = float(checkpoint.get("s0", S0))
    r = float(checkpoint.get("r", R))
    output_dim = int(checkpoint.get("output_dim", len(k_grid) * len(t_grid)))

    print(f"Checkpoint grid: len(T)={len(t_grid)}, len(K)={len(k_grid)}, K range={k_grid[0]:.2f} to {k_grid[-1]:.2f}")

    expected_targets = expected_iv_columns(k_grid, t_grid)
    if ckpt_target_cols != expected_targets:
        raise ValueError("Checkpoint target columns do not match expected grid columns.")
    if output_dim != len(k_grid) * len(t_grid):
        raise ValueError("Checkpoint output dimension does not match grid size.")

    df = pd.read_csv(data_path)
    missing_features = [c for c in ckpt_feature_cols if c not in df.columns]
    missing_targets = [c for c in ckpt_target_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    if missing_targets:
        raise ValueError(f"Missing target columns, first few: {missing_targets[:5]}")

    X_df = df[ckpt_feature_cols].copy()
    for col in ckpt_log_feature_cols:
        if (X_df[col] <= 0).any():
            raise ValueError(f"Column {col} contains non-positive values.")
        X_df[col] = np.log(X_df[col])
    X = X_df.to_numpy(dtype=np.float32)
    Y_iv = df[ckpt_target_cols].to_numpy(dtype=np.float32)
    W_true_all = total_variance_from_iv(Y_iv, k_grid, t_grid)
    Price_true_all = bs_call_price_np(Y_iv, s0, r, k_grid, t_grid)

    test_idx = np.asarray(checkpoint["test_idx"], dtype=int)
    X_test = X[test_idx]
    Y_test = Y_iv[test_idx]
    W_test = W_true_all[test_idx]
    Price_test = Price_true_all[test_idx]
    params_test = df.iloc[test_idx][ckpt_feature_cols].reset_index(drop=True)

    X_test_scaled = standardize_np(X_test, np.asarray(checkpoint["x_scaler_mean"], dtype=np.float32), np.asarray(checkpoint["x_scaler_scale"], dtype=np.float32))
    coord_features_scaled = np.asarray(checkpoint["coord_features_scaled"], dtype=np.float32)

    model = SurfaceCoordinateTotalVariancePriceConvexDNN(coord_features_scaled, k_grid, t_grid)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_iv, pred_w = model(torch.tensor(X_test_scaled, dtype=torch.float32))
    Y_pred = pred_iv.cpu().numpy().astype(np.float32)
    W_pred = pred_w.cpu().numpy().astype(np.float32)
    Price_pred = bs_call_price_np(Y_pred, s0, r, k_grid, t_grid)

    n_t, n_k = len(t_grid), len(k_grid)
    Y_true_surface = Y_test.reshape(len(test_idx), n_t, n_k)
    Y_pred_surface = Y_pred.reshape(len(test_idx), n_t, n_k)
    W_true_surface = W_test.reshape(len(test_idx), n_t, n_k)
    W_pred_surface = W_pred.reshape(len(test_idx), n_t, n_k)

    rows: list[dict[str, float | int]] = []
    for n in range(len(test_idx)):
        pred_surface = Y_pred_surface[n]
        true_surface = Y_true_surface[n]
        w_pred_surface = W_pred_surface[n]
        w_true_surface = W_true_surface[n]
        pred_prices, pred_invalid = iv_surface_to_price_surface(pred_surface, s0, r, k_grid, t_grid)
        true_prices, true_invalid = iv_surface_to_price_surface(true_surface, s0, r, k_grid, t_grid)
        pred_checks = surface_sanity_checks(pred_prices, s0, r, k_grid, t_grid, pred_invalid)
        true_checks = surface_sanity_checks(true_prices, s0, r, k_grid, t_grid, true_invalid)

        err_iv = pred_surface - true_surface
        abs_iv = np.abs(err_iv)
        err_w = w_pred_surface - w_true_surface
        abs_w = np.abs(err_w)
        price_err = (pred_prices - true_prices) / s0
        abs_price_err = np.abs(price_err)

        row: dict[str, float | int] = {
            "test_row": int(n),
            "original_dataset_index": int(test_idx[n]),
            "mae_iv": float(np.mean(abs_iv)),
            "rmse_iv": float(np.sqrt(np.mean(err_iv ** 2))),
            "max_abs_err_iv": float(np.max(abs_iv)),
            "mae_w": float(np.mean(abs_w)),
            "rmse_w": float(np.sqrt(np.mean(err_w ** 2))),
            "price_mae_normalized": float(np.mean(abs_price_err)),
            "price_rmse_normalized": float(np.sqrt(np.mean(price_err ** 2))),
        }
        for key, val in pred_checks.items():
            row[f"pred_{key}"] = val
        for key, val in true_checks.items():
            row[f"true_{key}"] = val
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    summary: dict[str, float | int] = {
        "test_surfaces": int(len(test_idx)),
        "mean_surface_mae_iv": float(metrics_df["mae_iv"].mean()),
        "mean_surface_rmse_iv": float(metrics_df["rmse_iv"].mean()),
        "max_surface_mae_iv": float(metrics_df["mae_iv"].max()),
        "mean_surface_mae_w": float(metrics_df["mae_w"].mean()),
        "mean_surface_rmse_w": float(metrics_df["rmse_w"].mean()),
        "mean_price_mae_normalized": float(metrics_df["price_mae_normalized"].mean()),
        "mean_price_rmse_normalized": float(metrics_df["price_rmse_normalized"].mean()),
    }
    summary.update(summarize_quality(metrics_df, "pred"))
    summary.update(summarize_quality(metrics_df, "true"))

    metrics_path = out_dir / "test_surface_metrics_10000_8x15_totalvar_priceconvex_v4.1.csv"
    selected_path = out_dir / "selected_samples_10000_8x15_totalvar_priceconvex_v4.1.csv"
    summary_path = out_dir / "surface_quality_summary_10000_8x15_totalvar_priceconvex_v4.1.txt"
    metrics_df.to_csv(metrics_path, index=False)

    order = metrics_df["mae_iv"].sort_values().index.to_numpy()
    selected = {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }

    selected_rows = []
    for label, idx in selected.items():
        selected_rows.append({
            "case": label,
            "test_row": idx,
            "original_dataset_index": int(test_idx[idx]),
            "mae_iv": float(metrics_df.loc[idx, "mae_iv"]),
            "rmse_iv": float(metrics_df.loc[idx, "rmse_iv"]),
            "params": format_params(params_test.iloc[idx]),
        })
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

    mean_abs_err = np.mean(np.abs(Y_pred_surface - Y_true_surface), axis=0)
    save_heatmap(mean_abs_err, "v4.1 Mean Absolute IV Error, 8x15 Test Set", out_dir / "mean_abs_error_heatmap_10000_8x15_totalvar_priceconvex_v4.1.png", k_grid, t_grid)

    for label, idx in selected.items():
        save_heatmap(Y_true_surface[idx], f"True IV Surface ({label}), v4.1 eval", out_dir / f"true_iv_{label}_10000_8x15_totalvar_priceconvex_v4.1.png", k_grid, t_grid)
        save_heatmap(Y_pred_surface[idx], f"Predicted IV Surface ({label}), v4.1", out_dir / f"pred_iv_{label}_10000_8x15_totalvar_priceconvex_v4.1.png", k_grid, t_grid)
        save_heatmap(np.abs(Y_pred_surface[idx] - Y_true_surface[idx]), f"Absolute IV Error ({label}), v4.1", out_dir / f"abs_error_{label}_10000_8x15_totalvar_priceconvex_v4.1.png", k_grid, t_grid)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("v4.1 total-variance price-convexity DNN evaluation summary\n")
        f.write("=" * 78 + "\n")
        f.write(f"dataset={data_path}\n")
        f.write(f"checkpoint={checkpoint_path}\n")
        f.write(f"grid={n_t}x{n_k}; K_grid={k_grid.tolist()}\n")
        f.write("\nAggregate error metrics\n")
        f.write("-" * 40 + "\n")
        for key in [
            "test_surfaces",
            "mean_surface_mae_iv",
            "mean_surface_rmse_iv",
            "max_surface_mae_iv",
            "mean_surface_mae_w",
            "mean_surface_rmse_w",
            "mean_price_mae_normalized",
            "mean_price_rmse_normalized",
        ]:
            val = summary[key]
            f.write(f"{key}: {val}\n" if isinstance(val, int) else f"{key}: {val:.10f}\n")
        f.write("\nPredicted surface-quality metrics\n")
        f.write("-" * 40 + "\n")
        for key in [k for k in summary if k.startswith("pct_pred") or k.startswith("mean_pred")]:
            f.write(f"{key}: {summary[key]:.6f}\n")
        f.write("\nTrue test-surface diagnostics\n")
        f.write("-" * 40 + "\n")
        for key in [k for k in summary if k.startswith("pct_true") or k.startswith("mean_true")]:
            f.write(f"{key}: {summary[key]:.6f}\n")

    print(f"Saved per-surface metrics: {metrics_path}")
    print(f"Saved selected-sample summary: {selected_path}")
    print(f"Saved surface-quality summary: {summary_path}")
    print("Saved plots in:")
    for p in sorted(out_dir.glob("*.png")):
        print(f" - {p}")

    print("\nAggregate error metrics, v4.1")
    for key in ["test_surfaces", "mean_surface_mae_iv", "mean_surface_rmse_iv", "max_surface_mae_iv", "mean_surface_mae_w", "mean_surface_rmse_w", "mean_price_mae_normalized"]:
        print(f"{key}: {summary[key]:.10f}" if isinstance(summary[key], float) else f"{key}: {summary[key]}")

    print("\nPredicted surface-quality metrics, v4.1")
    for key in [k for k in summary if k.startswith("pct_pred") or k.startswith("mean_pred")]:
        print(f"{key}: {summary[key]:.4f}")

    print("\nTrue test-surface diagnostics, v4.1")
    for key in [k for k in summary if k.startswith("pct_true") or k.startswith("mean_true")]:
        print(f"{key}: {summary[key]:.4f}")


if __name__ == "__main__":
    main()
