from __future__ import annotations

"""
Train v4.1: Total-Variance DNN with Price-Convexity Loss on the 8 x 15 Heston dataset.

Version: v4.1, 8x15 K90--118 restart
=====================================

Purpose
-------
This script implements Pipeline A from the project redesign:

    Theta = (v0, rho, kappa, theta, sigma_v)
        -> positive total implied variance w(T,K)
        -> implied volatility sigma_IV(T,K)
        -> Black--Scholes call price C(T,K).

The final surrogate is still evaluated as a Theta -> IV model, but the DNN is trained to
predict positive total variance first. This directly addresses the negative-IV problem of
the original flat IV MLP.

Grid
----
    T = 0.25, 0.50, ..., 2.00        (8 maturities)
    K = 90, 92, 94, ..., 118         (15 strikes)
    output dimension = 8 * 15 = 120

Difference from the original flat IV MLP
----------------------------------------
Original v1.0/v1.1 baseline:
    - input:  5 Heston parameters;
    - output: 120 IV nodes directly;
    - final layer is linear, so negative IV predictions are possible;
    - loss is mainly IV MSE.

v4.1 total-variance price-convexity model:
    - input:  5 Heston parameters;
    - internal output: positive total variance w(T,K) = sigma_IV(T,K)^2 T;
    - final IV is recovered by sigma_IV = sqrt(w/T);
    - Softplus activations are used in the hidden networks and at the total-variance output;
    - loss includes total-variance, IV, price, smoothness, and price-convexity terms;
    - price-convexity is applied to the Black--Scholes price surface implied by the predicted IV.

Architecture
------------
Parameter encoder:
    5 -> 64 -> 64 -> 32

Shared grid decoder:
    32 latent features + 5 grid features -> 128 -> 128 -> 64 -> 1 raw total-variance node

Grid features:
    [log(K/S0), T, sqrt(T), log(K/S0)^2, T*log(K/S0)]

Training objective
------------------
The model predicts positive total variance:

    w_pred = softplus(raw_w) + eps
    iv_pred = sqrt(w_pred / T)
    price_pred = BlackScholesCall(iv_pred)

The full loss is:

    L = L_w
        + aux_weight(epoch) * (
              lambda_iv      * L_iv
            + lambda_price   * L_price
            + lambda_smooth  * L_smooth
            + lambda_conv_sq * L_conv_sq
            + lambda_conv_li * L_conv_linear
          )

where:
    L_w            = MSE of standardized total variance;
    L_iv           = MSE of standardized IV;
    L_price        = MSE of BS call prices normalized by S0;
    L_smooth       = second-difference smoothness penalty on w_pred;
    L_conv_sq      = squared hinge penalty on negative strike convexity of C_pred;
    L_conv_linear  = linear hinge penalty on negative strike convexity of C_pred.

The convexity penalty uses:

    C(K_{i+1}) - 2*C(K_i) + C(K_{i-1}) >= margin.

This is a soft penalty, not a hard projection, so it encourages but does not guarantee
strike convexity. If exact convexity is required later, add a price projection step.
"""

from pathlib import Path
import copy
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# Configuration
# -----------------------------
RANDOM_SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 50
AUX_WARMUP_EPOCHS = 80

# Adaptive target contribution ratios relative to the initial total-variance loss.
TARGET_IV_RATIO = 0.50
TARGET_PRICE_RATIO = 0.10
TARGET_SMOOTH_RATIO = 0.01
TARGET_CONV_SQ_RATIO = 0.05
TARGET_CONV_LINEAR_RATIO = 0.02

# Weight caps / fallback values to avoid unstable weights.
MAX_LAMBDA_IV = 10.0
MAX_LAMBDA_PRICE = 100.0
MAX_LAMBDA_SMOOTH = 10.0
MAX_LAMBDA_CONV_SQ = 500.0
MAX_LAMBDA_CONV_LINEAR = 5.0
FALLBACK_LAMBDA_CONV_SQ = 50.0
FALLBACK_LAMBDA_CONV_LINEAR = 0.05

CONVEXITY_MARGIN = 1e-4
EPS_TOTAL_VARIANCE = 1e-8
EPS_CALIBRATION = 1e-12

FEATURE_COLS = ["v0", "rho", "kappa", "theta", "sigma_v"]
LOG_FEATURE_COLS = ["v0", "kappa", "theta", "sigma_v"]

S0 = 100.0
R = 0.02
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)

INPUT_DIM = 5
LATENT_DIM = 32
COORD_DIM = 5
OUTPUT_DIM = len(T_GRID) * len(K_GRID)

CUDA_DEVICE_INDEX: int | None = None


# -----------------------------
# Utility functions
# -----------------------------
def expected_iv_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"iv_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def total_variance_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"w_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def price_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"price_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def build_coord_features(k_grid: np.ndarray, t_grid: np.ndarray, s0: float) -> np.ndarray:
    rows = []
    for T in t_grid:
        for K in k_grid:
            m = float(np.log(K / s0))
            rows.append([m, float(T), float(np.sqrt(T)), m * m, float(T) * m])
    return np.asarray(rows, dtype=np.float32)


def standardize_np(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return ((x - mean) / safe_scale).astype(np.float32)


def total_variance_from_iv_np(iv: np.ndarray) -> np.ndarray:
    t_flat = np.repeat(T_GRID, len(K_GRID)).astype(np.float32)
    return (iv.astype(np.float32) ** 2) * t_flat[None, :]


def select_device() -> torch.device:
    if torch.cuda.is_available():
        if CUDA_DEVICE_INDEX is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normal_cdf_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


def bs_call_price_torch(iv: torch.Tensor, s0: float, r: float, k_grid: np.ndarray, t_grid: np.ndarray) -> torch.Tensor:
    """Differentiable Black--Scholes call prices for IV vectors of shape (B, 120)."""
    device = iv.device
    dtype = iv.dtype
    k_flat = torch.tensor(np.tile(k_grid, len(t_grid)), dtype=dtype, device=device)[None, :]
    t_flat = torch.tensor(np.repeat(t_grid, len(k_grid)), dtype=dtype, device=device)[None, :]
    sigma = torch.clamp(iv, min=1e-8)
    sqrt_t = torch.sqrt(t_flat)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (torch.log(torch.tensor(s0, dtype=dtype, device=device) / k_flat) + (r + 0.5 * sigma.pow(2)) * t_flat) / torch.clamp(vol_sqrt_t, min=1e-8)
    d2 = d1 - vol_sqrt_t
    price = s0 * normal_cdf_torch(d1) - k_flat * torch.exp(-r * t_flat) * normal_cdf_torch(d2)
    return price


def smoothness_loss_w(w_pred: torch.Tensor) -> torch.Tensor:
    w = w_pred.reshape(-1, len(T_GRID), len(K_GRID))
    second_k = w[:, :, 2:] - 2.0 * w[:, :, 1:-1] + w[:, :, :-2]
    second_t = w[:, 2:, :] - 2.0 * w[:, 1:-1, :] + w[:, :-2, :]
    return second_k.pow(2).mean() + second_t.pow(2).mean()


def price_convexity_losses(price_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = price_pred.reshape(-1, len(T_GRID), len(K_GRID))
    second_k = c[:, :, 2:] - 2.0 * c[:, :, 1:-1] + c[:, :, :-2]
    violation = F.relu(CONVEXITY_MARGIN - second_k)
    return violation.pow(2).mean(), violation.mean()


# -----------------------------
# Model
# -----------------------------
class SurfaceCoordinateTotalVariancePriceConvexDNN(nn.Module):
    def __init__(self, coord_features_scaled: np.ndarray, initial_total_variance: float = 0.04):
        super().__init__()
        if coord_features_scaled.shape != (OUTPUT_DIM, COORD_DIM):
            raise ValueError(f"coord_features_scaled must have shape {(OUTPUT_DIM, COORD_DIM)}, got {coord_features_scaled.shape}")

        # Softplus hidden activations make the surface generator smoother than a ReLU MLP.
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

        # Bias the initial raw output toward a typical total variance level.
        # This avoids starting with unrealistically large IVs from Softplus(raw_w) ~= 0.69.
        init_w = max(float(initial_total_variance), EPS_TOTAL_VARIANCE)
        if init_w < 20.0:
            init_bias = float(np.log(np.expm1(init_w)))
        else:
            init_bias = init_w
        with torch.no_grad():
            self.decoder[-1].bias.fill_(init_bias)

        coord = torch.tensor(coord_features_scaled, dtype=torch.float32)
        t_flat = np.repeat(T_GRID, len(K_GRID)).astype(np.float32)
        self.register_buffer("coord_features", coord)
        self.register_buffer("t_flat", torch.tensor(t_flat, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted IV vector, predicted total-variance vector)."""
        batch = x.shape[0]
        z = self.encoder(x)
        z_exp = z[:, None, :].expand(batch, OUTPUT_DIM, LATENT_DIM)
        c_exp = self.coord_features[None, :, :].expand(batch, OUTPUT_DIM, COORD_DIM)
        dec_in = torch.cat([z_exp, c_exp], dim=-1)
        raw_w = self.decoder(dec_in.reshape(batch * OUTPUT_DIM, LATENT_DIM + COORD_DIM)).reshape(batch, OUTPUT_DIM)
        w_pred = F.softplus(raw_w) + EPS_TOTAL_VARIANCE
        iv_pred = torch.sqrt(w_pred / self.t_flat[None, :])
        return iv_pred, w_pred


# -----------------------------
# Loss logic
# -----------------------------
def compute_components(
    model: SurfaceCoordinateTotalVariancePriceConvexDNN,
    xb: torch.Tensor,
    y_iv_scaled: torch.Tensor,
    y_iv_true: torch.Tensor,
    w_true_scaled: torch.Tensor,
    price_true: torch.Tensor,
    y_mean_t: torch.Tensor,
    y_scale_t: torch.Tensor,
    w_mean_t: torch.Tensor,
    w_scale_t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    iv_pred, w_pred = model(xb)
    iv_pred_scaled = (iv_pred - y_mean_t[None, :]) / y_scale_t[None, :]
    w_pred_scaled = (w_pred - w_mean_t[None, :]) / w_scale_t[None, :]
    price_pred = bs_call_price_torch(iv_pred, S0, R, K_GRID, T_GRID)

    w_scaled_mse = F.mse_loss(w_pred_scaled, w_true_scaled)
    iv_scaled_mse = F.mse_loss(iv_pred_scaled, y_iv_scaled)
    price_mse = F.mse_loss(price_pred / S0, price_true / S0)
    smooth_w = smoothness_loss_w(w_pred)
    conv_sq, conv_linear = price_convexity_losses(price_pred)
    iv_original_mse = F.mse_loss(iv_pred, y_iv_true)
    w_original_mse = F.mse_loss(w_pred, w_true_scaled * w_scale_t[None, :] + w_mean_t[None, :])

    return {
        "w_scaled_mse": w_scaled_mse,
        "iv_scaled_mse": iv_scaled_mse,
        "price_mse": price_mse,
        "smoothness_w": smooth_w,
        "price_convexity_sq": conv_sq,
        "price_convexity_linear": conv_linear,
        "iv_original_mse": iv_original_mse,
        "w_original_mse": w_original_mse,
    }


def calibrate_auxiliary_weights(components: dict[str, torch.Tensor]) -> dict[str, float]:
    ref = float(components["w_scaled_mse"].detach().cpu().item())
    if ref <= 0.0 or not np.isfinite(ref):
        ref = 1.0

    def weight_for(name: str, target_ratio: float, max_weight: float, fallback: float | None = None) -> float:
        val = float(components[name].detach().cpu().item())
        if not np.isfinite(val) or val < EPS_CALIBRATION:
            return float(fallback if fallback is not None else 0.0)
        return float(min(max_weight, target_ratio * ref / val))

    return {
        "lambda_iv": weight_for("iv_scaled_mse", TARGET_IV_RATIO, MAX_LAMBDA_IV),
        "lambda_price": weight_for("price_mse", TARGET_PRICE_RATIO, MAX_LAMBDA_PRICE),
        "lambda_smooth": weight_for("smoothness_w", TARGET_SMOOTH_RATIO, MAX_LAMBDA_SMOOTH),
        "lambda_conv_sq": weight_for("price_convexity_sq", TARGET_CONV_SQ_RATIO, MAX_LAMBDA_CONV_SQ, FALLBACK_LAMBDA_CONV_SQ),
        "lambda_conv_linear": weight_for("price_convexity_linear", TARGET_CONV_LINEAR_RATIO, MAX_LAMBDA_CONV_LINEAR, FALLBACK_LAMBDA_CONV_LINEAR),
    }


def combine_loss(components: dict[str, torch.Tensor], weights: dict[str, float], aux_weight: float) -> torch.Tensor:
    auxiliary = (
        weights["lambda_iv"] * components["iv_scaled_mse"]
        + weights["lambda_price"] * components["price_mse"]
        + weights["lambda_smooth"] * components["smoothness_w"]
        + weights["lambda_conv_sq"] * components["price_convexity_sq"]
        + weights["lambda_conv_linear"] * components["price_convexity_linear"]
    )
    return components["w_scaled_mse"] + aux_weight * auxiliary


def aux_schedule(epoch: int) -> float:
    return float(min(1.0, epoch / max(1, AUX_WARMUP_EPOCHS)))


def tensor_dataset_from_arrays(
    X: np.ndarray,
    Y_iv_scaled: np.ndarray,
    Y_iv_true: np.ndarray,
    W_scaled: np.ndarray,
    Price_true: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y_iv_scaled, dtype=torch.float32),
        torch.tensor(Y_iv_true, dtype=torch.float32),
        torch.tensor(W_scaled, dtype=torch.float32),
        torch.tensor(Price_true, dtype=torch.float32),
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    seed_everything(RANDOM_SEED)

    base = Path(__file__).resolve().parent
    data_path = base / "heston_dataset_10000_8x15_output" / "heston_dataset_10000_8x15_clean_train_only.csv"
    out_dir = base / "training_output_10000_8x15_totalvar_priceconvex_v4_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    print(f"Loading dataset from: {data_path}")
    print(f"Saving v4.1 outputs to: {out_dir}")
    print("Model: total-variance DNN + price-convexity loss")
    print(f"Grid: {len(T_GRID)} x {len(K_GRID)} = {OUTPUT_DIM}; K range {K_GRID[0]:.2f} to {K_GRID[-1]:.2f}")

    df = pd.read_csv(data_path)
    target_cols = [c for c in df.columns if c.startswith("iv_T")]
    expected_targets = expected_iv_columns(K_GRID, T_GRID)
    if target_cols != expected_targets:
        raise ValueError(
            "IV target columns do not match the expected 8x15 K=90,...,118 grid order.\n"
            f"Expected first/last: {expected_targets[:3]} ... {expected_targets[-3:]}\n"
            f"Found first/last:    {target_cols[:3]} ... {target_cols[-3:]}"
        )
    if len(target_cols) != OUTPUT_DIM:
        raise ValueError(f"Expected {OUTPUT_DIM} IV targets, found {len(target_cols)}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if int(df[FEATURE_COLS + target_cols].isna().sum().sum()) != 0:
        raise ValueError("NaNs found in feature/target data.")

    X_df = df[FEATURE_COLS].copy()
    for col in LOG_FEATURE_COLS:
        if (X_df[col] <= 0).any():
            raise ValueError(f"Column {col} contains non-positive values, cannot log-transform.")
        X_df[col] = np.log(X_df[col])

    X = X_df.to_numpy(dtype=np.float32)
    Y_iv = df[target_cols].to_numpy(dtype=np.float32)
    W = total_variance_from_iv_np(Y_iv)

    # True prices are auxiliary labels derived from IV labels.
    # Use torch in chunks to avoid duplicating the NumPy BS implementation here.
    with torch.no_grad():
        price_true_all = []
        chunk = 2048
        for start in range(0, len(Y_iv), chunk):
            iv_t = torch.tensor(Y_iv[start:start + chunk], dtype=torch.float32)
            price_true_all.append(bs_call_price_torch(iv_t, S0, R, K_GRID, T_GRID).cpu().numpy())
        Price = np.vstack(price_true_all).astype(np.float32)

    all_idx = np.arange(len(df), dtype=int)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, random_state=RANDOM_SEED, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=RANDOM_SEED, shuffle=True)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y_iv[train_idx], Y_iv[val_idx], Y_iv[test_idx]
    W_train, W_val, W_test = W[train_idx], W[val_idx], W[test_idx]
    P_train, P_val, P_test = Price[train_idx], Price[val_idx], Price[test_idx]

    print(f"Train split: X={X_train.shape}, IV={Y_train.shape}, W={W_train.shape}")
    print(f"Val split:   X={X_val.shape}, IV={Y_val.shape}, W={W_val.shape}")
    print(f"Test split:  X={X_test.shape}, IV={Y_test.shape}, W={W_test.shape}")

    x_scaler = StandardScaler().fit(X_train)
    iv_scaler = StandardScaler().fit(Y_train)
    w_scaler = StandardScaler().fit(W_train)
    coord_features = build_coord_features(K_GRID, T_GRID, S0)
    coord_scaler = StandardScaler().fit(coord_features)
    coord_features_scaled = coord_scaler.transform(coord_features).astype(np.float32)

    X_train_s = standardize_np(X_train, x_scaler.mean_, x_scaler.scale_)
    X_val_s = standardize_np(X_val, x_scaler.mean_, x_scaler.scale_)
    X_test_s = standardize_np(X_test, x_scaler.mean_, x_scaler.scale_)
    Y_train_s = standardize_np(Y_train, iv_scaler.mean_, iv_scaler.scale_)
    Y_val_s = standardize_np(Y_val, iv_scaler.mean_, iv_scaler.scale_)
    Y_test_s = standardize_np(Y_test, iv_scaler.mean_, iv_scaler.scale_)
    W_train_s = standardize_np(W_train, w_scaler.mean_, w_scaler.scale_)
    W_val_s = standardize_np(W_val, w_scaler.mean_, w_scaler.scale_)
    W_test_s = standardize_np(W_test, w_scaler.mean_, w_scaler.scale_)

    train_ds = tensor_dataset_from_arrays(X_train_s, Y_train_s, Y_train, W_train_s, P_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    device = select_device()
    print(f"Training on device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    initial_total_variance = float(np.mean(W_train))
    model = SurfaceCoordinateTotalVariancePriceConvexDNN(coord_features_scaled, initial_total_variance=initial_total_variance).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    y_mean_t = torch.tensor(iv_scaler.mean_, dtype=torch.float32, device=device)
    y_scale_t = torch.tensor(np.where(iv_scaler.scale_ == 0.0, 1.0, iv_scaler.scale_), dtype=torch.float32, device=device)
    w_mean_t = torch.tensor(w_scaler.mean_, dtype=torch.float32, device=device)
    w_scale_t = torch.tensor(np.where(w_scaler.scale_ == 0.0, 1.0, w_scaler.scale_), dtype=torch.float32, device=device)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    Y_val_s_t = torch.tensor(Y_val_s, dtype=torch.float32, device=device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)
    W_val_s_t = torch.tensor(W_val_s, dtype=torch.float32, device=device)
    P_val_t = torch.tensor(P_val, dtype=torch.float32, device=device)

    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)
    Y_test_s_t = torch.tensor(Y_test_s, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)
    W_test_s_t = torch.tensor(W_test_s, dtype=torch.float32, device=device)
    P_test_t = torch.tensor(P_test, dtype=torch.float32, device=device)

    # Adaptive calibration on the first training batch.
    first_batch = next(iter(train_loader))
    first_batch = [z.to(device) for z in first_batch]
    with torch.no_grad():
        initial_components = compute_components(
            model,
            first_batch[0],
            first_batch[1],
            first_batch[2],
            first_batch[3],
            first_batch[4],
            y_mean_t,
            y_scale_t,
            w_mean_t,
            w_scale_t,
        )
    loss_weights = calibrate_auxiliary_weights(initial_components)

    print("\nInitial component values used for adaptive calibration")
    for key, val in initial_components.items():
        print(f"  {key}: {float(val.detach().cpu().item()):.12g}")
    print("\nAdaptive/fallback auxiliary weights")
    for key, val in loss_weights.items():
        print(f"  {key}: {val:.12g}")
    print(f"Convexity margin: {CONVEXITY_MARGIN}")
    print(f"Auxiliary warm-up epochs: {AUX_WARMUP_EPOCHS}")

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    wait = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        aux_w = aux_schedule(epoch)
        train_totals: list[float] = []
        train_w_losses: list[float] = []
        train_iv_losses: list[float] = []
        train_conv_losses: list[float] = []

        for xb, yiv_s, yiv, w_s, p_true in train_loader:
            xb = xb.to(device)
            yiv_s = yiv_s.to(device)
            yiv = yiv.to(device)
            w_s = w_s.to(device)
            p_true = p_true.to(device)

            optimizer.zero_grad()
            components = compute_components(model, xb, yiv_s, yiv, w_s, p_true, y_mean_t, y_scale_t, w_mean_t, w_scale_t)
            loss = combine_loss(components, loss_weights, aux_w)
            loss.backward()
            optimizer.step()

            train_totals.append(float(loss.detach().cpu().item()))
            train_w_losses.append(float(components["w_scaled_mse"].detach().cpu().item()))
            train_iv_losses.append(float(components["iv_scaled_mse"].detach().cpu().item()))
            train_conv_losses.append(float(components["price_convexity_linear"].detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_components = compute_components(
                model,
                X_val_t,
                Y_val_s_t,
                Y_val_t,
                W_val_s_t,
                P_val_t,
                y_mean_t,
                y_scale_t,
                w_mean_t,
                w_scale_t,
            )
            val_total = combine_loss(val_components, loss_weights, aux_w)

        val_loss_value = float(val_total.detach().cpu().item())
        row = {
            "epoch": epoch,
            "aux_weight": aux_w,
            "train_total": float(np.mean(train_totals)),
            "train_w_scaled_mse": float(np.mean(train_w_losses)),
            "train_iv_scaled_mse": float(np.mean(train_iv_losses)),
            "train_price_convexity_linear": float(np.mean(train_conv_losses)),
            "val_total": val_loss_value,
            **{f"val_{k}": float(v.detach().cpu().item()) for k, v in val_components.items()},
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | aux_w={aux_w:.2f} | "
                f"train_total={row['train_total']:.6f} | val_total={val_loss_value:.6f} | "
                f"val_w={row['val_w_scaled_mse']:.6f} | val_iv={row['val_iv_scaled_mse']:.6f} | "
                f"val_price={row['val_price_mse']:.8f} | val_conv_lin={row['val_price_convexity_linear']:.8f}"
            )

        if val_loss_value < best_val_loss - 1e-8:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_components = compute_components(
            model,
            X_test_t,
            Y_test_s_t,
            Y_test_t,
            W_test_s_t,
            P_test_t,
            y_mean_t,
            y_scale_t,
            w_mean_t,
            w_scale_t,
        )
        test_total = combine_loss(test_components, loss_weights, 1.0)
        test_iv_pred, test_w_pred = model(X_test_t)
        test_price_pred = bs_call_price_torch(test_iv_pred, S0, R, K_GRID, T_GRID)

    test_iv_np = test_iv_pred.detach().cpu().numpy()
    test_w_np = test_w_pred.detach().cpu().numpy()
    test_price_np = test_price_pred.detach().cpu().numpy()

    rmse_iv = float(np.sqrt(np.mean((test_iv_np - Y_test) ** 2)))
    mae_iv = float(np.mean(np.abs(test_iv_np - Y_test)))
    rmse_w = float(np.sqrt(np.mean((test_w_np - W_test) ** 2)))
    mae_w = float(np.mean(np.abs(test_w_np - W_test)))
    price_rmse = float(np.sqrt(np.mean(((test_price_np - P_test) / S0) ** 2)))
    price_mae = float(np.mean(np.abs((test_price_np - P_test) / S0)))

    print("\nFinal test metrics on 10,000-row 8x15 dataset, v4.1")
    print(f"Best epoch:           {best_epoch}")
    print(f"Best val total loss:  {best_val_loss:.8f}")
    print(f"Test RMSE (IV units): {rmse_iv:.6f}")
    print(f"Test MAE  (IV units): {mae_iv:.6f}")
    print(f"Test RMSE (w units):  {rmse_w:.8f}")
    print(f"Test MAE  (w units):  {mae_w:.8f}")
    print(f"Test RMSE (price/S0): {price_rmse:.8f}")
    print(f"Test MAE  (price/S0): {price_mae:.8f}")
    print(f"Test total loss:      {float(test_total.detach().cpu().item()):.8f}")
    for key, value in test_components.items():
        print(f"Test {key}: {float(value.detach().cpu().item()):.10f}")

    history_path = out_dir / "loss_history_10000_8x15_totalvar_priceconvex_v4.1.csv"
    metrics_path = out_dir / "metrics_10000_8x15_totalvar_priceconvex_v4.1.txt"
    notes_path = out_dir / "model_design_notes_v4.1.txt"
    split_path = out_dir / "train_val_test_split_indices_10000_8x15_v4.1.npz"
    checkpoint_path = out_dir / "heston_totalvar_priceconvex_10000_8x15_v4.1.pt"

    pd.DataFrame(history).to_csv(history_path, index=False)
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    checkpoint = {
        "version": "v4.1_totalvar_priceconvex_8x15",
        "model_state_dict": model.state_dict(),
        "feature_cols": FEATURE_COLS,
        "log_feature_cols": LOG_FEATURE_COLS,
        "target_cols": target_cols,
        "total_variance_cols": total_variance_columns(K_GRID, T_GRID),
        "price_cols": price_columns(K_GRID, T_GRID),
        "x_scaler_mean": x_scaler.mean_,
        "x_scaler_scale": x_scaler.scale_,
        "iv_scaler_mean": iv_scaler.mean_,
        "iv_scaler_scale": iv_scaler.scale_,
        "w_scaler_mean": w_scaler.mean_,
        "w_scaler_scale": w_scaler.scale_,
        "coord_scaler_mean": coord_scaler.mean_,
        "coord_scaler_scale": coord_scaler.scale_,
        "coord_features_scaled": coord_features_scaled,
        "k_grid": K_GRID,
        "t_grid": T_GRID,
        "s0": S0,
        "r": R,
        "input_dim": INPUT_DIM,
        "latent_dim": LATENT_DIM,
        "coord_dim": COORD_DIM,
        "output_dim": OUTPUT_DIM,
        "initial_total_variance": initial_total_variance,
        "random_seed": RANDOM_SEED,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "loss_weights": loss_weights,
        "loss_config": {
            "target_iv_ratio": TARGET_IV_RATIO,
            "target_price_ratio": TARGET_PRICE_RATIO,
            "target_smooth_ratio": TARGET_SMOOTH_RATIO,
            "target_conv_sq_ratio": TARGET_CONV_SQ_RATIO,
            "target_conv_linear_ratio": TARGET_CONV_LINEAR_RATIO,
            "convexity_margin": CONVEXITY_MARGIN,
            "aux_warmup_epochs": AUX_WARMUP_EPOCHS,
        },
        "test_metrics": {
            "rmse_iv": rmse_iv,
            "mae_iv": mae_iv,
            "rmse_w": rmse_w,
            "mae_w": mae_w,
            "price_rmse_normalized": price_rmse,
            "price_mae_normalized": price_mae,
            "test_total_loss": float(test_total.detach().cpu().item()),
            **{f"test_{k}": float(v.detach().cpu().item()) for k, v in test_components.items()},
        },
    }
    torch.save(checkpoint, checkpoint_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("v4.1 total-variance price-convexity DNN metrics\n")
        f.write("=" * 70 + "\n")
        f.write(f"dataset={data_path}\n")
        f.write(f"grid={len(T_GRID)}x{len(K_GRID)}; surface_dim={OUTPUT_DIM}\n")
        f.write(f"K_grid={K_GRID.tolist()}\n")
        f.write(f"architecture=surface-coordinate total-variance DNN with Softplus hidden/output activations\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.10f}\n")
        f.write(f"test_rmse_iv={rmse_iv:.10f}\n")
        f.write(f"test_mae_iv={mae_iv:.10f}\n")
        f.write(f"test_rmse_w={rmse_w:.10f}\n")
        f.write(f"test_mae_w={mae_w:.10f}\n")
        f.write(f"test_price_rmse_normalized={price_rmse:.10f}\n")
        f.write(f"test_price_mae_normalized={price_mae:.10f}\n")
        f.write(f"test_total_loss={float(test_total.detach().cpu().item()):.10f}\n")
        f.write("loss_weights=\n")
        f.write(json.dumps(loss_weights, indent=2) + "\n")
        f.write("test_loss_components=\n")
        f.write(json.dumps({k: float(v.detach().cpu().item()) for k, v in test_components.items()}, indent=2) + "\n")

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("v4.1 model-design notes\n")
        f.write("=" * 70 + "\n")
        f.write("This model replaces direct IV output with positive total implied variance output.\n")
        f.write("The final IV surface is recovered as sigma_IV=sqrt(w/T), which removes negative IV predictions by construction.\n")
        f.write("Strike convexity is addressed by a soft price-space convexity penalty applied after converting predicted IV to BS call prices.\n")
        f.write("The convexity penalty is not a hard guarantee; exact convexity would require a post-processing projection or constrained decoder.\n")
        f.write("Hidden activations use Softplus instead of ReLU to produce smoother coordinate-decoded surfaces.\n")
        f.write("The model still takes the 5 Heston parameters and ultimately returns a 120-node IV surface.\n")

    print(f"\nSaved loss history: {history_path}")
    print(f"Saved split indices: {split_path}")
    print(f"Saved checkpoint:   {checkpoint_path}")
    print(f"Saved metrics:      {metrics_path}")
    print(f"Saved design notes: {notes_path}")


if __name__ == "__main__":
    main()
