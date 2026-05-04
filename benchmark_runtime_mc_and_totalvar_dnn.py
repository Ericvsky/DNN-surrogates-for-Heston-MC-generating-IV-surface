from __future__ import annotations

"""
Runtime benchmark for the final 8x15 Heston project using the v4.1 total-variance
price-convexity DNN.

This script benchmarks two quantities:

1. Monte Carlo generation time for one 8x15 implied-volatility surface using the
   final production settings:
       M=5000, n_steps=400, full_truncation, antithetic=True.

2. Inference time for the final v4.1 surrogate:
       Theta -> positive total variance -> IV,
   using the checkpoint:
       training_output_10000_8x15_totalvar_priceconvex_v4_1/
       heston_totalvar_priceconvex_10000_8x15_v4.1.pt

Expected project files:
  - heston_mc_benchmark_toolkit.py
  - heston_dataset_10000_8x15_output/heston_dataset_10000_8x15_clean_train_only.csv
  - training_output_10000_8x15_totalvar_priceconvex_v4_1/heston_totalvar_priceconvex_10000_8x15_v4.1.pt

Outputs are saved to:
  runtime_benchmark_10000_8x15_v4_1/
"""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from heston_mc_benchmark_toolkit import HestonParams, benchmark_runtime

S0 = 100.0
R = 0.02
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)
assert len(T_GRID) == 8
assert len(K_GRID) == 15
assert abs(float(K_GRID[0]) - 90.0) < 1e-12
assert abs(float(K_GRID[-1]) - 118.0) < 1e-12
assert len(T_GRID) * len(K_GRID) == 120

M = 5_000
N_STEPS = 400
SCHEME = "full_truncation"
ANTITHETIC = True
CLIP_PRICES_FOR_IV = False
MC_REPEATS_PER_CASE = 5

FEATURE_COLS = ["v0", "rho", "kappa", "theta", "sigma_v"]
LOG_FEATURE_COLS = ["v0", "kappa", "theta", "sigma_v"]

# Set to 0 or 1 to force a specific GPU for DNN timing. Leave as None for default CUDA device.
CUDA_DEVICE_INDEX: int | None = None


def select_representative_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Select a small, reproducible set of parameter cases from the dataset."""
    features = df[FEATURE_COLS].copy()
    med = features.median()
    scale = features.std(ddof=0).replace(0.0, 1.0)
    zdist = (((features - med) / scale) ** 2).sum(axis=1)

    feller = 2.0 * df["kappa"] * df["theta"] / (df["sigma_v"] ** 2)
    case_indices = {
        "typical_median_like": int(zdist.idxmin()),
        "low_v0": int(df["v0"].idxmin()),
        "high_sigma_v": int(df["sigma_v"].idxmax()),
        "strong_negative_rho": int(df["rho"].idxmin()),
        "low_feller_ratio": int(feller.idxmin()),
    }

    rows = []
    used: set[int] = set()
    for name, idx in case_indices.items():
        if idx in used:
            continue
        used.add(idx)
        row = df.loc[idx, FEATURE_COLS].to_dict()
        row["case_name"] = name
        row["source_row_index"] = idx
        row["feller_ratio"] = float(feller.loc[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def run_mc_runtime(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cases = select_representative_cases(df)
    cases_path = out_dir / "runtime_representative_parameter_cases_10000_8x15_v4.1.csv"
    cases.to_csv(cases_path, index=False)
    print("Representative cases saved to:", cases_path)

    rows = []
    for _, row in cases.iterrows():
        name = str(row["case_name"])
        params = HestonParams(
            v0=float(row["v0"]),
            rho=float(row["rho"]),
            kappa=float(row["kappa"]),
            theta=float(row["theta"]),
            sigma_v=float(row["sigma_v"]),
        )
        print(f"Running MC timing case: {name}")
        timed = benchmark_runtime(
            params=params,
            S0=S0,
            r=R,
            K_grid=K_GRID,
            T_grid=T_GRID,
            M=M,
            n_steps=N_STEPS,
            repeats=MC_REPEATS_PER_CASE,
            scheme=SCHEME,
            antithetic=ANTITHETIC,
            clip_prices_for_iv=CLIP_PRICES_FOR_IV,
        )
        rec = timed.iloc[0].to_dict()
        rec.update({
            "case_name": name,
            "v0": params.v0,
            "rho": params.rho,
            "kappa": params.kappa,
            "theta": params.theta,
            "sigma_v": params.sigma_v,
            "surface_dim": int(len(K_GRID) * len(T_GRID)),
        })
        rows.append(rec)

    mc = pd.DataFrame(rows)
    mc.to_csv(out_dir / "mc_runtime_by_case_10000_8x15_v4.1_settings.csv", index=False)

    mean_mc = float(mc["mean_runtime_s"].mean())
    median_mc = float(mc["mean_runtime_s"].median())
    summary = pd.DataFrame([{
        "n_cases": int(len(mc)),
        "repeats_per_case": int(MC_REPEATS_PER_CASE),
        "M": int(M),
        "n_steps": int(N_STEPS),
        "surface_dim": int(len(K_GRID) * len(T_GRID)),
        "mean_mc_runtime_s_per_surface": mean_mc,
        "median_mc_runtime_s_per_surface": median_mc,
        "min_mc_runtime_s_per_surface": float(mc["mean_runtime_s"].min()),
        "max_mc_runtime_s_per_surface": float(mc["mean_runtime_s"].max()),
        "mean_mc_simulation_s": float(mc["mean_simulation_s"].mean()),
        "mean_mc_iv_inversion_s": float(mc["mean_iv_inversion_s"].mean()),
        "estimated_serial_time_for_10000_surfaces_s": mean_mc * 10000.0,
        "estimated_serial_time_for_10000_surfaces_hours": mean_mc * 10000.0 / 3600.0,
    }])
    summary.to_csv(out_dir / "mc_runtime_summary_10000_8x15_v4.1_settings.csv", index=False)

    print("\nMC runtime summary")
    print(summary.to_string(index=False))
    return summary


def select_torch_device(torch: Any):
    if torch.cuda.is_available():
        if CUDA_DEVICE_INDEX is not None:
            return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
        return torch.device("cuda")
    return torch.device("cpu")


def build_v41_model_class(torch: Any, nn: Any, F: Any):
    class SurfaceCoordinateTotalVariancePriceConvexDNN(nn.Module):
        def __init__(self, coord_features_scaled: np.ndarray, t_grid: np.ndarray, k_grid: np.ndarray,
                     input_dim: int = 5, latent_dim: int = 32, coord_dim: int = 5,
                     initial_total_variance: float = 0.04):
            super().__init__()
            output_dim = len(t_grid) * len(k_grid)
            if coord_features_scaled.shape != (output_dim, coord_dim):
                raise ValueError(
                    f"coord_features_scaled must have shape {(output_dim, coord_dim)}, got {coord_features_scaled.shape}"
                )
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Softplus(),
                nn.Linear(64, 64),
                nn.Softplus(),
                nn.Linear(64, latent_dim),
                nn.Softplus(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + coord_dim, 128),
                nn.Softplus(),
                nn.Linear(128, 128),
                nn.Softplus(),
                nn.Linear(128, 64),
                nn.Softplus(),
                nn.Linear(64, 1),
            )
            init_w = max(float(initial_total_variance), 1e-10)
            init_bias = float(np.log(np.expm1(init_w))) if init_w < 20.0 else init_w
            with torch.no_grad():
                self.decoder[-1].bias.fill_(init_bias)

            t_flat = np.repeat(t_grid, len(k_grid)).astype(np.float32)
            self.register_buffer("coord_features", torch.tensor(coord_features_scaled, dtype=torch.float32))
            self.register_buffer("t_flat", torch.tensor(t_flat, dtype=torch.float32))
            self.output_dim = output_dim
            self.latent_dim = latent_dim
            self.coord_dim = coord_dim

        def forward(self, x):
            batch = x.shape[0]
            z = self.encoder(x)
            z_exp = z[:, None, :].expand(batch, self.output_dim, self.latent_dim)
            c_exp = self.coord_features[None, :, :].expand(batch, self.output_dim, self.coord_dim)
            dec_in = torch.cat([z_exp, c_exp], dim=-1)
            raw_w = self.decoder(dec_in.reshape(batch * self.output_dim, self.latent_dim + self.coord_dim)).reshape(batch, self.output_dim)
            w_pred = F.softplus(raw_w) + 1e-10
            iv_pred = torch.sqrt(w_pred / self.t_flat[None, :])
            return iv_pred, w_pred

    return SurfaceCoordinateTotalVariancePriceConvexDNN


def run_v41_dnn_runtime(df: pd.DataFrame, out_dir: Path, mc_summary: pd.DataFrame) -> pd.DataFrame | None:
    base = Path(__file__).resolve().parent
    checkpoint_path = base / "training_output_10000_8x15_totalvar_priceconvex_v4_1" / "heston_totalvar_priceconvex_10000_8x15_v4.1.pt"
    if not checkpoint_path.exists():
        print("\nSkipping v4.1 DNN runtime benchmark because checkpoint was not found:")
        print(checkpoint_path)
        return None

    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except Exception as exc:
        print("\nSkipping v4.1 DNN runtime benchmark because torch could not be imported:")
        print(exc)
        return None

    print("\nv4.1 DNN inference runtime benchmark")
    print("-" * 60)
    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    device = select_torch_device(torch)
    print(f"DNN timing device: {device}")
    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")

    feature_cols = list(checkpoint["feature_cols"])
    log_feature_cols = list(checkpoint.get("log_feature_cols", LOG_FEATURE_COLS))
    k_grid = np.asarray(checkpoint.get("k_grid", K_GRID), dtype=float)
    t_grid = np.asarray(checkpoint.get("t_grid", T_GRID), dtype=float)
    coord_features_scaled = np.asarray(checkpoint["coord_features_scaled"], dtype=np.float32)
    input_dim = int(checkpoint.get("input_dim", 5))
    latent_dim = int(checkpoint.get("latent_dim", 32))
    coord_dim = int(checkpoint.get("coord_dim", 5))
    initial_total_variance = float(checkpoint.get("initial_total_variance", 0.04))

    ModelCls = build_v41_model_class(torch, nn, F)
    model = ModelCls(
        coord_features_scaled=coord_features_scaled,
        t_grid=t_grid,
        k_grid=k_grid,
        input_dim=input_dim,
        latent_dim=latent_dim,
        coord_dim=coord_dim,
        initial_total_variance=initial_total_variance,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    X_df = df[feature_cols].copy()
    for col in log_feature_cols:
        X_df[col] = np.log(X_df[col])
    X = X_df.to_numpy(dtype=np.float32)
    x_mean = np.asarray(checkpoint["x_scaler_mean"], dtype=np.float32)
    x_scale = np.asarray(checkpoint["x_scaler_scale"], dtype=np.float32)
    X_scaled = ((X - x_mean) / x_scale).astype(np.float32)

    idx = np.asarray(checkpoint.get("test_idx", np.arange(min(1500, len(df)))), dtype=int)
    X_eval = X_scaled[idx]

    batch_sizes = [1, 32, 256, min(1500, len(X_eval))]
    batch_sizes = sorted(set([int(b) for b in batch_sizes if b >= 1]))

    rows = []
    for batch_size in batch_sizes:
        x_batch_np = X_eval[:batch_size]
        x_batch = torch.tensor(x_batch_np, dtype=torch.float32, device=device)
        repeats = 2000 if batch_size == 1 else 500 if batch_size <= 32 else 200 if batch_size <= 256 else 100
        warmups = 50 if batch_size == 1 else 20

        with torch.no_grad():
            for _ in range(warmups):
                _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = perf_counter()
            for _ in range(repeats):
                _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - t0

        total_surfaces = batch_size * repeats
        mean_batch_s = elapsed / repeats
        mean_surface_s = elapsed / total_surfaces
        rows.append({
            "model": "v4.1_total_variance_price_convexity",
            "device": str(device),
            "batch_size": int(batch_size),
            "repeats": int(repeats),
            "mean_batch_runtime_s": float(mean_batch_s),
            "mean_runtime_s_per_surface": float(mean_surface_s),
            "surfaces_per_second": float(1.0 / mean_surface_s) if mean_surface_s > 0 else np.nan,
        })

    dnn = pd.DataFrame(rows)
    mc_mean = float(mc_summary["mean_mc_runtime_s_per_surface"].iloc[0])
    dnn["mc_mean_runtime_s_per_surface"] = mc_mean
    dnn["speedup_vs_mean_mc"] = mc_mean / dnn["mean_runtime_s_per_surface"]
    dnn.to_csv(out_dir / "dnn_inference_runtime_10000_8x15_totalvar_priceconvex_v4.1.csv", index=False)

    print("\nv4.1 DNN inference runtime summary")
    print(dnn.to_string(index=False))
    return dnn


def write_summary(out_dir: Path, mc_summary: pd.DataFrame, dnn: pd.DataFrame | None) -> None:
    path = out_dir / "runtime_benchmark_summary_10000_8x15_totalvar_priceconvex_v4.1.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Runtime benchmark summary: 8x15 Heston MC vs v4.1 total-variance DNN\n")
        f.write("=" * 78 + "\n")
        f.write(f"Grid: {len(T_GRID)} x {len(K_GRID)} = {len(T_GRID) * len(K_GRID)}\n")
        f.write(f"K_grid={K_GRID.tolist()}\n")
        f.write(f"T_grid={T_GRID.tolist()}\n")
        f.write(f"MC settings: M={M}, n_steps={N_STEPS}, scheme={SCHEME}, antithetic={ANTITHETIC}\n\n")

        m = mc_summary.iloc[0]
        f.write("Monte Carlo runtime\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean MC runtime per surface: {m['mean_mc_runtime_s_per_surface']:.10f} s\n")
        f.write(f"Median MC runtime per surface: {m['median_mc_runtime_s_per_surface']:.10f} s\n")
        f.write(f"Estimated serial time for 10000 surfaces: {m['estimated_serial_time_for_10000_surfaces_s']:.4f} s\n")
        f.write(f"Estimated serial time for 10000 surfaces: {m['estimated_serial_time_for_10000_surfaces_hours']:.6f} h\n\n")

        if dnn is not None and len(dnn):
            f.write("v4.1 DNN inference runtime\n")
            f.write("-" * 40 + "\n")
            for _, row in dnn.iterrows():
                f.write(
                    f"batch_size={int(row['batch_size'])}: "
                    f"{row['mean_runtime_s_per_surface']:.10e} s/surface, "
                    f"speedup={row['speedup_vs_mean_mc']:.4e}x, "
                    f"device={row['device']}\n"
                )
        else:
            f.write("v4.1 DNN timing skipped because checkpoint or torch was unavailable.\n")
    print("\nSaved runtime text summary:", path)


def main() -> None:
    base = Path(__file__).resolve().parent
    out_dir = base / "runtime_benchmark_10000_8x15_v4_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = base / "heston_dataset_10000_8x15_output" / "heston_dataset_10000_8x15_clean_train_only.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training-only dataset not found: {data_path}")

    print("Loading dataset from:", data_path)
    print("Saving runtime outputs to:", out_dir)
    print(f"Grid: {len(T_GRID)} x {len(K_GRID)} = {len(T_GRID) * len(K_GRID)}, K range {K_GRID[0]:.2f} to {K_GRID[-1]:.2f}")

    df = pd.read_csv(data_path)
    mc_summary = run_mc_runtime(df, out_dir)
    dnn = run_v41_dnn_runtime(df, out_dir, mc_summary)
    write_summary(out_dir, mc_summary, dnn)

    print("\nRuntime benchmark complete.")
    print("CSV outputs are in:", out_dir)


if __name__ == "__main__":
    main()
