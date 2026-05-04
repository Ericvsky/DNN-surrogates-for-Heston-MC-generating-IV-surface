from __future__ import annotations

"""
Build a clean 10,000-row Heston implied-volatility dataset on an 8 x 15 grid.

Version: 8x15 v1.0
-------------------
This script restarts the project with a smaller output grid:

    maturities: 8 values, T = 0.25, 0.50, ..., 2.00
    strikes:    15 values, K = linspace(90, 110, 15)
    output dim: 8 x 15 = 120 IV targets

The Monte Carlo settings are kept consistent with the previous production run:

    M = 5000 paths
    n_steps = 400
    scheme = full_truncation
    antithetic = True

Outputs
-------
heston_dataset_10000_8x15_output/
    heston_dataset_10000_8x15_clean_batch_01.csv
    ...
    heston_dataset_10000_8x15_clean_batch_10.csv
    heston_dataset_10000_8x15_clean_full.csv      # full data + diagnostics
    heston_dataset_10000_8x15_clean_train_only.csv # 5 inputs + 120 IV targets
    heston_dataset_10000_8x15_clean_metadata.txt

Expected shapes
---------------
Full clean dataset:      (10000, 136) = 5 parameters + 10 diagnostics + 120 IV + accepted_attempt
Training-only dataset:   (10000, 125) = 5 parameters + 120 IV
"""

from pathlib import Path

import numpy as np
import pandas as pd

from heston_mc_benchmark_toolkit import generate_clean_mc_dataset


def main() -> None:
    # Fixed market inputs
    S0 = 100.0
    r = 0.02

    # Restart grid: 8 maturities x 15 strikes = 120 IV nodes.
    # The strike range remains 90--118 so the moneyness range is comparable to the previous 8x21 setup.
    T_grid = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
    K_grid = np.linspace(90.0, 118.0, 15, dtype=float)
    if len(K_grid) != 15 or abs(K_grid[0] - 90.0) > 1e-12 or abs(K_grid[-1] - 118.0) > 1e-12:
        raise ValueError(f"Unexpected strike grid: {K_grid}")
    surface_dim = len(T_grid) * len(K_grid)
    if surface_dim != 120:
        raise ValueError(f"Expected 120 IV nodes for the 8x15 grid, got {surface_dim}.")

    # MC settings
    M = 5_000
    n_steps = 400
    scheme = "full_truncation"
    antithetic = True
    clip_prices_for_iv = False

    # Quality filters
    require_zero_iv_failures = True
    require_no_bound_violations = True
    require_strike_monotonicity = True
    require_strike_convexity = False
    allow_repaired_prices_for_iv = False

    # Parameter sampling settings
    enforce_feller = False
    log_sample_positive = True

    # Dataset size / batching
    total_samples = 10_000
    batch_size = 1_000
    if total_samples % batch_size != 0:
        raise ValueError("total_samples must be divisible by batch_size.")
    n_batches = total_samples // batch_size

    # Output location
    output_dir = Path("heston_dataset_10000_8x15_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # DNN view of the dataset
    feature_cols = ["v0", "rho", "kappa", "theta", "sigma_v"]

    print("Starting clean 8x15 dataset generation")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Grid: {len(T_grid)} maturities x {len(K_grid)} strikes = {surface_dim} IV targets")
    print(f"K_grid={K_grid.tolist()}")
    print(f"T_grid={T_grid.tolist()}")
    print(f"M={M}, n_steps={n_steps}, scheme={scheme}, antithetic={antithetic}")
    print(
        "Quality filters: "
        f"iv_failures={require_zero_iv_failures}, "
        f"bounds={require_no_bound_violations}, "
        f"strike_monotonicity={require_strike_monotonicity}, "
        f"strike_convexity={require_strike_convexity}, "
        f"allow_repaired_prices_for_iv={allow_repaired_prices_for_iv}"
    )
    print(
        "Sampling settings: "
        f"enforce_feller={enforce_feller}, "
        f"log_sample_positive={log_sample_positive}"
    )

    batch_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for batch_idx in range(n_batches):
        print("\n" + "=" * 80)
        print(f"Generating batch {batch_idx + 1}/{n_batches}")

        # Shift seeds between batches so parameter draws and MC randomness differ.
        sampler_seed = 2026 + 10_000 * batch_idx
        base_seed = 5000 + 100_000 * batch_idx

        df_batch = generate_clean_mc_dataset(
            n_samples=batch_size,
            S0=S0,
            r=r,
            K_grid=K_grid,
            T_grid=T_grid,
            M=M,
            n_steps=n_steps,
            sampler_seed=sampler_seed,
            base_seed=base_seed,
            scheme=scheme,
            antithetic=antithetic,
            clip_prices_for_iv=clip_prices_for_iv,
            require_zero_iv_failures=require_zero_iv_failures,
            require_no_bound_violations=require_no_bound_violations,
            require_strike_monotonicity=require_strike_monotonicity,
            require_strike_convexity=require_strike_convexity,
            allow_repaired_prices_for_iv=allow_repaired_prices_for_iv,
            max_attempts_multiplier=30,
            param_sampling_kwargs={
                "enforce_feller": enforce_feller,
                "log_sample_positive": log_sample_positive,
            },
        )

        batch_path = output_dir / f"heston_dataset_10000_8x15_clean_batch_{batch_idx + 1:02d}.csv"
        df_batch.to_csv(batch_path, index=False)
        batch_paths.append(batch_path)
        frames.append(df_batch)
        print(f"Saved {len(df_batch)} rows to {batch_path}")

    df_full = pd.concat(frames, axis=0, ignore_index=True)
    full_path = output_dir / "heston_dataset_10000_8x15_clean_full.csv"
    df_full.to_csv(full_path, index=False)

    iv_cols = [c for c in df_full.columns if c.startswith("iv_T")]
    if len(iv_cols) != surface_dim:
        raise ValueError(
            f"Expected {surface_dim} IV target columns in the full dataset, found {len(iv_cols)}."
        )

    df_train = df_full[feature_cols + iv_cols].copy()
    train_path = output_dir / "heston_dataset_10000_8x15_clean_train_only.csv"
    df_train.to_csv(train_path, index=False)

    diagnostic_cols = [
        c for c in df_full.columns if c not in set(feature_cols + iv_cols + ["accepted_attempt"])
    ]

    metadata_path = output_dir / "heston_dataset_10000_8x15_clean_metadata.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("Heston IV dataset generation metadata: 8x15 restart\n")
        f.write("=" * 55 + "\n")
        f.write(f"S0={S0}\n")
        f.write(f"r={r}\n")
        f.write(f"T_grid={T_grid.tolist()}\n")
        f.write(f"K_grid={K_grid.tolist()}\n")
        f.write(f"K_grid_start={float(K_grid[0])}\n")
        f.write(f"K_grid_end={float(K_grid[-1])}\n")
        f.write(f"n_strikes={len(K_grid)}\n")
        f.write(f"n_maturities={len(T_grid)}\n")
        f.write(f"surface_dim={surface_dim}\n")
        f.write(f"M={M}\n")
        f.write(f"n_steps={n_steps}\n")
        f.write(f"scheme={scheme}\n")
        f.write(f"antithetic={antithetic}\n")
        f.write(f"clip_prices_for_iv={clip_prices_for_iv}\n")

        f.write("quality_filters:\n")
        f.write(f"  require_zero_iv_failures={require_zero_iv_failures}\n")
        f.write(f"  require_no_bound_violations={require_no_bound_violations}\n")
        f.write(f"  require_strike_monotonicity={require_strike_monotonicity}\n")
        f.write(f"  require_strike_convexity={require_strike_convexity}\n")
        f.write(f"  allow_repaired_prices_for_iv={allow_repaired_prices_for_iv}\n")

        f.write("sampling:\n")
        f.write(f"  enforce_feller={enforce_feller}\n")
        f.write(f"  log_sample_positive={log_sample_positive}\n")

        f.write("column_layout:\n")
        f.write(f"  feature_columns={feature_cols}\n")
        f.write(f"  n_feature_columns={len(feature_cols)}\n")
        f.write(f"  n_diagnostic_columns={len(diagnostic_cols)}\n")
        f.write(f"  n_iv_columns={len(iv_cols)}\n")
        f.write("  includes_accepted_attempt=True\n")
        f.write(f"train_columns={df_train.shape[1]}\n")
        f.write(f"total_rows={len(df_full)}\n")
        f.write(f"total_columns={df_full.shape[1]}\n")
        f.write("dataset_files:\n")
        f.write(f"  full_clean_csv={full_path.name}\n")
        f.write(f"  train_only_csv={train_path.name}\n")
        f.write("batch_files:\n")
        for p in batch_paths:
            f.write(f"- {p.name}\n")

    print("\n" + "=" * 80)
    print(f"Done. Full dataset saved to: {full_path}")
    print(f"Training-only dataset saved to: {train_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Full dataset shape: {df_full.shape}")
    print(f"Training-only shape: {df_train.shape}")
    print(f"Input columns ({len(feature_cols)}): {feature_cols}")
    print(f"Number of IV target columns: {len(iv_cols)}")


if __name__ == "__main__":
    main()
