from __future__ import annotations

"""
Full quality check for the restarted 8 x 15 Heston dataset.

Expected files
--------------
heston_dataset_10000_8x15_output/heston_dataset_10000_8x15_clean_full.csv
heston_dataset_10000_8x15_output/heston_dataset_10000_8x15_clean_train_only.csv

Expected shapes
---------------
Full clean dataset:    (10000, 136)
Training-only dataset: (10000, 125)
"""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLS = ["v0", "rho", "kappa", "theta", "sigma_v"]
K_GRID = np.linspace(90.0, 118.0, 15, dtype=float)
assert len(K_GRID) == 15 and abs(K_GRID[0] - 90.0) < 1e-12 and abs(K_GRID[-1] - 118.0) < 1e-12
T_GRID = np.array([0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00], dtype=float)
EXPECTED_SURFACE_DIM = len(K_GRID) * len(T_GRID)
EXPECTED_FULL_COLS = 5 + 10 + EXPECTED_SURFACE_DIM + 1
EXPECTED_TRAIN_COLS = 5 + EXPECTED_SURFACE_DIM


def expected_target_columns(k_grid: np.ndarray, t_grid: np.ndarray) -> list[str]:
    return [f"iv_T{float(t):.2f}_K{float(k):.2f}" for t in t_grid for k in k_grid]


def main() -> None:
    base = Path(__file__).resolve().parent
    out = base / "heston_dataset_10000_8x15_output"
    full_path = out / "heston_dataset_10000_8x15_clean_full.csv"
    train_path = out / "heston_dataset_10000_8x15_clean_train_only.csv"

    print("Full dataset path:", full_path)
    print("Training-only path:", train_path)

    if not full_path.exists():
        raise FileNotFoundError(full_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    full = pd.read_csv(full_path)
    train = pd.read_csv(train_path)

    iv_cols = [c for c in full.columns if c.startswith("iv_T")]
    expected_iv_cols = expected_target_columns(K_GRID, T_GRID)
    diag_cols = [c for c in full.columns if c not in set(FEATURE_COLS + iv_cols + ["accepted_attempt"])]

    print("\nShapes")
    print("-" * 40)
    print("Full shape:", full.shape)
    print("Train-only shape:", train.shape)
    print("Feature columns:", len(FEATURE_COLS))
    print("Diagnostic columns:", len(diag_cols))
    print("IV columns:", len(iv_cols))
    print("Has accepted_attempt:", "accepted_attempt" in full.columns)

    assert EXPECTED_SURFACE_DIM == 120
    assert full.shape == (10000, EXPECTED_FULL_COLS), f"Unexpected full shape: {full.shape}"
    assert train.shape == (10000, EXPECTED_TRAIN_COLS), f"Unexpected train-only shape: {train.shape}"
    assert len(iv_cols) == EXPECTED_SURFACE_DIM, f"Expected {EXPECTED_SURFACE_DIM} IV columns, found {len(iv_cols)}"
    assert len(diag_cols) == 10, f"Expected 10 diagnostic columns, found {len(diag_cols)}"
    assert iv_cols == expected_iv_cols, "IV target columns are not in the expected 8x15 order"
    assert train.columns.tolist() == FEATURE_COLS + expected_iv_cols, "Train-only column order is wrong"

    print("\nNaN checks")
    print("-" * 40)
    print("NaNs in full feature/IV block:", full[FEATURE_COLS + iv_cols].isna().sum().sum())
    print("NaNs in train-only file:", train.isna().sum().sum())
    assert full[FEATURE_COLS + iv_cols].isna().sum().sum() == 0
    assert train.isna().sum().sum() == 0

    print("\nEnforced quality filters")
    print("-" * 40)
    enforced = [
        "iv_failures",
        "repaired_prices_for_iv",
        "viol_lower_bound",
        "viol_upper_bound",
        "monotonicity_violations_across_strike",
    ]
    for col in enforced:
        print(f"{col}: total={int(full[col].sum())}, max={int(full[col].max())}")
        assert (full[col] == 0).all(), f"Filter failed for {col}"

    print("\nDiagnostic-only checks, not enforced")
    print("-" * 40)
    for col in ["convexity_violations_across_strike", "monotonicity_violations_across_maturity"]:
        clean_pct = (full[col] == 0).mean() * 100.0
        print(f"{col}: clean_pct={clean_pct:.4f}%, mean={full[col].mean():.6f}, max={int(full[col].max())}")

    print("\nParameter ranges")
    print("-" * 40)
    for col in FEATURE_COLS:
        print(f"{col}: min={full[col].min():.6g}, mean={full[col].mean():.6g}, max={full[col].max():.6g}")

    print("\nFeller-ratio diagnostics")
    print("-" * 40)
    feller_ratio = 2.0 * full["kappa"] * full["theta"] / (full["sigma_v"] ** 2)
    print(f"Feller ratio min:  {feller_ratio.min():.6g}")
    print(f"Feller ratio mean: {feller_ratio.mean():.6g}")
    print(f"Feller ratio max:  {feller_ratio.max():.6g}")
    print(f"Pct satisfying Feller ratio >= 1: {(feller_ratio >= 1.0).mean() * 100:.4f}%")

    print("\nAcceptance-attempt diagnostics")
    print("-" * 40)
    batch_files = sorted(out.glob("heston_dataset_10000_8x15_clean_batch_*.csv"))
    if batch_files:
        accepted_total = 0
        attempts_total = 0
        for p in batch_files:
            b = pd.read_csv(p, usecols=["accepted_attempt"])
            accepted_total += len(b)
            attempts_total += int(b["accepted_attempt"].iloc[-1])
        acceptance_rate = accepted_total / attempts_total
        print("Batch files:", len(batch_files))
        print("Accepted rows from batches:", accepted_total)
        print("Total attempts from batches:", attempts_total)
        print(f"Approx acceptance rate: {acceptance_rate:.4%}")
    else:
        print("No batch files found, skipping acceptance-rate calculation.")

    print("\nFull 10,000-row 8x15 dataset checks passed.")


if __name__ == "__main__":
    main()
