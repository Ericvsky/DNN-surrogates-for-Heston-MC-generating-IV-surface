Clean file set for the final 8x15 Heston total-variance price-convexity project
================================================================================

Final research objective
------------------------
The main surrogate is no longer the direct-IV flat MLP. The final model is:

    Heston parameters Theta -> positive total variance w(T,K) -> IV -> BS prices

with a price-space strike-convexity loss. The 8x15 grid is:

    T = 0.25, 0.50, ..., 2.00
    K = 90, 92, 94, ..., 118
    output dimension = 8 x 15 = 120

Core files to keep in the new PyCharm project
---------------------------------------------
1. heston_mc_benchmark_toolkit.py
   Shared Monte Carlo / Heston / Black-Scholes toolkit. The demo block inside the toolkit may show an older grid, but the functions are grid-agnostic when called with explicit K_grid and T_grid.

2. benchmark_mc.py
   Monte Carlo baseline checks and figures for the final 8x15, K=90..118 grid.

3. build_heston_dataset.py
   Generates the 10000-row clean IV dataset. Writes the same output folder expected by the v4.1 scripts:
       heston_dataset_10000_8x15_output/

4. check_heston_dataset.py
   Verifies dataset shape, target columns, filters and quality diagnostics.

5. prepare_totalvar_price_dataset.py
   Derives total variance and BS price labels from the IV dataset for audit/documentation. The training script can compute these internally, but this file is useful for reporting.

6. train_heston_totalvar_priceconvex.py
   Main final DNN training script.

7. evaluate_heston_totalvar_priceconvex.py
   Main final DNN evaluation script.

8. benchmark_runtime_mc_and_totalvar_dnn.py
   Runtime benchmark for MC vs the final v4.1 DNN. This replaces the old flat-MLP runtime benchmark.

Recommended run order
---------------------
1. python benchmark_mc.py
2. python build_heston_dataset.py
3. python check_heston_dataset.py
4. python prepare_totalvar_price_dataset.py
5. python train_heston_totalvar_priceconvex.py
6. python evaluate_heston_totalvar_priceconvex.py
7. python benchmark_runtime_mc_and_totalvar_dnn.py

Files not needed for the final project folder
---------------------------------------------
- Old pilot scripts
- Old 8x21 scripts
- train_heston_mlp.py, unless you want to reproduce the direct-IV baseline comparison
- evaluate_heston_predictions_and_surface_quality.py, unless you want to reproduce the direct-IV baseline comparison
- benchmark_runtime_mc_and_dnn_10000_.py, because it benchmarks the old flat MLP rather than the final v4.1 DNN

Why there are multiple benchmark files
--------------------------------------
The project had different benchmark stages:

- benchmark_mc_*: validates the Monte Carlo generator and label quality.
- benchmark_runtime_*: measures speed of MC versus a trained DNN surrogate.
- heston_mc_benchmark_toolkit_*: reusable library functions used by other scripts.

For the project, keep one MC benchmark script and one runtime benchmark script only.
