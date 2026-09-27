# Verification record

Reconstruction verified locally on Windows with Python 3.10.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2 and LightGBM 4.6.0. Commands were run from the repository root. CI's Linux/Python 3.10 and 3.12 matrix is configured; its remote outcome is not claimed here.

## Executed checks

| Check | Actual outcome |
|---|---|
| Full original backup | 19,223 files, ZIP integrity verified; per-file SHA256 inventory retained privately |
| Original assets | Both copies of dataset/checkpoint triples hash-identical |
| Automated tests | 10 tests pass with original local checkpoints enabled; without checkpoints, 9 pass and 1 explicitly skips |
| Checkpoint regression | Exact price, quantile and bid equality against submitted agent, including missing numeric fields, rare make, missing identity and 2016 age edge case |
| Bidding regression | Original formula compared across 5 bankrolls × 4 win rates × 4 current bids |
| Synthetic training test | Trains both pipelines on 40 rows, reloads artifacts, verifies disjoint saved split and protects output directory |
| Seeded simulation test | Same seed yields identical outputs; counts and invalid round requests checked |
| Installation | Editable package build/install succeeds; old bundled pip/setuptools upgraded locally first |
| Dependencies | `python -m pip check`: no broken requirements |
| CLI | Help and installed `auction-bot` entry point checked; prediction, training, evaluation and simulation execute |

Run the core suite:

```powershell
python -m unittest discover -s tests -v
```

Include local checkpoint regression:

```powershell
$env:AUCTION_TEST_CHECKPOINTS="checkpoints"
python -m unittest discover -s tests -v
```

Checkpoint comparisons are compatibility checks, not independent price-validation results. No original conventional automated suite existed: files named test.py/test2.py/test3.py are training/optimization programs with import side effects. Expensive historical Optuna studies and 10000-game simulations were not rerun.

## New baseline holdout result

Executed `python -m auction_bot train --data data/car_auction_train.csv --output outputs/baseline-seed42 --baseline --seed 42` using the full locally available CSV. After filtering missing make/model/target: 438,676 rows; 350,940 train and 87,736 holdout. Split uses seed 42; metadata and encoders are fit on training rows; price and quantile encoders are independent. The model is not refit on full data.

| Metric | Holdout value |
|---|---:|
| R² | 0.9217397391 |
| RMSE | 2669.6258349 |
| MAE | 1698.8989573 |

Source CSV SHA256: `41eef8c0612696e07c7e551ccec627caa2d7337211914bca7c7d5e1785e57aea`. Run artifacts and exact source row split are local in ignored `outputs/baseline-seed42/`; metrics are also recorded in [validation_metrics.json](validation_metrics.json). The CLI `evaluate` was checked against that saved holdout. This is one seeded random holdout, not a time-based or external benchmark, and does not reproduce the original unseeded split. Full tuned-model retraining was not run; its entry point shares the tested baseline path and stored tuned parameters.

## Original checkpoint smoke outputs

Example prediction: mean-price estimate **21606.4186142**, 0.1-quantile estimate **17090.3446864**, next bid **10050** from current bid 10000.

Original-checkpoint simulation: 500 sampled cars, seed 42, 9 missing-identity skips, 491 auctioned, 304 wins, win rate **0.6191446**, ending bankroll **641107.5452**. This uses synthetic true-price-based opponents, immediate resale and possible training-data overlap; it does not establish real-world profitability or reproduce the historical five-agent Monte Carlo study.

## Warnings and external limitations

Legacy inference emits scikit-learn feature-name warnings because pipelines feed transformed arrays into LightGBM. Missing numeric values can trigger pandas' downcasting FutureWarning. They did not prevent execution or equality tests; pinned versions preserve current behavior. The 2016 infinite mileage feature remains intentionally unchanged. Original datasets/checkpoints, PDF and exact notebook outputs are local only; a fresh clone needs authorized data or trusted model files. Dataset provenance/rights, historical split IDs and Optuna study records remain unavailable.
