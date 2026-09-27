# Verification record

Reconstruction verified locally on Windows with Python 3.10.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2 and LightGBM 4.6.0. Commands were run from the repository root. CI's Linux/Python 3.10 and 3.12 matrix is configured; its remote outcome is not claimed here.

## Initial reconstruction checks

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

## IronLot environment and agent-training integration

The later supplied `PreyashPratyush_IronLot/` folder contains seven files; all match the earlier submission copies by SHA256. The original final-agent source is unchanged. Notebook copies were sanitized and consolidated into that folder, with originals backed up separately. Binary checkpoints and the personal PDF remain local.

The complete suite now has **17 tests**, all passing locally with original checkpoints and agent-training dependencies. With `.[agent-training]` installed but no private assets, 16 pass and one checkpoint test skips. With core dependencies only, seven environment tests also skip. New checks cover:

- Original notebook versus adapted environment: exact reward, info and observation equality for controlled nonterminal win/loss transitions.
- Historical objective versus supported policy formula, including a bankroll above the starting balance.
- Gymnasium environment contract, seeded resets/noise and observation-space membership.
- Safe termination on the final dataset row and rejection of steps after episode end.
- Original bankruptcy penalty and rejected invalid actions/data.
- Two independently persisted, identically seeded small Optuna studies with identical results; study/trial files and output protection verified.

Executed cached-data smoke command:

```powershell
python -m auction_bot.optimize --data new/test.csv --output outputs/ironlot-smoke-integration --trials 2 --games 2 --rounds 10 --seed 42
```

It completed and saved `study.sqlite3`, `trials.csv` and `policy.json`. Mean final bankroll of its selected trial was **499972.1907827909**; selected alpha/beta/gamma were **0.9381218779 / 0.0780093202 / 0.0077997260**. These are small-run functional smoke results, not recommended deployment weights, external benchmarks or a reproduction of the original 500-trial study. Existing live-agent weights remain 0.1007 / 0.3792 / 0.0125. Input cache SHA256: `3218c8d0cae68e6d6dd0285c9f9710c967f25fcbf506c94ad6d022782d4ada2e`.

Gymnasium's checker warns that the declared bid/observation spaces have infinite bounds, which is intentional for arbitrary monetary scales. The spaces contain tested observations and the checker passes. SQLite study connections are explicitly closed after export to avoid Windows file locks. CI now installs `.[agent-training]` and checks both `auction_bot` and `auction_bot.optimize` startup on Python 3.10/3.12.
