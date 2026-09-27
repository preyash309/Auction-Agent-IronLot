# Experiments and historical evidence

The supported path is `python -m auction_bot`. Files under `experiments/archive/` are copies of original research, retain original side effects/paths, and are **not supported entry points**. Notebooks retain original code and narrative, with outputs removed for publication; the full originals, outputs, figures and PDF are preserved locally in `new/Test/` and the private backup. Historical results below were transcribed from stored output and are not independent reruns.

## Pricing experiments

| Experiment | Evidence | Method and conclusion |
|---|---|---|
| Exploratory analysis and base model | [Analysis notebook](experiments/notebooks/analysis_Preyash_Pratyush.ipynb) | Missing identity filtering, manufacturer normalization, split-local imputation; engineered age/mileage/luxury/rare/depreciation. Default LightGBM with one-hot and target encoding. Stored R² 0.9128612418; RMSE 2902.1986034. |
| Tuned price model | Same notebook | Optuna objective: five-fold negative RMSE, 20 trials commented out; selected 2400 trees, depth 10, 176 leaves, learning rate 0.0303845274, min child samples 10. Stored holdout R² 0.9537913169; RMSE 2096.4591634. No saved trial database. Narrative/PDF claims RMSE 2244.34; discrepancy unresolved. |
| Quantile model | Same notebook | LightGBM quantile objective, alpha 0.1, default tree count, seed 42. Provides lower conditional quantile for risk heuristic. No saved calibration result. |
| Model-frequency rarity alternative | [price_frequency_rarity.py](experiments/archive/price_frequency_rarity.py) | `is_rare` uses fewer than 50 model occurrences within each split, unlike final brand-based feature. Uses same tuned price parameters. No recorded metrics in the script; not equivalent to submitted feature semantics. |
| Rare-brand typo variant | [rare_string_agent.py](experiments/archive/rare_string_agent.py) | Contains one string `Rolls-Royce,Lamborghini` instead of two brands, so normal rare cars are never flagged. Kept as provenance; final [submitted_agent.py](experiments/archive/submitted_agent.py) matches analysis notebook's two-brand list. |

The notebook's “TRAINING ON FULL DATASET” section concatenates rows and predicts; it does **not** fit the model again. Its R² 0.9690607584 / RMSE 1707.2655089 mix training and test rows and are not holdout metrics. The original price/quantile pipelines share and refit one encoder object. Original imputation derives separate train/test statistics, while exported maps derive from combined rows. These limit interpretation of historical scores.

## Bidding experiments

| Experiment | Evidence | Configuration, evaluation and result |
|---|---|---|
| Noisy single rival tuning | [Environment notebook](experiments/notebooks/environment_Preyash_Pratyush.ipynb) | Rival = true price + Normal(0, true price × 0.1); 500 cars, starting 500000. Optuna objective averages bankroll over 100 games. Commented 500 trials, coefficients alpha 0.1007, beta 0.3792, boost 0.0125; trial records absent. This is tuning a heuristic, not neural/RL training. |
| Five-agent Monte Carlo | Same notebook | 10000 × 500-car games. Opponents: Normal(prediction, absolute prediction-minus-quantile), 90% prediction, quantile+50, bankroll-dependent 98/90/75% prediction. Second-highest bid+50 payment; immediate true-price resale. Stored agent mean bankroll 522587.5195, empirical 2.5–97.5 percentiles 486960.1917–568090.1400. Mean wins 102.3247, range 55–128; mean win rate 0.2046494, range 0.11–0.256. |
| Earlier tuning/simulation variant | [noisy_rival_strategy.py](experiments/archive/noisy_rival_strategy.py) | Additional unused gamma trial parameter and a hunger term of 1−win_rate in simulation; differs from submitted max(0,0.2−win_rate). No independently reproduced results. |
| Margin/rubric alternative | [margin_strategy.py](experiments/archive/margin_strategy.py) | Margin clamped 2–15%; volatility ratio, bankroll stress, hunger. Six rival bids despite comment saying four; 50 games per trial, 200 Optuna trials; objective profit−0.2×bankroll std+500×wins. No saved results or evidence it became the final strategy. |
| English-auction smoke harnesses | [synthetic_tournament.py](experiments/archive/synthetic_tournament.py), [dataset_tournament.py](experiments/archive/dataset_tournament.py), [submission_tournament.py](experiments/archive/submission_tournament.py) | Random synthetic or 500 sampled real cars, rising bids against true-price-based opponents, immediate resale. Adapted into seeded supported `simulate` command. Original scripts are unseeded and swallow some errors. |
| Example interface | [example_agent.py](experiments/archive/example_agent.py) | Label encoders and 90% price bidding; referenced example checkpoints are absent. Historical scaffold, not a working alternative model. |
| Unrelated conversion classification | [conversion_classifier.py](experiments/archive/conversion_classifier.py) | User conversion prediction, target encoding and LightGBM binary classification; 5-fold F1 and Optuna threshold search (50 trials). `new/df/` data and CatBoost F1/logloss logs suggest other classification work. No evidence links it to car pricing; kept separate. |

### Interpretation limits

The original Monte Carlo “95% CI” is an empirical outcome percentile interval, not a confidence interval for a mean, and its lower endpoint is below the starting bankroll. The assertion that the agent profits 95% of the time is unsupported. Original sampling and NumPy noise are unseeded. The Gym environments call the next observation even at episode end, potentially indexing past available rows; their reset seed does not govern pandas sampling/global noise. Tuning uses a recent 50-win window whereas the live agent uses lifetime win rate. Some tuning variants allow a negative bankroll penalty above the starting balance; submitted inference clamps it.

## New reproducible experiments

Run `train --baseline` and `train` with the same dataset and seed to compare default versus stored tuned hyperparameters. New runs fit metadata on training rows only, seed the encoders and models, and use independent pipelines. `split.json` and dataset SHA256 bind metrics to source row IDs. The quantile tree override and explicit `--estimators` are for smoke tests; use defaults for methodological comparisons. A new full baseline run and checkpoint regression checks are recorded in [validation](docs/VALIDATION.md). Full historical Optuna searches and 10000-game reproduction were not rerun.
