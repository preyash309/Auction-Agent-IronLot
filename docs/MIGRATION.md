# Reconstruction and migration map

No original source or research evidence was deleted. A verified full ZIP backup and CSV inventory of 19,223 original files (path, size, SHA256) are local under ignored `.reconstruction/`. Original files remain in place. There was no Git history or remote to preserve; the first new commit captures publication-safe research copies.

| Original | New destination | Reason |
|---|---|---|
| `new/Test/agent_PreyashPratyush.py` | `auction_bot/agent.py`, `auction_bot/preprocessing.py`; verbatim `experiments/archive/submitted_agent.py` | Supported submission implementation, split into asset/bidding and preprocessing responsibilities |
| `new/agent.py` | `experiments/archive/rare_string_agent.py` | Preserve divergent rare-feature typo outside supported flow |
| `new/test.py` | `experiments/archive/price_frequency_rarity.py` | Meaningful earlier pricing feature alternative |
| `new/test2.py` | `experiments/archive/noisy_rival_strategy.py` | Early opponent/tuning and Monte Carlo variant |
| `new/test3.py` | `experiments/archive/margin_strategy.py` | Separate margin/rubric alternative with no final adoption evidence |
| `new/test_run.py` | `experiments/archive/synthetic_tournament.py` | Historical synthetic smoke harness |
| `new/new_test.py`, `new/Test/new_test.py` | `experiments/archive/dataset_tournament.py`, `submission_tournament.py`; adapted `auction_bot/simulation.py` | Preserve originals and provide one seeded runnable simulation |
| `new/agent_Example.py` | `experiments/archive/example_agent.py` | Example interface; missing example assets |
| `trainingmodels.py` | `experiments/archive/conversion_classifier.py` | Unrelated classifier separated from auction workflow |
| `new/Test/*.ipynb` | `experiments/notebooks/` | Preserve code/markdown, remove outputs/metadata and portable dataset path; original notebook evidence retained locally |
| Duplicate `new/` and `new/Test/` checkpoint triples | local `checkpoints/` plus `docs/checkpoint_manifest.json` | Hash-identical originals, compatible names retained; binaries excluded |
| Duplicate car CSVs | local `data/car_auction_train.csv` | Hash-identical originals; data excluded |
| `new/Test/report_PreyashPratyush.pdf` | local original/backup; technical findings in `EXPERIMENTS.md` | Contains personal identifiers; exclude from publication |
| `new/test.csv`, `new/df/`, `catboost_info/` | retained locally | Cached experiment data, unrelated classification data/logs |
| `new/tempCodeRunnerFile.py`, Python runtime/cache folders | retained locally | Incomplete editor snippet and environment, no supported contribution |

## Behavior changes

Inference feature formulas, submitted bidding coefficients, 500000 starting bankroll, 50 increment and evaluator-controlled settlement are unchanged. Tests compare exact predictions/bids with original checkpoints, including missing values, a rare make and 2016 infinite mileage-per-year. Required field validation, finite price/bid/bankroll checks, clearing stale predictions, actionable missing-asset errors and configurable paths were added. Extra input fields are excluded. Missing identity retains the original fold behavior.

New training is a documented protocol, not a claim to reconstruct the exact lost split: seeded 80/20 row split, train-only imputation, separate price/quantile encoders, seeded target encoding, one LightGBM thread, saved row IDs/hash and no full-data refit. It retains feature definitions, selected price hyperparameters and the 0.1 quantile objective. Retrained checkpoints use the original names, but have different learned parameters and must be evaluated separately.

Simulation retains English-auction opponent formulas and immediate resale from the submission harness, with explicit sampling/noise seeds, positive round validation, honest counts and surfaced errors. Original second-price/Gym variants remain archival and are not silently replaced by this simulation.
