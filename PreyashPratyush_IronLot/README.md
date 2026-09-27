# IronLot: valuation models and trained auction agent

This is the original submission bundle that connects the project's two parts: **ML valuation models** and an **auction agent whose bidding coefficients were optimized in a Gymnasium auction environment**.

| File | Role |
|---|---|
| [analysis_Preyash_Pratyush.ipynb](analysis_Preyash_Pratyush.ipynb) | EDA, preprocessing, LightGBM price/quantile training, model-parameter search and checkpoint export |
| [environment_Preyash_Pratyush.ipynb](environment_Preyash_Pratyush.ipynb) | `CarAuctionEnv`, Optuna bidding-coefficient objective, selected coefficients and five-agent Monte Carlo evaluation |
| [agent_PreyashPratyush.py](agent_PreyashPratyush.py) | Original final live agent: loads both ML pipelines and saved preprocessing metadata, predicts car values, applies the optimized bidding formula |
| `model_PreyashPratyush.pkl`, `quantile_PreyashPratyush.pkl`, `others_PreyashPratyush.pkl` | Original local model/metadata artifacts; ignored by Git |
| `report_PreyashPratyush.pdf` | Original local report; ignored because it includes personal identifiers |

## Original training environment

The environment consumes precomputed `pred` (price prediction), `std` (price-minus-quantile gap, despite its name), and `sellingprice` columns. It starts with bankroll 500000 and runs 500 cars. Observations are `[price, gap, bankroll, round, recent_win_rate]`, with wins measured over the latest 50 rounds.

Actual code generates a rival bid as `sellingprice + Normal(0, sellingprice * 0.1)`. This differs from portions of the notebook narrative; the noise scale is the true price, not the predicted gap. The agent submits a maximum bid. If it exceeds the rival, it pays rival+50 when affordable, immediately resells at true price and receives the resulting profit as reward. An unaffordable purchase zeroes bankroll with a negative-bankroll reward. Episodes end after 500 rounds or bankruptcy.

Optuna maximizes mean final bankroll over 100 games per trial. The historical commented study specifies 500 trials, with alpha in [0.1,1.5], beta in [0,0.5], and boost weight in [0,0.05]. The boost weight is named `delta` in the Optuna search but becomes `GAMMA` in the submitted agent. Selected weights are 0.1007, 0.3792 and 0.0125. This is parameter training for a deterministic bidding policy; no neural policy or RL checkpoint is present.

## Supported execution

Use the package from the project root. The copied notebooks retain original methodology and are archival; their full execution is not revalidated.

```powershell
python -m pip install -e ".[agent-training]"
python -m auction_bot.optimize --data data/auction_predictions.csv --output outputs/ironlot-search --trials 500 --games 100 --rounds 500 --seed 42
```

`data/auction_predictions.csv` must be supplied; the original cached equivalent is local `new/test.csv` and is not published. The root README explains how to generate a new cache from authorized data and trusted checkpoints. For a smoke run, use `--trials 2 --games 2 --rounds 10`. Larger searches can be expensive.

[auction_bot/environment.py](../auction_bot/environment.py) exposes the environment with the original reward/opponent equations, seeded sampling/noise, declared Gymnasium spaces and safe terminal observations. [auction_bot/optimize.py](../auction_bot/optimize.py) persists new studies and policy results. It uses one worker and shared episode seeds for reproducible trial comparison, and maps negative policy proposals to zero. These changes are documented; they do not reproduce the original unseeded study. Existing live-agent coefficients are preserved and are not automatically overwritten by new studies.

## Evidence and preservation

All seven newly supplied files were hash-identical to the earlier `new/Test/` submission copies. Their untouched contents and hashes were backed up locally under `.reconstruction/ironlot/` before preparation for publication. Notebook outputs/metadata are removed to avoid embedded dataset previews; the analysis CSV path is changed to `../data/car_auction_train.csv`. Original notebook code and narrative otherwise remain unchanged, including historical discrepancies. Published duplicates under `experiments/notebooks/` and the duplicate submitted-agent archive were consolidated into this named folder.

The notebooks refer to `model.pkl`, `quantile.pkl` and `others.pkl`, while the submitted agent expects participant-suffixed filenames. Supported commands use the suffixed artifact names and explicit directories. The original simulation's empirical outcome percentiles are recorded in [EXPERIMENTS.md](../EXPERIMENTS.md); they do not demonstrate a 95% chance of profit. The tuning environment uses a recent win window and an unclamped bankroll penalty, while live inference uses lifetime wins and clamps capital stress at zero.

Dataset/model redistribution rights and original Optuna trial records remain undocumented. Binary checkpoints, cached data and the personal report stay local under the existing publication policy. See the [root README](../README.md) for the complete system, setup and supported commands.
