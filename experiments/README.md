# Research archive

See [EXPERIMENTS.md](../EXPERIMENTS.md) for each experiment's objective, implementation, evidence and limitations.

`archive/` preserves original alternative scripts verbatim. They may execute expensive training at import, expect old filenames/current directories, or require absent assets. Do not import them as application modules. The final submitted agent now resides in [PreyashPratyush_IronLot](../PreyashPratyush_IronLot/README.md), alongside its valuation and environment notebooks. It is imported by regression tests with checkpoint loading bypassed or trusted local assets explicitly supplied.

The original notebook code and markdown are consolidated into `PreyashPratyush_IronLot/`, with outputs/Colab metadata removed and the cloud training CSV path changed to `../data/car_auction_train.csv`. They are historical methodology records. Their random splits and encoding/imputation behavior are not changed; full execution has not been revalidated. Original notebook outputs/plots/PDF remain in the private backup. The environment and objective now have tested supported counterparts in `auction_bot/environment.py` and `auction_bot/optimize.py`. No systematic feature-removal ablation results were found, so no ablation scores are claimed.
