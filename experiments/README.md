# Research archive

See [EXPERIMENTS.md](../EXPERIMENTS.md) for each experiment's objective, implementation, evidence and limitations.

`archive/` preserves original scripts verbatim. They may execute expensive training at import, expect old filenames/current directories, or require absent assets. Do not import them as application modules. The submitted agent is imported only by regression tests; its constructor is bypassed there unless trusted local checkpoints are explicitly enabled.

`notebooks/` preserves code and markdown with outputs/Colab metadata removed and the cloud training CSV path changed to `../../data/car_auction_train.csv`. These are historical methodology records. Their random splits and encoding/imputation behavior are not changed; execution has not been revalidated. Original notebook outputs/plots/PDF remain in the private backup. No systematic feature-removal ablation results were found, so no ablation scores are claimed.
