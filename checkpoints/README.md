# Trusted checkpoint setup

Place these three files together, or pass another directory with `--checkpoints` / `AUCTION_CHECKPOINT_DIR`:

- `model_PreyashPratyush.pkl`: fitted sklearn price pipeline.
- `quantile_PreyashPratyush.pkl`: fitted sklearn 0.1-quantile pipeline.
- `others_PreyashPratyush.pkl`: cleanup map, per-make trim/body maps, odometer median.

The local original copies under `new/` and `new/Test/` were hash-identical. Canonical copies are retained here but ignored by Git. Expected originals' sizes and SHA256 hashes are in [checkpoint_manifest.json](../docs/checkpoint_manifest.json). The manifest is for original artifacts, not newly trained ones.

To produce new artifacts, run `python -m auction_bot train --data data/car_auction_train.csv --output outputs/my-run --seed 42`. Then pass `--checkpoints outputs/my-run`. Python 3.10 with the pinned numerical dependencies loaded the originals successfully; the original notebooks ran under Python 3.12, but original checkpoint serialization environment is not fully recorded.

Only load trusted pickles: they can execute arbitrary code. No model download source or model redistribution license was documented. Model binaries are excluded from publication.
