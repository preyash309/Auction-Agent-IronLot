# Publication hygiene and rights

There was no original Git history to audit. Newly created commits are limited to source, sanitized notebooks, documentation, example JSON, package metadata and tests/CI. A local scanner checks tracked text for common GitHub/AWS credentials, private-key headers, quoted secret assignments, personal report identifiers, notebook outputs and unexpected large files. The scan is a useful check, not a guarantee against every possible secret format.

Excluded by `.gitignore`:

- `new/`: original evidence mixed with a roughly 1 GB Windows environment, caches, duplicate code and assets.
- `.reconstruction/`: full backup, hashed inventory, private report, tool download and verification logs.
- CSV datasets and cached predictions: provenance and redistribution rights unverified.
- Pickle models/metadata: binaries, undocumented redistribution terms; trusted local loading only.
- Original PDF: personal student identifiers, residence and email; technical findings documented separately.
- IronLot bundle: model pickle files and report remain ignored; only final agent source, sanitized notebooks and documentation are published. Original notebook outputs/files have a separate private backup.
- Training outputs, environments, caches, logs and editor artifacts: generated or machine-specific.

No credential is required by the application. `.env.example` contains only a checkpoint path. Git commit metadata uses the account's GitHub noreply address locally; no global Git configuration was changed. Authentication tokens must stay in the normal credential store and must never enter a project file or log.

No license file or ownership grant existed in the original project. The reconstruction does not add an open-source license, assert dataset ownership, or claim permission to redistribute the checkpoints. Core dependencies (NumPy, pandas, SciPy, scikit-learn, LightGBM) and optional research packages remain governed by their upstream distributions' licenses; their installed binaries/licenses are not vendored here. Confirm data/model rights before any future redistribution or public release. The requested GitHub visibility is private.
