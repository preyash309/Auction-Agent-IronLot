"""Reproducible new training runs; original notebooks remain archival evidence."""
import copy
import hashlib
import json
import pickle
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from .agent import ASSET_NAMES, FEATURES
from .preprocessing import finalizeData

CLEANUP = {"chev truck": "chevrolet", "gmc truck": "gmc", "ford truck": "ford", "dodge tk": "dodge", "hyundai tk": "hyundai", "mercedes-b": "mercedes-benz", "mercedes": "mercedes-benz", "landrover": "land rover", "vw": "volkswagen"}
BEST_PARAMS = {"n_estimators": 2400, "max_depth": 10, "num_leaves": 176, "learning_rate": 0.03038452742864471, "min_child_samples": 10}


def read_dataset(path):
    frame = pd.read_csv(path)
    required = set(FEATURES) | {"sellingprice"}
    if required - set(frame):
        raise ValueError(f"Dataset missing columns: {sorted(required - set(frame))}")
    # Preserve source row IDs before filtering, so runs can reconstruct the split.
    frame = frame.loc[:, list(FEATURES) + ["sellingprice"]]
    frame = frame.dropna(subset=["make", "model", "sellingprice"])
    if len(frame) < 10:
        raise ValueError("At least 10 complete labeled rows are required")
    if not np.isfinite(frame["sellingprice"]).all():
        raise ValueError("Target prices must be finite")
    return frame


def fit_metadata(frame):
    normalized = frame.copy()
    normalized["make"] = normalized["make"].astype(str).str.lower().str.strip().replace(CLEANUP).str.title()
    def modes(column):
        return normalized.groupby("make")[column].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown").to_dict()
    median = float(normalized["odometer"].median())
    if not np.isfinite(median):
        raise ValueError("Training data needs at least one finite odometer")
    return {"cleanup_names": CLEANUP, "trim_map": modes("trim"), "body_map": modes("body"), "median": median}


def prepare(frame, metadata):
    return finalizeData(frame.loc[:, list(FEATURES)].copy(), metadata["cleanup_names"], metadata["trim_map"], metadata["body_map"], metadata["median"])


def make_pipeline(seed, tuned=True, quantile=False, estimators=None):
    numeric = ["year", "condition", "odometer", "years_used", "miles_per_year", "luxury_brands", "is_rare", "depreciation"]
    categorical = ["make", "model", "trim", "body", "state", "color", "interior"]
    encoder = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), ["transmission"]),
        ("target", TargetEncoder(target_type="continuous", random_state=seed), categorical),
    ])
    params = copy.deepcopy(BEST_PARAMS) if tuned and not quantile else {}
    if estimators is not None:
        params["n_estimators"] = estimators
    if quantile:
        params.update(objective="quantile", alpha=0.1, metric="quantile")
    model = LGBMRegressor(**params, random_state=seed, n_jobs=1, verbose=-1)
    return Pipeline([("preprocessor", encoder), ("quantile" if quantile else "model", model)])


def metrics(target, predictions):
    return {"rows": len(target), "r2": float(r2_score(target, predictions)), "rmse": float(np.sqrt(mean_squared_error(target, predictions))), "mae": float(mean_absolute_error(target, predictions))}


def train(dataset, output, seed=42, baseline=False, estimators=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be a new or empty directory; existing checkpoints are preserved")
    frame = read_dataset(dataset)
    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=seed)
    metadata = fit_metadata(train_frame)
    x_train, x_test = prepare(train_frame, metadata), prepare(test_frame, metadata)
    model = make_pipeline(seed, tuned=not baseline, estimators=estimators)
    quantile = make_pipeline(seed, quantile=True, estimators=estimators)
    model.fit(x_train, train_frame["sellingprice"])
    quantile.fit(x_train, train_frame["sellingprice"])
    result = metrics(test_frame["sellingprice"], model.predict(x_test))
    result.update(seed=seed, train_rows=len(train_frame), baseline=baseline,
                  price_parameters=model.named_steps["model"].get_params(),
                  quantile_parameters=quantile.named_steps["quantile"].get_params(),
                  library_versions={name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")},
                  dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(), protocol="seeded 80/20 row split; preprocessing fitted on training only; no full-data refit")
    output.mkdir(parents=True, exist_ok=True)
    for name, obj in zip(ASSET_NAMES, (model, quantile, metadata)):
        with (output/name).open("wb") as stream:
            pickle.dump(obj, stream)
    (output/"metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    (output/"split.json").write_text(json.dumps({"train": train_frame.index.tolist(), "test": test_frame.index.tolist()}, indent=2)+"\n")
    return result
