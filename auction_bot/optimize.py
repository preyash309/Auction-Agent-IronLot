"""IronLot bidding-coefficient optimization, separate from ML model fitting."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .environment import CarAuctionEnv


def historical_max_bid(observation, alpha, beta, gamma):
    price, volatility, bankroll, _, win_rate = map(float, observation)
    base = price - alpha*volatility
    # Notebook tuning allows a negative penalty above starting bankroll.
    penalty = beta*(500000-bankroll)/500000*price
    boost = gamma*max(0, 0.2-win_rate)*price
    return max(0.0, min(base-penalty+boost, bankroll))


def score_policy(dataset, parameters, *, rounds=500, games=100, seed=42):
    if games <= 0:
        raise ValueError("Games must be positive")
    environment = CarAuctionEnv(dataset, rounds)
    bankrolls = []
    for game in range(games):
        observation, _ = environment.reset(seed=seed+game)
        done = False
        while not done:
            bid = historical_max_bid(observation, **parameters)
            observation, _, done, _, _ = environment.step(bid)
        bankrolls.append(environment.bankroll)
    environment.close()
    return float(np.mean(bankrolls))


def optimize(dataset_path, output, *, trials=500, games=100, rounds=500, seed=42):
    if trials <= 0 or games <= 0:
        raise ValueError("Trials and games must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Optimization output must be a new or empty directory")
    dataset = pd.read_csv(dataset_path)
    CarAuctionEnv(dataset, rounds)  # Validate before creating output artifacts.
    output.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.RDBStorage("sqlite:///"+(output/"study.sqlite3").resolve().as_posix())
    def objective(trial):
        parameters = {"alpha": trial.suggest_float("alpha",0.1,1.5),
                      "beta": trial.suggest_float("beta",0.0,0.5),
                      "gamma": trial.suggest_float("delta",0.0,0.05)}
        return score_policy(dataset, parameters, rounds=rounds, games=games, seed=seed)
    try:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    storage=storage, study_name="ironlot-bidding")
        study.optimize(objective, n_trials=trials, n_jobs=1)
        result = {
            "best_parameters": {"alpha": study.best_params["alpha"], "beta": study.best_params["beta"], "gamma": study.best_params["delta"]},
            "optuna_parameters": study.best_params, "mean_final_bankroll": study.best_value,
            "seed": seed, "trials": trials, "games_per_trial": games, "rounds_per_game": rounds,
            "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
            "library_versions": {name: version(name) for name in ("numpy","pandas","gymnasium","optuna")},
            "protocol": "IronLot single noisy rival; recent 50-round win rate; historical unclamped bankroll penalty; fixed per-game seeds across trials; single worker; negative bid proposals folded to zero",
            "status": "new synthetic-environment optimization; does not reproduce the original unseeded study or update live agent coefficients",
        }
        (output/"policy.json").write_text(json.dumps(result,indent=2)+"\n")
        study.trials_dataframe().to_csv(output/"trials.csv",index=False)
        return result
    finally:
        storage.remove_session()
        storage.engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimize bidding coefficients in the IronLot Gymnasium environment")
    parser.add_argument("--data", required=True, help="CSV with sellingprice, pred and std")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        result = optimize(args.data,args.output,trials=args.trials,games=args.games,rounds=args.rounds,seed=args.seed)
        print(json.dumps(result,indent=2))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(2,f"Error: {error}\n")


if __name__ == "__main__":
    main()
