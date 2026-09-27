"""Submission-compatible live auction interface with explicit asset location."""
import math
import os
import pickle
from pathlib import Path

import pandas as pd

from .preprocessing import finalizeData

ASSET_NAMES = ("model_PreyashPratyush.pkl", "quantile_PreyashPratyush.pkl", "others_PreyashPratyush.pkl")
FEATURES = ("year", "make", "model", "trim", "body", "transmission", "state", "condition", "odometer", "color", "interior")


class LiveAuctionAgent:
    """Analyze a car, bid in 50-unit increments, then accept official results.

    Pickle files execute code during loading: only use trusted local artifacts.
    The initial bankroll and strategy coefficients match the submitted agent.
    """

    def __init__(self, checkpoint_dir=None):
        self.bankroll = 500000.0
        self.predicted_price = self.p10_price = 0.0
        self.win_rate = self.wins = self.round_nums = 0
        self.car_make = self.car_model = ""
        self._analyzed = False
        directory = checkpoint_dir or os.environ.get("AUCTION_CHECKPOINT_DIR") or "checkpoints"
        self.checkpoint_dir = Path(directory).expanduser().resolve()
        missing = [name for name in ASSET_NAMES if not (self.checkpoint_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints in {self.checkpoint_dir}: {', '.join(missing)}. See checkpoints/README.md.")
        objects = []
        for name in ASSET_NAMES:
            with (self.checkpoint_dir / name).open("rb") as stream:
                objects.append(pickle.load(stream))
        self.model, self.quantile, self.others = objects
        if not all(key in self.others for key in ("cleanup_names", "trim_map", "body_map", "median")):
            raise ValueError("Preprocessing metadata is incomplete")

    def analyze_item(self, item_features: dict):
        # Clear prior predictions so an invalid next item cannot reuse its bid.
        self._analyzed = False
        self.predicted_price = self.p10_price = 0.0
        self.car_make = self.car_model = ""
        missing = set(FEATURES) - item_features.keys()
        if missing:
            raise ValueError(f"Missing vehicle fields: {', '.join(sorted(missing))}. Use null for unknown values.")
        car = pd.DataFrame([{key: item_features[key] for key in FEATURES}])
        car = finalizeData(car, self.others["cleanup_names"], self.others["trim_map"], self.others["body_map"], self.others["median"])
        self.car_make, self.car_model = str(car["make"].iloc[0]), str(car["model"].iloc[0])
        if self.car_make != "0" and self.car_model != "0":
            price = float(self.model.predict(car)[0])
            p10 = float(self.quantile.predict(car)[0])
            if not math.isfinite(price) or not math.isfinite(p10):
                raise ValueError("Model returned a non-finite price")
            self.predicted_price, self.p10_price = price, p10
        self._analyzed = True

    def place_bid(self, current_highest_bid: float) -> float:
        if not math.isfinite(current_highest_bid) or current_highest_bid < 0:
            raise ValueError("Current bid must be finite and nonnegative")
        if not self._analyzed or self.car_make == "0" or self.car_model == "0":
            return 0.0
        price, p10 = self.predicted_price, self.p10_price
        base = price - 0.1007 * (price - p10)
        penalty = 0.3792 * max(0, 500000 - self.bankroll) / 500000 * price
        boost = 0.0125 * max(0.0, 0.2 - self.win_rate) * price
        maximum = min(base - penalty + boost, self.bankroll)
        next_bid = current_highest_bid + 50.0
        return next_bid if next_bid <= maximum and next_bid <= self.bankroll else 0.0

    def auction_result(self, won, winning_bid, actual_price, current_bankroll):
        if not math.isfinite(current_bankroll) or current_bankroll < 0:
            raise ValueError("Official bankroll must be finite and nonnegative")
        self.bankroll = current_bankroll
        self.wins += int(bool(won))
        self.round_nums += 1
        self.win_rate = self.wins / self.round_nums
