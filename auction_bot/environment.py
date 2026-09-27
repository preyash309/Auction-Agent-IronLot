"""Portable form of IronLot's single-rival Gymnasium training environment.

Original economic equations are preserved; sampling/noise are seeded and
terminal observations no longer index beyond the dataset. This environment
uses maximum bids, unlike the live agent's incremental English-auction API.
"""
from collections import deque

import gymnasium as gym
import numpy as np


class CarAuctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset, rounds=500):
        super().__init__()
        required = ["pred", "std", "sellingprice"]
        if set(required) - set(dataset):
            raise ValueError("Environment data requires pred, std and sellingprice columns")
        if rounds <= 0 or len(dataset) < rounds:
            raise ValueError("Rounds must be positive and no greater than the number of rows")
        values = dataset.loc[:, required].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, 2] < 0).any():
            raise ValueError("Environment inputs must be finite; sellingprice must be nonnegative")
        self._source = dataset.loc[:, required].copy().reset_index(drop=True)
        self.cars = rounds
        self.action_space = gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, 0, 1, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = 500000.0
        self.current_car_index = 0
        self.recent_wins = deque(maxlen=50)
        self.wins = 0
        self.dataset = self._source.iloc[self.np_random.permutation(len(self._source))].reset_index(drop=True)
        self._done = False
        return self._get_obs(), {}

    def _get_obs(self):
        rate = sum(self.recent_wins)/len(self.recent_wins) if self.recent_wins else 0.0
        if self._done:
            return np.array([0, 0, self.bankroll, self.current_car_index+1, rate], dtype=np.float32)
        row = self.dataset.iloc[self.current_car_index]
        return np.array([row["pred"], row["std"], self.bankroll, self.current_car_index+1, rate], dtype=np.float32)

    def step(self, agent_bid):
        if self._done:
            raise RuntimeError("Call reset before stepping a new or completed episode")
        action = np.asarray(agent_bid, dtype=float)
        if action.size != 1:
            raise ValueError("Provide one maximum bid per step")
        bid = float(action.reshape(-1)[0])
        if not np.isfinite(bid) or bid < 0:
            raise ValueError("Bid must be finite and nonnegative")
        row = self.dataset.iloc[self.current_car_index]
        selling_price = float(row["sellingprice"])
        # Preserve notebook behavior: noise is based on true price, not std.
        rival_bid = selling_price + float(self.np_random.normal(0, selling_price*0.1))
        profit = 0.0
        won = False
        if bid > rival_bid:
            paid = rival_bid + 50.0
            if paid <= self.bankroll:
                won = True
                self.wins += 1
                profit = selling_price - paid
                self.bankroll += profit
            else:
                profit = -self.bankroll
                self.bankroll = 0.0
        self.recent_wins.append(won)
        self.current_car_index += 1
        self._done = self.bankroll <= 0 or self.current_car_index >= self.cars
        info = {"won": won, "true_price": selling_price, "agent_bid": bid, "rival_bid": rival_bid, "profit": profit}
        return self._get_obs(), profit, self._done, False, info
