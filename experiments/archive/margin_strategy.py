import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import optuna
import gymnasium as gym
from collections import deque

import os
path = os.getcwd()
df = pd.read_csv(f"{path}/new/test.csv")

with open("model.pkl","rb") as f:
  model = pickle.load(f)

with open("others.pkl","rb") as f:
  others = pickle.load(f)

with open("quantile.pkl","rb") as f:
  quantile = pickle.load(f)


class CarAuctionEnv(gym.Env):
    def __init__(self, df):
        super(CarAuctionEnv, self).__init__()
        self.cars = 500
        self.dataset = df
        self.bankroll = 500000.0
        self.wins = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = 500000.0
        self.wins = 0
        self.current_car_index = 0
        self.recent_wins = deque(maxlen=50)

        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.dataset.iloc[self.current_car_index]
        price = row["pred"]
        volatility = row["std"]
        
        # Calculate overall win rate safely
        win_rate = self.wins / max(1, self.current_car_index)
        
        return np.array([price, volatility, self.bankroll, self.current_car_index + 1, win_rate], dtype=np.float32)

    def step(self, agent_bid):
        row = self.dataset.iloc[self.current_car_index]
        selling_price = row["sellingprice"]
        predicted_price = row["pred"]

        # ---------------------------------------------------------
        # REALISTIC LOBBY: Simulating 4 standard student LLM scripts
        # ---------------------------------------------------------
        rival_bids = [
            predicted_price * 0.90,                           # The standard ChatGPT suggestion
            predicted_price * 0.92,                           # Slightly aggressive student
            predicted_price * 0.85,                           # Value investor
            predicted_price * np.random.normal(0.90, 0.05),    # Stochastic human error
            predicted_price * 0.97,
            predicted_price * 1.00
        ]
        
        highest_rival_bid = max(rival_bids)
        
        profit = 0
        won_auction = False

        # Second Price Auction Logic
        if agent_bid > highest_rival_bid:
            won_auction = True
            self.wins += 1
            price_paid = highest_rival_bid + 50.0

            if price_paid <= self.bankroll:
                profit = selling_price - price_paid
                self.bankroll += profit
            else:
                # Bankruptcy penalty
                won_auction = False
                profit = -self.bankroll
                self.bankroll = 0

        self.recent_wins.append(won_auction)
        self.current_car_index += 1

        terminated = bool((self.bankroll <= 0) or (self.current_car_index >= self.cars))

        info = {
            "won": won_auction,
            "true_price": selling_price,
            "agent_bid": agent_bid,
            "rival_bid": highest_rival_bid,
            "profit": profit
        }

        return self._get_obs(), profit, terminated, False, info


def objective(trial):
    # Optuna is now tuning the MARGINS directly
    alpha = trial.suggest_float("alpha", 0.05, 0.12)  # Base margin (5% to 12%)
    beta = trial.suggest_float("beta", 0.0, 0.05)     # Volatility penalty
    gamma = trial.suggest_float("gamma", 0.0, 0.10)   # Bankroll fear penalty
    delta = trial.suggest_float("delta", 0.0, 0.05)   # Hunger push (win rate drop)

    env = CarAuctionEnv(df)

    final_bankrolls = []
    total_wins = []

    # Run 50 different 500-car hackathons to find the most stable parameters
    for game in range(50):
        obs, _ = env.reset()
        terminated = False

        while not terminated:
            price = obs[0]
            volatility = obs[1]
            bankroll = obs[2]
            round_num = obs[3]
            win_rate = obs[4]

            # ---------------------------------------------------------
            # THE DETERMINISTIC "QUANT" BID SEQUENCE
            # ---------------------------------------------------------
            
            # Normalize volatility so expensive cars don't break the math
            vol_ratio = volatility / max(price, 1.0)
            
            # Base margin required
            margin = alpha
            
            # Add margin for high volatility (Winner's Curse protection)
            margin += beta * vol_ratio
            
            # Add margin if losing money (Bankroll preservation)
            bankroll_stress = max(0.0, (500000.0 - bankroll) / 500000.0)
            margin += gamma * bankroll_stress
            
            # Reduce margin if we aren't winning enough cars (Rubric: 20% Wins)
            hunger = max(0.0, 0.10 - win_rate)
            margin -= delta * hunger
            
            # Safety rails: Never demand more than 15% margin, never accept less than 2%
            margin = max(0.02, min(margin, 0.15))
            
            # Calculate final bid
            agent_bid = price * (1.0 - margin)

            if agent_bid >= bankroll:
                agent_bid = bankroll

            obs, reward, terminated, truncated, info = env.step(agent_bid)

        # Track the FINAL state of the game, not the last car's profit
        final_bankrolls.append(env.bankroll)
        total_wins.append(env.wins)

    # ---------------------------------------------------------
    # RUBRIC-ALIGNED OPTIMIZATION SCORE
    # ---------------------------------------------------------
    mean_bankroll = np.mean(final_bankrolls)
    std_bankroll = np.std(final_bankrolls)
    mean_wins = np.mean(total_wins)

    # 1. Profit (Net gain over starting $500k)
    net_profit = mean_bankroll - 500000.0
    
    # 2. Consistency (Penalize strategies that rely on luck)
    risk_penalty = 0.20 * std_bankroll
    
    # 3. Volume (Value each win at ~$1000 so Optuna doesn't become too passive)
    volume_bonus = mean_wins * 500.0

    # Optuna will maximize this total composite score
    return net_profit - risk_penalty + volume_bonus


# Run the study
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
print("Starting Rubric-Optimized Training...")
study.optimize(objective, n_trials=200, n_jobs=-1)

print("\n🏆 OPTIMIZATION COMPLETE 🏆")
print("Plug these exact parameters into your LiveAuctionAgent class:")
for key, value in study.best_params.items():
    print(f"  {key}: {value:.4f}")