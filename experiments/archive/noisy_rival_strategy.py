import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import optuna
import gymnasium as gym
from collections import deque
import os

with open("model.pkl","rb") as f:
  model = pickle.load(f)

with open("others.pkl","rb") as f:
  others = pickle.load(f)

with open("quantile.pkl","rb") as f:
  quantile = pickle.load(f)

path = os.getcwd()
df = pd.read_csv(f"{path}/new/test.csv")

class CarAuctionEnv(gym.Env):

  def __init__(self,df):
    super(CarAuctionEnv, self).__init__()
    self.cars = 500
    self.dataset = df
    self.bankroll = 500000

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.bankroll = 500000
    self.current_car_index = 0
    self.recent_wins = deque(maxlen=50)

    self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)
    return self._get_obs(), {}

  def _get_obs(self):
    row = self.dataset.iloc[self.current_car_index]

    price = row["pred"]
    volatility = row["std"]
    win_rate = sum(self.recent_wins) / len(self.recent_wins) if len(self.recent_wins) > 0 else 0.0

    return np.array([price,volatility,self.bankroll, self.current_car_index + 1, win_rate], dtype=np.float32)

  def step(self, agent_bid):
    row = self.dataset.iloc[self.current_car_index]
    selling_price = row["sellingprice"]

    noise = np.random.normal(loc=0, scale=selling_price*0.1)
    rival_bid = selling_price + noise

    profit = 0
    won_auction = False

    if agent_bid > rival_bid:
      won_auction = True
      price_paid = rival_bid + 50.0

      if price_paid <= self.bankroll:
        profit = selling_price - price_paid
        self.bankroll += profit

      else:
        won_auction = False
        profit = -self.bankroll
        self.bankroll = 0

    self.recent_wins.append(won_auction)
    self.current_car_index += 1

    terminated = bool((self.bankroll <= 0) or (self.current_car_index >= self.cars))

    reward = profit

    info = {
        "won":won_auction,
        "true_price":selling_price,
        "agent_bid":agent_bid,
        "rival_bid":rival_bid,
        "profit":profit
    }

    return self._get_obs(), reward, terminated, False, info

def objective(trial):
  alpha = trial.suggest_float("alpha",0.1,1.5)
  beta = trial.suggest_float("beta",0.0,0.5)
  gamma = trial.suggest_float("gamma",0.0,1.0)
  delta = trial.suggest_float("delta",0.0,0.05)

  env = CarAuctionEnv(df)

  trial_scores = []
  for game in range(100):
    obs, _ = env.reset()
    terminated = False

    while not terminated:
      price = obs[0]
      volatility = obs[1]
      bankroll = obs[2]
      round_num = obs[3]
      win_rate = obs[4]

      hunger = max(0,0.2 - win_rate)
      base = price - (alpha * volatility)
      fear = beta * (max(0,500000) - bankroll)/500000 * price
      boost = delta * hunger * price

      agent_bid = base - fear + boost

      if agent_bid >= bankroll:
        agent_bid = bankroll

      obs, reward, terminated, truncated, info = env.step(agent_bid)

    trial_scores.append(env.bankroll)

  return np.mean(trial_scores)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials = 500, n_jobs=-1)

#study.best_params:
#   ALPHA: 0.1007
#   BETA: 0.3792
#   GAMMA: 0.9595
#   DELTA: 0.0125

class simulateAgents:

  def agent1Bid(self,price,volatility,bakroll,round_num,win_rate):
    alpha = 0.1007
    beta = 0.3792
    gamma = 0.9595
    delta = 0.0125

    hunger = 1 - win_rate
    base = price - (alpha * volatility)
    fear = beta * (max(0,500000 - bakroll))/500000 * price
    boost = delta * hunger * price

    bid = base - fear + boost

    return bid

  def agent2Bid(self,price, volatility):
    bid = np.random.normal(loc=price, scale=abs(volatility))
    return max(0.0, bid)

  def agent3Bid(self,price):
    return price * 0.9

  def agent4Bid(self,p10):
      return p10 + 50.0

  def agent5Bid(self,price, current_bankroll):
    if current_bankroll > 400000:
        return price * 0.98
    elif current_bankroll > 200000:
        return price * 0.90
    else:
        return price * 0.75

def runSimulation(df):
  bankroll = [500000.0] * 5
  wins = [0] * 5
  win_rate = [0.0] * 5
  round_num = 0
  MAX_SIMULATIONS = 500

  agents = simulateAgents()
  sampled_df = df.sample(n=MAX_SIMULATIONS, replace=False).reset_index(drop=True)
  for index in range(MAX_SIMULATIONS):
      row = sampled_df.iloc[index]
      true_price = row['sellingprice']
      hammer_price = row['pred']
      volatility = row['std']
      p10 = row["quantile"]

      bid1 = agents.agent1Bid(hammer_price, volatility, bankroll[0], round_num, win_rate[0])
      bid2 = agents.agent2Bid(hammer_price, volatility)
      bid3 = agents.agent3Bid(hammer_price)
      bid4 = agents.agent4Bid(p10)
      bid5 = agents.agent5Bid(hammer_price,bankroll[4])

      bids = [bid1, bid2, bid3, bid4, bid5]

      sorted_bids = sorted(bids, reverse=True)
      highest_bid = sorted_bids[0]
      second_highest_bid = sorted_bids[1]

      price_paid = second_highest_bid + 50.0
      winner_index = bids.index(highest_bid)

      if price_paid <= bankroll[winner_index]:
          profit = true_price - price_paid
          bankroll[winner_index] += profit
          wins[winner_index] += 1

      for i in range(5):
          win_rate[i] = wins[i] / (round_num + 1)

      round_num += 1

  return bankroll, wins, win_rate

N_SIMULATIONS = 10000

result_bankroll = []
result_wins = []
result_win_rate = []

for sim in range(N_SIMULATIONS):

    final_bankrolls, final_wins, final_win_rate = runSimulation(df)

    result_bankroll.append(final_bankrolls)
    result_wins.append(final_wins)
    result_win_rate.append(final_win_rate)

result_bankroll = np.array(result_bankroll)
result_wins = np.array(result_wins)
result_win_rate = np.array(result_win_rate)

agent1_bankroll = result_bankroll[:, 0]
agent1_wins = result_wins[:, 0]
agent1_win_rate = result_win_rate[:, 0]

mean_bankroll = np.mean(agent1_bankroll)
lower_bankroll = np.percentile(agent1_bankroll, 2.5)
upper_bankroll = np.percentile(agent1_bankroll, 97.5)

mean_wins = np.mean(agent1_wins)
lower_wins = np.percentile(agent1_wins, 2.5)
upper_wins = np.percentile(agent1_wins, 97.5)

mean_win_rate = np.mean(agent1_win_rate)
lower_win_rate = np.percentile(agent1_win_rate, 2.5)
upper_win_rate = np.percentile(agent1_win_rate, 97.5)

print("Agent 1:")

print("Mean:", mean_bankroll)
print("95% CI:", lower_bankroll, upper_bankroll)
print("Mean:", mean_wins)
print("95% CI:", lower_wins, upper_wins)
print("Mean:", mean_win_rate)
print("95% CI:", lower_wins, upper_wins)

plt.figure(figsize=(10, 5))
plt.hist(agent1_bankroll, bins=20)
plt.axvline(lower_bankroll, color='red', linestyle='dashed', linewidth=2)
plt.axvline(upper_bankroll, color='red', linestyle='dashed', linewidth=2)
plt.axvline(mean_bankroll, color='green', linestyle='solid', linewidth=2)