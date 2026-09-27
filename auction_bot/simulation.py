"""Seeded English-auction smoke evaluation, adapted from submission tournament."""
import random

import pandas as pd

from .agent import LiveAuctionAgent


def simulate(dataset, checkpoint_dir, rounds=500, seed=42):
    if rounds <= 0:
        raise ValueError("Rounds must be positive")
    frame = pd.read_csv(dataset)
    if "sellingprice" not in frame:
        raise ValueError("Simulation dataset requires sellingprice")
    frame = frame.dropna(subset=["sellingprice"])
    if len(frame) < rounds:
        raise ValueError(f"Requested {rounds} rounds but only {len(frame)} labeled rows available")
    rng = random.Random(seed)
    sample = frame.sample(n=rounds, random_state=seed)
    agent = LiveAuctionAgent(checkpoint_dir)
    skipped = 0
    for _, row in sample.iterrows():
        true_value = float(row["sellingprice"])
        agent.analyze_item(row.drop("sellingprice").to_dict())
        if agent.predicted_price == 0:
            skipped += 1
            continue
        opponents = [true_value*0.80, true_value*0.95, true_value*rng.uniform(0.70,1.05), true_value*0.92, true_value*0.90]
        limit = max(opponents)
        current_bid = true_value*0.50
        won = False
        while True:
            bid = agent.place_bid(current_bid)
            if bid == 0:
                break
            if bid > limit:
                won = True
                break
            current_bid = bid + 50.0
        paid = limit + 50.0 if won else current_bid
        bankroll = agent.bankroll + (true_value-paid if won else 0)
        agent.auction_result(won, paid, true_value, bankroll)
    return {"seed": seed, "sampled": rounds, "skipped": skipped, "auctioned": agent.round_nums, "wins": agent.wins, "win_rate": agent.win_rate, "bankroll": agent.bankroll, "profit": agent.bankroll-500000.0, "protocol": "synthetic opponents based on actual price; immediate resale; data may overlap model training"}
