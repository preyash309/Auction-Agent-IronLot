import os
import sys
import random
import time
import pandas as pd

try:
    from agent import LiveAuctionAgent
except ImportError:
    print("Error: Make sure this script is in the same folder as agent.py!")
    sys.exit(1)

def run_realistic_tournament():
    print("=== INITIALIZING 500-CAR REALISTIC TOURNAMENT ===")
    
    # 1. Load the actual raw dataset
    try:
        path = os.getcwd()
        df = pd.read_csv(f"{path}/new/car_auction_train.csv") # Adjust path if necessary
        print(f"Dataset loaded successfully: {len(df)} total rows found.")
    except FileNotFoundError:
        print("Error: 'car_auction_train.csv' not found in the current directory.")
        return
        
    # 2. Sample 500 random cars for the hackathon simulation
    sample_df = df.sample(n=500, replace=False).reset_index(drop=True)
    
    agent = LiveAuctionAgent()
    print(f"Agent loaded. Starting Bankroll: ${agent.bankroll:,.2f}\n")
    time.sleep(1)
    
    total_cars = 500
    skipped_cars = 0
    
    for round_num in range(1, total_cars + 1):
        row = sample_df.iloc[round_num - 1]
        
        # Extract the true price for our opponent simulator
        # If true price is somehow NaN, skip the row for the sake of the test
        if pd.isna(row['sellingprice']):
            continue
        true_value = float(row['sellingprice'])
        
        # Create the dictionary exactly as the API will send it
        # We drop 'sellingprice' so the agent doesn't accidentally cheat
        item_features = row.drop('sellingprice').to_dict()
        
        # Phase 1: Analysis (This tests your preprocessing!)
        try:
            agent.analyze_item(item_features)
        except Exception as e:
            print(f"Round {round_num:03d} | ❌ CRASHED DURING ANALYSIS | Error: {e}")
            print(f"Faulty Data: {item_features}")
            break
            
        # Check if the short-circuit guardrail triggered
        if agent.predicted_price == 0.0:
            skipped_cars += 1
            print(f"Round {round_num:03d} | ⚠️ SKIPPED | Missing Make/Model. Guardrail activated.")
            continue
            
        # Phase 2: Generate Opponents
        # Using the same spread: Bargain Hunter, Rational, and Wildcard
        opponents = [
            true_value * 0.80,
            true_value * 0.95,
            true_value * random.uniform(0.70, 1.05),
            true_value * 0.92,
            true_value * 0.90,
        ]
        max_opponent_limit = max(opponents)
        
        # Phase 3: The English Auction
        current_bid = true_value * 0.50 # Bidding starts at 50%
        agent_won = False
        winning_bid = 0.0
        
        while True:
            try:
                my_bid = agent.place_bid(current_bid)
            except Exception as e:
                print(f"Round {round_num:03d} | ❌ CRASHED DURING BIDDING | Error: {e}")
                break
                
            if my_bid == 0.0:
                winning_bid = current_bid
                break
                
            if my_bid > max_opponent_limit:
                agent_won = True
                winning_bid = max_opponent_limit + 50.0
                break
                
            current_bid = my_bid + 50.0
            
        # Phase 4: Auction Result
        new_bankroll = agent.bankroll
        if agent_won:
            profit = true_value - winning_bid
            new_bankroll += profit
            
        agent.auction_result(won=agent_won, winning_bid=winning_bid, actual_price=true_value, current_bankroll=new_bankroll)
        
        # Print a clean log
        status = "🟢 WON " if agent_won else "🔴 LOST"
        make_model = f"{agent.car_make.title()} {agent.car_model.title()}"
        print(f"Round {round_num:03d} | {status} | Bankroll: ${agent.bankroll:,.0f} | Win Rate: {agent.win_rate:.1%} | True Value: ${true_value:,.0f} | {make_model}")

    print("\n=== FINAL TOURNAMENT RESULTS ===")
    print(f"Starting Bankroll : $500,000")
    print(f"Ending Bankroll   : ${agent.bankroll:,.0f}")
    print(f"Total Net Profit  : ${(agent.bankroll - 500000):,.0f}")
    print(f"Total Cars Won    : {agent.wins} out of {total_cars - skipped_cars} bid on")
    print(f"Cars Skipped (NaN): {skipped_cars}")
    print(f"Final Win Rate    : {agent.win_rate:.1%}")

if __name__ == "__main__":
    run_realistic_tournament()