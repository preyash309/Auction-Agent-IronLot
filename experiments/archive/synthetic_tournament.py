import os
import sys
import random
import time

try:
    from agent import LiveAuctionAgent
except ImportError:
    print("Error: Make sure this script is in the same folder as agent.py!")
    sys.exit(1)

# A small pool of valid base cars to mutate
CAR_TEMPLATES = [
    {"year": 2013, "make": "ford", "model": "f-150", "trim": "xlt", "body": "supercrew", "transmission": "automatic", "state": "tx", "color": "white", "interior": "black"},
    {"year": 2015, "make": "toyota", "model": "camry", "trim": "se", "body": "sedan", "transmission": "automatic", "state": "ca", "color": "silver", "interior": "black"},
    {"year": 2014, "make": "honda", "model": "accord", "trim": "ex-l", "body": "sedan", "transmission": "automatic", "state": "fl", "color": "black", "interior": "gray"},
    {"year": 2016, "make": "porsche", "model": "911", "trim": "carrera", "body": "coupe", "transmission": "automatic", "state": "ny", "color": "red", "interior": "black"} # High volatility test
]

def generate_random_car():
    car = random.choice(CAR_TEMPLATES).copy()
    car["odometer"] = random.uniform(10000.0, 150000.0)
    car["condition"] = random.choice([2.0, 3.0, 4.0, 5.0])
    # Give the car a hidden "True Market Value" between $8k and $35k
    true_value = random.uniform(8000.0, 35000.0) 
    return car, true_value

def run_tournament():
    print("=== INITIALIZING 100-CAR TOURNAMENT ===")
    agent = LiveAuctionAgent()
    print(f"Agent loaded. Starting Bankroll: ${agent.bankroll:,.2f}\n")
    
    time.sleep(1)
    
    total_cars = 100
    
    for round_num in range(1, total_cars + 1):
        car, true_value = generate_random_car()
        
        # Phase 1: Analysis
        agent.analyze_item(car)
        if agent.predicted_price == 0.0:
            continue # Skipped due to short-circuit guardrail
            
        # Phase 2: Generate Opponents
        # Opponent 1: The Bargain Hunter (Bids 80% of true value)
        # Opponent 2: The Rational Bidder (Bids 95% of true value)
        # Opponent 3: The Wildcard (Randomly bids between 70% and 105% of true value)
        opponents = [
            true_value * 0.80,
            true_value * 0.95,
            true_value * random.uniform(0.70, 1.05)
        ]
        max_opponent_limit = max(opponents)
        
        # Phase 3: The English Auction
        current_bid = true_value * 0.50 # Bidding starts at 50% of car's value
        agent_won = False
        winning_bid = 0.0
        
        while True:
            my_bid = agent.place_bid(current_bid)
            
            if my_bid == 0.0:
                # Agent folded
                winning_bid = current_bid
                break
                
            if my_bid > max_opponent_limit:
                # Opponents hit their limit and folded. Agent wins!
                agent_won = True
                winning_bid = max_opponent_limit + 50.0
                break
                
            # Opponent matches/exceeds
            current_bid = my_bid + 50.0
            
        # Phase 4: Auction Result
        new_bankroll = agent.bankroll
        if agent_won:
            profit = true_value - winning_bid
            new_bankroll += profit
            
        agent.auction_result(won=agent_won, winning_bid=winning_bid, actual_price=true_value, current_bankroll=new_bankroll)
        
        # Print a clean, minimal log
        status = "🟢 WON " if agent_won else "🔴 LOST"
        make_model = f"{agent.car_make.title()} {agent.car_model.title()}"
        print(f"Round {round_num:03d} | {status} | Bankroll: ${agent.bankroll:,.0f} | Win Rate: {agent.win_rate:.1%} | {make_model}")

    print("\n=== FINAL TOURNAMENT RESULTS ===")
    print(f"Starting Bankroll : $500,000")
    print(f"Ending Bankroll   : ${agent.bankroll:,.0f}")
    print(f"Total Net Profit  : ${(agent.bankroll - 500000):,.0f}")
    print(f"Total Cars Won    : {agent.wins} out of {total_cars}")
    print(f"Final Win Rate    : {agent.win_rate:.1%}")

if __name__ == "__main__":
    run_tournament()