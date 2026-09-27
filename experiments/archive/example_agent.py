import os
import joblib
import numpy as np
import warnings

class LiveAuctionAgent:
    def __init__(self):
        # Initial bankroll is 00,000
        self.bankroll = 500000.0
        self.predicted_value = 0.0
        
        # Get the path to where this script is saved
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Load the model and the encoders we saved during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(os.path.join(base_path, "model_Example.pkl"))
            self.encoders = joblib.load(os.path.join(base_path, "encoders_Example.pkl"))

    def analyze_item(self, item_features: dict):
        # 1. Define which columns are numbers and which are text
        numeric_cols = ['year', 'condition', 'odometer']
        categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
        
        # 2. Build our feature list starting with the numbers
        features = []
        for col in numeric_cols:
            value = item_features.get(col, 0)
            features.append(float(value))
        
        # 3. Use our encoders to turn text columns into numbers
        for col in categorical_cols:
            text_value = str(item_features.get(col, ""))
            encoder = self.encoders[col]
            
            # If we recognize the word, transform it. Otherwise, use 0.
            if text_value in encoder.classes_:
                encoded_val = int(encoder.transform([text_value])[0])
                features.append(encoded_val)
            else:
                features.append(0)
        
        # 4. Predict the price using the model
        # We wrap it in a list [features] because the model expects a 2D array
        input_data = np.array([features])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self.model.predict(input_data)
            self.predicted_value = float(prediction[0])

    def place_bid(self, current_highest_bid: float) -> float:
        # Strategy: We are willing to pay up to 90% of our predicted value
        my_max_price = self.predicted_value * 0.90
        
        # The next minimum bid allowed is the current highest + 0
        my_next_bid = current_highest_bid + 50.0
        
        # If the next bid is within our limit and we have the money, we bid!
        if my_next_bid <= my_max_price and my_next_bid <= self.bankroll:
            return my_next_bid
        
        # Otherwise, we fold (return 0)
        return 0.0

    def auction_result(self, won, winning_bid, actual_price, current_bankroll):
        # The evaluator tells us our new official bankroll after the round
        self.bankroll = current_bankroll
