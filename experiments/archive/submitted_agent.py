import os
import pickle
import numpy as np
import pandas as pd
import warnings

def fillCondition(row):
  if row["odometer"] <= 30000.0:
    return 4.0
  elif row["odometer"] <= 100000.0:
    return 3.0
  elif row["odometer"] <= 200000.0:
    return 2.0
  else:
    return 1.0

class Preprocessing:

  def __init__(self, df, cleanup_names, trim_map, body_map, odometer_median):
    self.df = df
    self.cleanup_names = cleanup_names
    self.trim_map = trim_map
    self.body_map = body_map
    self.odometer_median = odometer_median

  def handleEmpty(self):
    self.df['make'] = self.df['make'].fillna('0')
    self.df['model'] = self.df['model'].fillna('0')
    return self.df

  def modifyNames(self):
    self.df['make'] = self.df['make'].astype(str).str.lower().str.strip()
    self.df['make'] = self.df['make'].replace(self.cleanup_names)
    self.df['make'] = self.df['make'].str.title()

    self.df['model'] = self.df['model'].astype(str).str.lower().str.replace("-","").str.replace("/","")
    self.df['model'] = self.df['model'].str.title()

    return self.df


  def imputeUnknown(self):
    self.df["trim"] = self.df["trim"].fillna(self.df["make"].map(self.trim_map)).fillna("Unknown")
    self.df["body"] = self.df["body"].fillna(self.df["make"].map(self.body_map)).fillna("Unknown")

    self.df["transmission"] = self.df["transmission"].fillna("Unknown")
    self.df["color"] = self.df["color"].fillna("Unknown")
    self.df["interior"] = self.df["interior"].fillna("Unknown")

    return self.df

  def imputeNumeric(self):
    self.df["odometer"] = self.df["odometer"].fillna(self.odometer_median)

    self.df.loc[self.df["condition"].isna(),"condition"] = self.df.loc[self.df["condition"].isna()].apply(fillCondition, axis=1)

    return self.df

  def handleEdgeCases(self):
    self.df["year"] = self.df["year"].fillna(2010)
    self.df["state"] = self.df["state"].fillna("va")

    return self.df

  def applyPreprocessing(self):
    self.df = self.handleEmpty()
    self.df = self.modifyNames()
    self.df = self.imputeUnknown()
    self.df = self.imputeNumeric()
    self.df = self.handleEdgeCases()

    return self.df

def featureEngineering(df):
  df["years_used"] = df['year'].apply(lambda x: 2016 - x)
  df["miles_per_year"] = df["odometer"]/df["years_used"]

  luxury_brands = ["Rolls-Royce","Ferrari","Lamborghini","Bentley","Porsche","Aston Martin"]
  rare_brands = ["Rolls-Royce","Lamborghini"]
  df["luxury_brands"] = df["make"].apply(lambda x: 1 if x in luxury_brands else 0)
  df["is_rare"] = df["make"].apply(lambda x: 1 if x in rare_brands else 0)
  df["depreciation"] = df["years_used"]*(df["condition"]/5)

  return df

def finalizeData(df,cleanup_names,trim_map,body_map,odometer_median):
  pre = Preprocessing(df,cleanup_names,trim_map,body_map,odometer_median)
  df = pre.applyPreprocessing()
  df = featureEngineering(df)

  return df

class LiveAuctionAgent:

  def __init__(self):

    self.bankroll = 500000.0
    self.predicted_price = 0.0
    self.p10_price = 0.0
    self.win_rate = 0
    self.wins = 0
    self.round_nums = 0
    self.car_make = ""
    self.car_model = ""

    base_path = os.path.dirname(os.path.abspath(__file__))

    with warnings.catch_warnings():
      with open(os.path.join(base_path,"model_PreyashPratyush.pkl"),"rb") as f:
        self.model = pickle.load(f)

      with open(os.path.join(base_path,"quantile_PreyashPratyush.pkl"),"rb") as f:
        self.quantile = pickle.load(f)

      with open(os.path.join(base_path,"others_PreyashPratyush.pkl"),"rb") as f:
        self.others = pickle.load(f)

  def analyze_item(self, item_features: dict):
    car = pd.DataFrame([item_features])
    car = finalizeData(car,self.others["cleanup_names"],self.others["trim_map"],self.others["body_map"],self.others["median"])
    self.car_make = str(car["make"].iloc[0])
    self.car_model = str(car["model"].iloc[0])

    if self.car_make == '0' or self.car_model == '0':
      self.predicted_price = 0.0
      self.p10_price = 0.0

    else:
      with warnings.catch_warnings():
        prediction_price = self.model.predict(car)
        quantile_price = self.quantile.predict(car)
        self.predicted_price = float(prediction_price[0])
        self.p10_price = float(quantile_price[0])

  def place_bid(self, current_highest_bid: float) -> float:

    if self.car_make == "0" or self.car_model == "0":
      return 0.0

    ALPHA = 0.1007
    BETA = 0.3792
    GAMMA = 0.0125

    price = self.predicted_price
    p10 = self.p10_price
    deviation = price - p10

    win = max(0.0, 0.2 - self.win_rate)
    base = price - (ALPHA * deviation)
    penalty = BETA * max(0,(500000-self.bankroll))/500000 * price
    boost = GAMMA * win * price

    my_max_bid = base - penalty + boost

    my_max_bid = min(my_max_bid,self.bankroll)

    my_next_bid = current_highest_bid + 50.0

    if my_next_bid <= my_max_bid and my_next_bid <= self.bankroll:
      return my_next_bid
    else:
      return 0.0 

  def auction_result(self, won, winning_bid, actual_price, current_bankroll):
    
    self.bankroll = current_bankroll

    if won:
      self.wins += 1

    self.round_nums += 1
    self.win_rate = self.wins/(self.round_nums)