"""Checkpoint-compatible submission preprocessing; historical formulas retained."""
import pandas as pd

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

