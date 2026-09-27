import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
import optuna
import os
import pickle

path = os.getcwd()
df = pd.read_csv(f"{path}/new/car_auction_train.csv")

train, test = train_test_split(df, test_size=0.2)

def dropRows(df):
  df = df[(df['make'].notna())]
  df = df[(df['model'].notna())]
  return df

def modifyNames(df):
  df['make'] = df['make'].str.lower().str.strip()

  make_cleanup = {
    'chev truck': 'chevrolet',
    'gmc truck': 'gmc',
    'ford truck': 'ford',
    'dodge tk': 'dodge',
    'hyundai tk': 'hyundai',
    'mercedes-b': 'mercedes-benz',
    'mercedes': 'mercedes-benz',
    'landrover': 'land rover',
    'vw': 'volkswagen'
  }
  df['make'] = df['make'].replace(make_cleanup)

  df['make'] = df['make'].str.title()

  df['model'] = df['model'].str.lower().str.replace("-","").str.replace("/","")
  df['model'] = df['model'].str.title()

  return df

def imputeUnknown(df):
  df['trim'] = df['trim'].fillna(df.groupby('make')['trim'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'))
  df['body'] = df['body'].fillna(df.groupby('make')['body'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'))

  df['transmission'] = df['transmission'].fillna("Unknown")
  df['color'] = df['color'].fillna("Unknown")
  df['interior'] = df['interior'].fillna("Unknown")

  return df

def fillCondition(row):
  if row['odometer'] <= 30000:
    return 4.0
  elif (row['odometer'] > 30000) & (row['odometer'] <= 100000):
    return 3.0
  elif (row['odometer'] > 100000) & (row['odometer'] <= 200000):
    return 2.0
  else:
    return 1.0

def imputeNumeric(df):
  df.loc[df['condition'].isna(),"condition"] = df.loc[df['condition'].isna()].apply(fillCondition, axis=1)

  odometer_median = df['odometer'].median()
  df['odometer'] = df['odometer'].fillna(odometer_median) 
  return df

def dataPreprocessing(df):
  df = dropRows(df)
  df = modifyNames(df)
  df = imputeUnknown(df)
  df = imputeNumeric(df)

  return df

def featureEngineering(df):
  df['years_used'] = df['year'].apply(lambda x: 2016 - x)
  df["miles_per_year"] = df['odometer']/df['years_used']

  luxury_brands = ["Rolls-Royce","Ferrari","Lamborghini","Bentley","Porsche","Aston Martin"]
  df["luxury_brands"] = df['make'].apply(lambda x: 1 if x in luxury_brands else 0)
  df['is_rare'] = (df.groupby('model')['model'].transform('size') < 50).astype(int)
  df['depreciation'] = df['years_used']*(df['condition']/5) 
  return df

train = dataPreprocessing(train)
train = featureEngineering(train)

test = dataPreprocessing(test)
test = featureEngineering(test)

train_target = train['sellingprice']
train = train.drop(columns=['sellingprice'])

test_target = test['sellingprice']
test = test.drop(columns=['sellingprice'])

numeric_cols = train.select_dtypes(include=['int64','float64']).columns.to_list()
one_hot_cols = ['transmission']
target_cols = train.select_dtypes(include=['object']).columns.to_list()
target_cols = [obj for obj in target_cols if obj != 'transmission']

preprocessor = ColumnTransformer(transformers=[
    ("numeric","passthrough",numeric_cols),
    ("one_hot",OneHotEncoder(handle_unknown='ignore'),one_hot_cols),
    ("target",TargetEncoder(target_type='continuous'),target_cols)
])

# def objective(trial):
#   n_estimators = trial.suggest_int("n_estimators",100,3000,step=100)
#   max_depth = trial.suggest_int("max_depth",2,10)
#   num_leaves = trial.suggest_int("num_leaves",2,min(2**max_depth,256))
#   learning_rate = trial.suggest_float("learning_rate",1e-3,1e-1,log=True)
#   min_child_samples = trial.suggest_int("min_child_samples",10,100,step=10)

#   model = LGBMRegressor(n_estimators=n_estimators,max_depth=max_depth, num_leaves=num_leaves,
#                         learning_rate=learning_rate,min_child_samples=min_child_samples,n_jobs=-1,verbose=-1)
  
#   pipeline = Pipeline(steps=[
#       ("preprocessor",preprocessor),
#       ("model",model)
#   ])

#   score = cross_val_score(pipeline,train,train_target,cv=5,scoring="neg_root_mean_squared_error",n_jobs=1)
#   rmse = -score.mean()
#   return rmse

# study = optuna.create_study(direction="minimize")
# study.optimize(objective,n_trials=20)

# print(study.best_params)

model = LGBMRegressor(n_estimators=2400,max_depth=10,num_leaves=176,learning_rate=0.03038452742864471,min_child_samples=10,n_jobs=-1,verbose=-1)

pipeline = Pipeline(steps=[
  ("preprocessor",preprocessor),
  ("model",model)
])

pipeline.fit(train,train_target)
y_pred = pipeline.predict(test)

r2 = r2_score(test_target,y_pred)
print(r2)

rmse = np.sqrt(mean_squared_error(test_target,y_pred))
print(rmse)

mae = mean_absolute_error(test_target,y_pred)
print(mae)

final = pd.concat([train,test]).reset_index(drop=True)
final_target = pd.concat([train_target,test_target]).reset_index(drop=True)

trim_map = final.groupby('make')['trim'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown").to_dict()
body_map = final.groupby('make')['body'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown").to_dict()

cleanup_names = {
    'chev truck': 'chevrolet',
    'gmc truck': 'gmc',
    'ford truck': 'ford',
    'dodge tk': 'dodge',
    'hyundai tk': 'hyundai',
    'mercedes-b': 'mercedes-benz',
    'mercedes': 'mercedes-benz',
    'landrover': 'land rover',
    'vw': 'volkswagen'
}
median = final['odometer'].median()

others = {
    "cleanup_names":cleanup_names,
    "trim_map":trim_map,
    "body_map":body_map,
    "median":median
}

quantile = LGBMRegressor(objective="quantile",alpha=0.1,metric="quantile",n_jobs=-1,verbose=-1,random_state=42)

quantile_pipeline = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("quantile",quantile)
])

quantile_pipeline.fit(train,train_target)
# pred = quantile.predict(test)

with open("model.pkl","wb") as f:
  pickle.dump(pipeline,f)

with open("others.pkl","wb") as f:
  pickle.dump(others,f)

with open("quantile.pkl","wb") as f:
  pickle.dump(quantile_pipeline,f)

