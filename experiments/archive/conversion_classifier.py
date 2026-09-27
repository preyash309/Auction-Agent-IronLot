import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import optuna
import os

path = os.getcwd()


df1 = pd.read_csv(f"{path}/new/df/training (1).csv")
df2 = pd.read_csv(f"{path}/new/df/testing.csv")

df1 = df1.drop(columns=["Unnamed: 0"])
df2 = df2.drop(columns=["Unnamed: 0"])

train = df1.drop(columns=["User_ID","Converted"])
target = df1["Converted"]

test = df2.drop(columns=["User_ID","Converted"])
test_target = df2["Converted"]

num_features = train.select_dtypes(include=[np.number]).columns.to_list()
cat_features = train.select_dtypes(include=["object","category"]).columns.to_list()

preprocessor = ColumnTransformer(transformers=[
    ("encoder",TargetEncoder(target_type="continuous"),cat_features),
    ("numeric","passthrough",num_features)
])

def objective(trial):
  n_estimators = trial.suggest_int("n_estimators",100,3000,step=10)
  max_depth = trial.suggest_int("max_depth",2,10)
  num_leaves = trial.suggest_int("num_leaves",2,min(2**max_depth,256))
  min_child_samples = trial.suggest_int("min_child_samples",10,100,step=5)
  learning_rate = trial.suggest_float("learning_rate",1e-5,1e-1,log=True)
  threshold = trial.suggest_float("threshold",0.2,0.8)

  model = LGBMClassifier(n_estimators=n_estimators, max_depth = max_depth, num_leaves=num_leaves,
                         min_child_samples=min_child_samples, learning_rate=learning_rate, is_unbalance=True,
                         objective="binary",metric="binary_error",verbosity=-1)
  
  pipeline = Pipeline(steps=[
      ("preprocessor",preprocessor),
      ("model",model)
  ])

  def f1scoring(pipeline, train, target):
    probs = pipeline.predict_proba(train)[:,1]
    preds = (probs > threshold).astype(int)
    return f1_score(target,preds)

  
  score = cross_val_score(pipeline, train, target, cv=5, scoring=f1scoring,n_jobs=-1).mean()
  return score

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=50)


print(study.best_params)