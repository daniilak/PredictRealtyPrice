#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import argparse
import json
import pickle as cPickle
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import optuna
import ast
import joblib

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-r', '--region', default=0)
    parser.add_argument('-o', '--ot', default=0)
    return parser
parser = createParser()
namespace = parser.parse_args(sys.argv[1:])
region = str(namespace.region.split()[0])
ot = int(namespace.ot.split()[0])
df = pd.read_csv( '~/csv_data/'+str(region)+'/export_dataframe_'+str(ot)+'.csv', sep=",", encoding = 'utf8')
def clean_dataset(df):
  assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
  df.dropna(inplace=True)
  indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
  return df[indices_to_keep].astype(np.float32)
df = clean_dataset(df)
y = df['priceMetr']
features = ['time','distance','azimuth','usd','building_type','level','levels','rooms','area','kitchen_area']
X = df[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=1)


def objective(trial):
    print('')
    params = {
      'n_estimators': int(trial.suggest_loguniform('n_estimators', 500, 2000)),
      'bootstrap':False,
      'max_features':'sqrt',
      'random_state':1,
      'max_depth' : trial.suggest_int('max_depth', 5, 300),
      'min_samples_leaf':trial.suggest_int("min_samples_leaf", 1, 15),
      'min_samples_split':0.05,
      'n_jobs':-1,
      }
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_valid_pred = model.predict(X_valid)
    return mean_absolute_error(y_valid, y_valid_pred)


study = optuna.create_study()
study.optimize(objective, n_trials=150)

print(study.best_trial)

with open('/home/daniilak/csv_data/'+str(region)+'/best_params_'+str(ot)+'.txt', 'w') as f:
  f.write(json.dumps(study.best_params) + '\n')
  f.close()

print("Gen Model")
f = open('/home/daniilak/csv_data/'+str(region)+'/best_params_'+str(ot)+'.txt', 'r')
params = ast.literal_eval(str(f.read()))


rf_model = RandomForestRegressor(
      n_estimators= int(params['n_estimators']),
      max_depth= int(params['max_depth']),
      min_samples_leaf=int(params['min_samples_leaf']),
      bootstrap=False,
      max_features='sqrt',
      criterion='mse',
      random_state=1,
      min_samples_split=0.05,
      n_jobs=-1,
)

rf_model.fit(train_X, train_y)
joblib.dump(rf_model, '/home/daniilak/csv_data/'+str(region)+'/rf_model_'+str(ot)+'.h5',protocol=4)


# from sklearn.metrics import mean_squared_error
 
# def objective2(trial):
#     params = {
#         "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
#         "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
#         "colsample_bytree": trial.suggest_categorical(
#             "colsample_bytree", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#         ),
#         "subsample": trial.suggest_categorical(
#             "subsample", [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
#         ),
#         "learning_rate": trial.suggest_categorical(
#             "learning_rate", [0.008, 0.009, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02]
#         ),
#         "n_estimators": int(trial.suggest_loguniform('n_estimators', 500, 2000)),
#         "max_depth": trial.suggest_categorical(
#             "max_depth", [5, 7, 9, 11, 13, 15, 17, 20]
#         ),
#         "random_state": 1,
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 300),
#     }
#     model = XGBRegressor(**params)
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
#     model.fit(X_train, y_train)
#     y_valid_pred = model.predict(X_valid)
#     return mean_absolute_error(y_valid, y_valid_pred)

# study = optuna.create_study(direction="minimize")
# study.optimize(objective2, n_trials=50)
# print(study.best_trial)

# with open('/home/daniilak/csv_data/'+str(region)+'/best_params2_'+str(ot)+'.txt', 'w') as f:
#   f.write(json.dumps(study.best_params) + '\n')
#   f.close()

# # # # # # # # # # # # # # # # # # # # # # # # # 
# print("Gen Model XGB")
# f = open('/home/daniilak/csv_data/2/best_params2_'+str(ot)+'.txt', 'r')
# # f = open('/home/daniilak/csv_data/'+str(region)+'/best_params2_'+str(ot)+'.txt', 'r')
# params = ast.literal_eval(str(f.read()))


# paramsModel = {
#          "lambda": float(params['lambda']),
#          "alpha": float(params['alpha']),
#          "colsample_bytree": float(params['colsample_bytree']),
#          "subsample": float(params['subsample']),
#          "learning_rate": float(params['learning_rate']),
#          "learning_rate": float(params['learning_rate']),
#          "n_estimators": int(params['n_estimators']),
#          "max_depth": int(params['max_depth']),
#          "random_state": 1,
#          "n_jobs":-1,
#          "min_child_weight": int(params['min_child_weight']),
#      }
# xgb_model = XGBRegressor(**paramsModel)
# xgb_model.fit(train_X, train_y)
# joblib.dump(xgb_model, '/home/daniilak/csv_data/'+str(region)+'/xgb_model_'+str(ot)+'.h5',protocol=4)


# from catboost import CatBoostRegressor

# model_CBR = CatBoostRegressor()
# parameters = {
#   'depth'         : [i for i in  range(5, 16, 1)],
#   'learning_rate' : [0.01, 0.05, 0.1],
#   'iterations'    : [i for i in  range(50, 2000, 50)]
# }
# grid = GridSearchCV(estimator=model_CBR, param_grid = parameters, cv = 2, n_jobs=-1)
# grid.fit(train_X, train_y)
# print(" Results from Grid Search " )
# print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)
# print("\n The best score across ALL searched params:\n", grid.best_score_)
# print("\n The best parameters across ALL searched params:\n", grid.best_params_)
# exit()
# def objective3(trial):
    
#     train_X,test_X,train_y,test_y = train_test_split(X, y, test_size = 0.25,random_state = 1)

#     params = {
#       "n_estimators" : trial.suggest_int('n_estimators', 0, 2000),
#       "learning_rate": trial.suggest_loguniform('learning_rate',0.01,0.5),
#       "depth": trial.suggest_int('depth', 6, 20),
#       "l2_leaf_reg": trial.suggest_int('l2_leaf_reg', 2, 30),
#       "bagging_temperature": trial.suggest_discrete_uniform('bagging_temperature',0.1,1,0.01),
#       "n_jobs":-1
#     }
#     model = CatBoostRegressor(**params, verbose=False)
#     model.fit(train_X, train_y)
#     y_valid_pred = model.predict(test_X)
#     return mean_absolute_error(test_y, y_valid_pred)

# # study = optuna.create_study(direction="minimize")
# # study.optimize(objective2, n_trials=200)
# # print(study.best_trial)

# with open('/home/daniilak/csv_data/'+str(region)+'/best_params3.txt', 'w') as f:
#   f.write(json.dumps(study.best_params) + '\n')
#   f.close()

# print("Gen Model Cat")
# f = open('/home/daniilak/csv_data/'+str(region)+'/best_params3.txt', 'r')
# params = ast.literal_eval(str(f.read()))

# paramsModel = {
#       "learning_rate": params['learning_rate'],
#       "iterations": int(params['iterations']),
#       "depth": int(params['depth']),
# }

# cat_model = CatBoostRegressor(**paramsModel, verbose=False)
# cat_model.fit(train_X, train_y)
# joblib.dump(cat_model, '/home/daniilak/csv_data/'+str(region)+'/cat_model.h5',protocol=4)



