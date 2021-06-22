# -*- coding: utf-8 -*-
import sys
import argparse
import joblib
# import xgboost as xgb
import pandas as pd
# import numpy as np

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--today', required=False)
    parser.add_argument('--distance', required=False)
    parser.add_argument('--azimuth', required=False)
    parser.add_argument('--usd', required=False)
    parser.add_argument('--area', required=False)
    parser.add_argument('--bt', required=False)
    parser.add_argument('--roo', required=False)
    parser.add_argument('--ot', required=False)
    parser.add_argument('--lvls', required=False)
    parser.add_argument('--lvl', required=False)
    parser.add_argument('--kitch', required=False)
    parser.add_argument('--reg', required=False)
    namespace = parser.parse_args(sys.argv[1:])
    return namespace
namespace = createParser()

id_region   = int(namespace.reg.split()[0])
area        = float(namespace.area.split()[0])
ot = int(namespace.ot.split()[0])
flat = pd.DataFrame({
    'time':[int(namespace.today.split()[0])],
    'distance':[float(namespace.distance.split()[0])],
    'azimuth':[float(namespace.azimuth.split()[0])],
    'usd':[float(namespace.usd.split()[0])],
    'building_type':[int(namespace.bt.split()[0])],
    'level':[int(namespace.lvl.split()[0])],
    'levels':[int(namespace.lvls.split()[0])],
    'rooms': [int(namespace.roo.split()[0])],
    'area':[area],
    'kitchen_area':[float(namespace.kitch.split()[0])]
})

import numpy as np
def clean_dataset(df):
  assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
  df.dropna(inplace=True)
  indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
  return df[indices_to_keep].astype(np.float32)
flat = clean_dataset(flat)
# print({"random_forest":0, 'xgboost':0})
# exit()
if ot == 1:
  ot = 1
if ot == 2:
  ot = 11
rf_model = joblib.load('/home/daniilak/csv_data/'+str(id_region)+'/rf_model_'+str(ot)+'.h5')
rf_prediction_flat = rf_model.predict(flat).round(0)
random_forest_price1 = rf_prediction_flat*area
del rf_model
del rf_prediction_flat

# rf_model = joblib.load('/home/daniilak/csv_data/'+str(id_region)+'/rf_model2.h5')
# rf_prediction_flat = rf_model.predict(flat).round(0)
# rfp2 = rf_prediction_flat*area
# del rf_model
# del rf_prediction_flat
# rf_model = joblib.load('/home/daniilak/csv_data/'+str(id_region)+'/rf_model3.h5')
# rf_prediction_flat = rf_model.predict(flat).round(0)
# rfp3 = rf_prediction_flat*area
# del rf_model
# del rf_prediction_flat

xgb_model = joblib.load('/home/daniilak/csv_data/'+str(id_region)+'/xgb_model_'+str(ot)+'.h5')
xgb_prediction_flat = xgb_model.predict(flat).round(0)
xgb_price = xgb_prediction_flat*area
del xgb_model
del xgb_prediction_flat

# cat_model = joblib.load('/home/daniilak/csv_data/'+str(id_region)+'/cat_model.h5')
# cat_prediction_flat = cat_model.predict(flat).round(0)
# cat_price = cat_prediction_flat*area
# del cat_model
# del cat_prediction_flat
# "rf2":int(rfp2.round(-3)), "rf3":int(rfp3.round(-3)),
# print({"random_forest":int(random_forest_price1.round(-3)), 'xgboost':int(xgb_price.round(-3))})
# exit()
import random

random_forest = int(random_forest_price1.round(-3))
xgb_price = int(xgb_price.round(-3))

print({"random_forest":random_forest, 'xgboost':xgb_price})
exit()


