#!/usr/bin/python3
# -*- coding: utf-8 -*-
import psycopg2 as pg
import ast 
import time
from datetime import datetime
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas.io.sql as psql
import sys
import argparse

DB_USER="daniilak"
DB_NAME="daniilak"
DB_PASS="410552"
DB_HOST="localhost"
def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--region', default=0)
    parser.add_argument('-o', '--ot', default=0)
    # parser.add_argument('-y', '--year', default=0)
    namespace = parser.parse_args(sys.argv[1:])
    return namespace
namespace = createParser()
region = int(namespace.region.split()[0])
ot = int(namespace.ot.split()[0])
# year = int(namespace.year.split()[0])

import csv
from datetime import datetime, timedelta, date
import time
import pandas_datareader.data as web

from geopy.distance import geodesic 
import math

start = datetime(2018, 2, 2)
end = datetime.today() + timedelta(days=1)
aud = web.DataReader('RUB=X', 'yahoo', start, end)
# print(aud['Open']['2020-01-31'])
usd = {}
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)
for single_date in daterange(start, end):
    s_date = single_date.strftime("%Y-%m-%d")
    if s_date in aud.index:
        usd[s_date] = aud['Open'][s_date]
        LAST = s_date
        continue
    else:
        usd[s_date] = usd[LAST]
        LAST = s_date

def getTime(s, s1):
    s += " " + str(s1)
    return int(( int(time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())) - 1514754000)/60)

def getUSD(s):
    return usd[s]

# 0 - Другое. 1 - Панельный. 2 - Монолитный. 3 - Кирпичный. 4 - Блочный. 5 - Деревянный
def getBT(s):
    if s == 850:
        return 0
    if s == 2:
        return 1
    if s == 12:
        return 2
    if s == 25 or s == 49343:
        return 3
    if s == 96:
        return 4
    if s == 2220:
        return 5 

# 1 - Вторичка; 2 - Новостройка;
def getOT(s):
    if s == 1:
        return 1
    if s == 11:
        return 2

regions_coordinates = {
    0: (55.7522, 37.6156),
    2: (54.7347, 55.9578),
    5: (42.9830, 47.5046),
    12: (56.6344, 47.8998),
    13: (54.1838, 45.1749),
    16: (55.7985,49.1063),
    18: (56.8527, 53.2114),
    19: (53.7223, 91.4436),
    21: (56.1438, 47.2489),
    22: (53.3479, 83.7798),
    23: (45.0401, 38.9759),
    24: (56.0093, 92.8524),
    31: (50.5976, 36.5856),
    36: (51.6593, 39.1969),
    39: (54.7074, 20.5073),
    40: (54.5059, 36.2516),
    50: (55.7522, 37.6156),
    52: (56.3240, 44.0053),
    54: (55.0281, 82.9211),
    66: (56.8385, 60.6054),
    69: (56.8586, 35.9116),
    70: (56.4845, 84.9481),
    72: (57.1529, 65.5344),
    74: (55.1602, 61.4008),
    76: (57.6215, 39.8977),
    77: (55.7522, 37.6156),
    78: (59.9391, 30.3159)
}

def get_azimuth(latitude, longitude):
 
    rad = 6372795

    llat1 = city_center_coordinates[0]
    llong1 = city_center_coordinates[1]
    llat2 = latitude
    llong2 = longitude

    lat1 = llat1*math.pi/180.
    lat2 = llat2*math.pi/180.
    long1 = llong1*math.pi/180.
    long2 = llong2*math.pi/180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2*sdelta,2)+math.pow(cl1*sl2-sl1*cl2*cdelta,2))
    x = sl1*sl2+cl1*cl2*cdelta
    ad = math.atan2(y,x)

    x = (cl1*sl2) - (sl1*cl2*cdelta)
    y = sdelta*cl2
    z = math.degrees(math.atan(-y/x))

    if (x < 0):
        z = z+180.

    z2 = (z+180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2*math.pi)*math.floor((z2/(2*math.pi))) )
    angledeg = (anglerad2*180.)/math.pi
    
    return round(angledeg, 2)

city_center_coordinates = regions_coordinates[int(region)]


connection = pg.connect("host="+DB_HOST+" dbname="+DB_USER+" user="+DB_NAME+" password="+DB_PASS)

train = psql.read_sql(
    # ' SELECT  adv.price, adv.date, adv.time, ad.geo_lat, ad.geo_lon, h.building_type, h.level, h.levels, h.rooms, h.area, h.kitchen_area' + 
    # ' from adverts_2018 as adv ' + 
    # ' inner join addresses_2 AS ad ON ad.id = adv.id_new  '+
    # ' INNER JOIN houses AS h ON h.id = adv.id_house_info ' +
    # ' where ad.isset IS NOT NULL AND ad.geo_lat IS NOT NULL AND ad.geo_lon IS NOT NULL AND ad.fias_id IS NOT NULL  ' +
    # ' and ad.id_region = '+str(region) +
    # ' and h.object_type = ' + str(ot) +
    # ' and rooms != -100  and h.area != -100 and object_type != -100 and building_type != -100 and kitchen_area != -100 and level != -100 and levels != -100  ' +
    # ' and rooms != 0 and h.area > 0 and kitchen_area > 0  and level != 0 and levels != 0 '

    # ' UNION ALL '

    # ' SELECT  adv.price, adv.date, adv.time, ad.geo_lat, ad.geo_lon,  h.building_type, h.level, h.levels, h.rooms, h.area, h.kitchen_area' + 
    # ' from adverts_2019 as adv ' + 
    # ' inner join addresses_2 AS ad ON ad.id = adv.id_new  '+
    # ' INNER JOIN houses AS h ON h.id = adv.id_house_info ' +
    # ' where ad.isset IS NOT NULL AND ad.geo_lat IS NOT NULL AND ad.geo_lon IS NOT NULL AND ad.fias_id IS NOT NULL  ' +
    # ' and ad.id_region = '+str(region) +
    # ' and h.object_type = ' + str(ot) +
    # ' and rooms != -100 and h.area != -100 and object_type != -100 and building_type != -100 and kitchen_area != -100 and level != -100 and levels != -100  ' +
    # ' and rooms != 0 and h.area > 0 and kitchen_area > 0  and level != 0 and levels != 0 '
    
    # ' UNION ALL '

    ' SELECT  adv.price, adv.date, adv.time,  ad.geo_lat, ad.geo_lon,  h.building_type, h.level, h.levels, h.rooms, h.area, h.kitchen_area' + 
    ' from adverts_2020 as adv ' + 
    ' inner join addresses_2 AS ad ON ad.id = adv.id_new  '+
    ' INNER JOIN houses AS h ON h.id = adv.id_house_info ' +
    ' where ad.isset IS NOT NULL AND ad.geo_lat IS NOT NULL AND ad.geo_lon IS NOT NULL AND ad.fias_id IS NOT NULL  ' +
    ' and ad.id_region = '+str(region) +
    ' and h.object_type = ' + str(ot) +
    ' and rooms != -100 and h.area != -100 and object_type != -100 and building_type != -100 and kitchen_area != -100 and level != -100 and levels != -100  ' +
    ' and rooms != 0 and h.area > 0 and kitchen_area > 0  and level != 0 and levels != 0 '
    
    ' UNION ALL '

    ' SELECT  adv.price, adv.date, adv.time, ad.geo_lat, ad.geo_lon, h.building_type, h.level, h.levels, h.rooms, h.area, h.kitchen_area' + 
    ' from adverts_2021 as adv ' + 
    ' inner join addresses_2 AS ad ON ad.id = adv.id_new  '+
    ' INNER JOIN houses AS h ON h.id = adv.id_house_info ' +
    ' where ad.isset IS NOT NULL AND ad.geo_lat IS NOT NULL AND ad.geo_lon IS NOT NULL AND ad.fias_id IS NOT NULL  ' +
    ' and ad.id_region = '+str(region) +
    ' and h.object_type = ' + str(ot) +
    ' and rooms != -100 and h.area != -100 and object_type != -100 and building_type != -100 and kitchen_area != -100 and level != -100 and levels != -100  ' +
    ' and rooms != 0 and h.area > 0 and kitchen_area > 0  and level != 0 and levels != 0 '
    ' order by date, time', connection)

def checkPrice(s):
    try:
        return int(s)
    except:
        print(s)
        return 0
# train['price'] = list(map(lambda x: checkPrice(x), train['price']))
# exit()

# train["id_region"] = train["id_region"].astype(np.int32)

train['date'] = train['date'].astype(str)
train['time'] = train['time'].astype(str)
train['time'] = list(map(lambda x, y: getTime(x, y), train['date'], train['time']))
train['time'] = train['time'].astype(np.int32).round(0)

train['usd'] = list(map(lambda x: getUSD(x), train['date']))
train['usd'] = train['usd'].astype(np.float32).round(2)

train.drop('date', inplace=True, axis=1)

# train['object_type'] = list(map(lambda x: getOT(x), train['object_type']))
# train["object_type"] = train["object_type"].astype(np.int32)

train['building_type'] = list(map(lambda x: getBT(x), train['building_type']))
train["building_type"] = train["building_type"].astype(np.int32)

train['level'] = train['level'].astype(np.int32)

train['levels'] = train['levels'].astype(np.int32)

train['rooms'] = train['rooms'].astype(np.int32)

train['price'] = train['price'].astype(np.int32)


train['area'] = train['area'].astype(np.float32).round(2)
train['kitchen_area'] = train['kitchen_area'].astype(np.float32).round(2)

train['priceMetr'] = train['price']/train['area']
train['priceMetr'] = train['priceMetr'].astype(np.float32).round(2)

train.drop('price', inplace=True, axis=1)

train['distance'] = list(map(lambda x, y: geodesic(city_center_coordinates, (x, y)).meters, train['geo_lat'], train['geo_lon']))
train['distance'] = train['distance'].astype(np.int32).round()
train['azimuth'] = list(map(lambda x, y: get_azimuth(x, y), train['geo_lat'], train['geo_lon']))
train['azimuth'] = train['azimuth'].astype(np.int32).round()
train.drop('geo_lat', inplace=True, axis=1)
train.drop('geo_lon', inplace=True, axis=1)


train['priceMetr'] = train['priceMetr'].astype(np.float32).round(2)


train.replace([np.inf, -np.inf], np.nan)


train = train.drop(train[train.area>1000].index)

train = train.drop(train[train.area<1].index)
train = train.drop(train[train.kitchen_area<1].index)
train = train.drop(train[train.level<=0].index)
train = train.drop(train[train.levels<=0].index)


train = train.drop(train[train.priceMetr<=0].index)
train = train.drop(train[train.priceMetr>500000].index)
train = train.drop(train[train.priceMetr<10].index)
train = train.drop(train[train.area<train.kitchen_area].index)
train = train.drop(train[train.levels<train.level].index)


train = train.drop(train[train.kitchen_area>150].index)
train = train.drop(train[train.level>40].index)
train = train.drop(train[train.levels>40].index)
train = train.drop(train[train.rooms>6].index)
train = train.drop(train[train.rooms>100].index)
train = train.drop(train[train.distance>1000000].index)

# print(train.head())
# print(train.info())
train.to_csv (r'~/csv_data/'+str(region)+'/export_dataframe_'+str(ot)+'.csv', index = False, header=True)
print("OK! region: "+str(region))
exit()
