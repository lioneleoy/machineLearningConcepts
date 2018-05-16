from datetime import datetime, timedelta
import time
from collections import namedtuple
import pandas as pd 
import requests
import matplotlib.pyplot as plt

BASE_URL = "http://api.wunderground.com/api/{}/history_{}/q/NE/Lincoln.json"
API_KEY = '8e5882650987c735'

target_date = datetime(2017, 5, 16)
features = ["date", "meantempm", "meandewptm", "meanpressurem", "maxhumidity", "minhumidity", "maxtempm",  
            "mintempm", "maxdewptm", "mindewptm", "maxpressurem", "minpressurem", "precipm"]
DailySummary = namedtuple("DailySummary", features)

def extract_weather_data(url, api_key, target_date, days):
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        # print response
        if response.status_code == 200:
            data = response.json()['history']['dailysummary'][0]
            records.append(DailySummary(
                date=target_date,
                meantempm=data['meantempm'],
                meandewptm=data['meandewptm'],
                meanpressurem=data['meanpressurem'],
                maxhumidity=data['maxhumidity'],
                minhumidity=data['minhumidity'],
                maxtempm=data['maxtempm'],
                mintempm=data['mintempm'],
                maxdewptm=data['maxdewptm'],
                mindewptm=data['mindewptm'],
                maxpressurem=data['maxpressurem'],
                minpressurem=data['minpressurem'],
                precipm=data['precipm']))
        time.sleep(15)
        target_date += timedelta(days= 1)
        print (records)
    return records

weather_data = extract_weather_data(BASE_URL, API_KEY, target_date, 500)
print "initiating"







