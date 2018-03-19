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
DailyDummary = namedtuple("DailySummary", features)

def extract_weather_data(url, api_key, target_date, days):
    records = []
    for _ in range(days):
        request = BASE_URL.format(API_KEY, target_date.strftime('%Y%m%d'))
        response = requests.get(request)
        

