import preprocess

import numpy as np
import pandas as pd
from geopy.distance import geodesic
import plotly.express as px
from scipy.stats import pearsonr

from datetime import datetime

datetime_format = "%H:%M:%S"

FET_HENHECER = [
    ("arrival_time", "passengers_continue")
    ,("direction", "rush_hour")
    ,("direction", "arrival_time")
    ,("rush_hour", "station_id")
    ,("arrival_time", "station_id")
]

def convert_time_to_float(df: pd.DataFrame):
    
    df['hours'] = df['time'].apply(lambda x: x.hour)
    df['minutes'] = df['time'].apply(lambda x: x.minute)
    df['hours_float'] = df['hours'] + df['minutes'] / 60
    
    return df
