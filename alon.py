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



def add_rush_h_col(X: pd.DataFrame) -> pd.DataFrame:
    
    rush_hours = get_rush_h(X)
    
    def is_rush_hour(hour):
        return 1 if hour in rush_hours else 0
    
    X['hour'] = X['arrival_time'].dt.hour
    X['rush_hour'] = X['hour'].apply(is_rush_hour)
    return X



def get_rush_h(X: pd.DataFrame):
    
    X['arrival_time'] = pd.to_datetime(X['arrival_time'], format=datetime_format)
    X['hour'] = X['arrival_time'].dt.hour
    hourly_passenger_counts = X.groupby('hour')["passengers_up"].sum()

    rush_hours = hourly_passenger_counts.nlargest(3).index.tolist()
    return rush_hours



def mult_cul(X: pd.DataFrame, col_1, col_2):
    col_name = "{}_vs_{}".format(col_1, col_2)
    X[col_name] = X[col_1] * X [col_2]
    return X


def get_hen_fet_cor(X: pd.DataFrame):
    for fet_1, fet_2 in FET_HENHECER:
        X = mult_cul(X, fet_1, fet_2)
    return X




ADD_FIT_KEY = [
    "add_rush_h_col"
]


ADD_F_FUNC = {
    "add_rush_h_col": add_rush_h_col
}

