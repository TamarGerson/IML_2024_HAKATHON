import preprocess

import numpy as np
import pandas as pd
from geopy.distance import geodesic
import plotly.express as px
from scipy.stats import pearsonr

from datetime import datetime


def add_rush_h_col(X: pd.DataFrame, rush_hours) -> pd.DataFrame:
    def is_rush_hour(hour):
        return 1 if hour in rush_hours else 0
    
    X['hour'] = X['arrival_time'].dt.hour
    X['rush_hour'] = X['hour'].apply(is_rush_hour)
    return X



