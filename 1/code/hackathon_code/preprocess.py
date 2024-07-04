import statsmodels.api as sm
import numpy as np
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from datetime import datetime

datetime_format = "%H:%M:%S"
RUSH_OR_SCALE = 1
SCALE = 0
MARK = 1
RUSH_H_HYPER = 1
NUM_OF_RUSH_H = 6
TIME_COLUMNS = ['arrival_time', 'door_closing_time']
PASSENGER_PRE_PRO_COLUMNS = ['passengers_continue', 'passengers_up']
CATEGORIAL_COLUMNS = ['alternative', 'cluster', 'station_name', 'part', 'trip_id_unique_station,'
                      'trip_id_unique']

# passengers_continue_menupach
# The Passenger Continuity Inflation Factor (PCIF) 
# in transportation refers to the additional time built 
# into a schedule to account for the process of passengers boarding,
# alighting, and transferring between services.

# mekadem_nipuach_luz
# The timetable inflation factor (TIF)
# in transportation refers to the practice of adding extra time
# to a transportation schedule to account for potential delays
# and ensure a more reliable service. 


def convert_time_to_float(df: pd.DataFrame):
    df['hours'] = df['arrival_time'].apply(lambda x: x.hour)
    df['minutes'] = df['arrival_time'].apply(lambda x: x.minute)
    df['hours_float'] = df['hours'] + df['minutes'] / 60
    
    return df


def mult_cul(X: pd.DataFrame, col_1, col_2):
    col_name = "{}_vs_{}".format(col_1, col_2)
    X[col_name] = X[col_1] * X [col_2]
    return X


def clean_negative_passengers(X: pd.DataFrame):
    for passeng_fitch in PASSENGER_PRE_PRO_COLUMNS:
        X = X[X[passeng_fitch] >= 0]
    return X


def clean_time_in_station(X: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist
    if 'door_closing_time' not in X.columns or 'arrival_time' not in X.columns:
        raise ValueError("DataFrame must contain 'door_closing_time' and 'arrival_time' columns")

    def fill_missing_times(series):
        for i in range(len(series)):
            if pd.isnull(series.iloc[i]):
                # Find nearest non-null values above and below
                prev_value = series.iloc[:i].dropna().iloc[-1] if not series.iloc[:i].dropna().empty else None
                next_value = series.iloc[i + 1:].dropna().iloc[0] if not series.iloc[i + 1:].dropna().empty else None

                if prev_value is not None and next_value is not None:
                    series.iloc[i] = prev_value + (next_value - prev_value) / 2
                elif prev_value is not None:
                    series.iloc[i] = prev_value
                elif next_value is not None:
                    series.iloc[i] = next_value
        return series

    # Convert columns to datetime if they are not already
    X['door_closing_time'] = pd.to_datetime(X['door_closing_time'])
    X['arrival_time'] = pd.to_datetime(X['arrival_time'])
    # Fill missing values
    X['door_closing_time'] = X['door_closing_time'].combine_first(X['arrival_time'])
    X['arrival_time'] = X['arrival_time'].combine_first(X['door_closing_time'])
    # Fill missing values
    X['door_closing_time'] = fill_missing_times(X['door_closing_time'])
    X['arrival_time'] = fill_missing_times(X['arrival_time'])
    # Calculate the time difference and filter rows
    X = X[(X['door_closing_time'] - X['arrival_time']) >= pd.Timedelta(0)]

    return X


def convert_to_datetime(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')
    return df


def calculate_time_diff(df, start_column, end_column, diff_column_name = "door_open_time"):
    df = convert_to_datetime(df, [start_column, end_column])
    df[start_column] = pd.to_datetime(df[start_column], errors='coerce')
    df[end_column] = pd.to_datetime(df[end_column], errors='coerce')
    df[end_column] = df[end_column].fillna(df[start_column])

    mask = df[end_column] < df[start_column]
    df.loc[mask, [start_column, end_column]] = df.loc[mask, [end_column, start_column]].values

    df[diff_column_name] = (df[end_column] - df[start_column]).dt.total_seconds() / 60
    df[diff_column_name].fillna(0, inplace=True)
    return df


def add_square_station_index_column(data):
    # Create a column for the square of station_index
    data['station_index_squared'] = data['station_index'] ** 2
    return data


def add_30_minute_interval(df, time_column, interval_column_name):
    df[interval_column_name] = df[time_column].dt.floor('30T')
    return df


def delete_null(X: pd.DataFrame):
    df = X.copy()
    X = df.dropna()  # .drop_duplicates()
    return X


def numeric_cols(X: pd.DataFrame):
    columns = [["passengers_up"]]
    for col in columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).round().astype(int)
    return X


def convert_col_to_numeric(data, col):
    data[col], _ = pd.factorize(data[col])
    return data


def clean_door_open_time(X: pd.DataFrame) -> pd.DataFrame:
    threshold = 50
    if 'door_open_time' in X.columns:
        X = X[X['door_open_time'] <= threshold]
    return X


COLUMNS_TO_DELETE = {
    'cluster': 'cluster',
    'direction': 'direction',
    'lat': 'latitude',
    'long': 'longitude',
    'line_id': 'line_id',
    'station_id': 'station_id',
    'rush_hour': 'rush_hour',
    'trip_id_unique': 'trip_id_unique',
    'trip_id_unique_station': 'trip_id_unique_station',
    'arrival_time': 'arrival_time',
    'arrival_is_estimated': 'arrival_is_estimated',
    'area': 'area',
    'direction_vs_hours_float': 'direction_vs_hours_float',
    'direction_vs_rush_hour': 'direction_vs_rush_hour',
    'door_closing_time': 'door_closing_time',
    'hour': 'hour',
    'part': 'part',
    'station_name': 'station_name',
    'alternative' : 'alternative',
    '30_min_interval': '30_min_interval'
}


def clean_outliers(X):
    X = clean_negative_passengers(X)
    X = clean_time_in_station(X)
    return X

##############################################################


def preprocess_data(X: pd.DataFrame):
    X = delete_null(X)
    X = clean_outliers(X)
    return X


def preprocess_passengers_eda(data):
    # Ensure 'cluster' column exists
    if 'cluster' not in data.columns:
        raise KeyError("'cluster' column is missing from DataFrame")
    print("Columns:", data.columns)
    data = pd.get_dummies(data, columns=['cluster'], prefix='cluster', drop_first=True)
    data = preprocess_data(data)
    data = convert_to_datetime(data, TIME_COLUMNS)
    data = add_30_minute_interval(data, 'arrival_time', '30_min_interval')
    data = calculate_time_diff(data, 'arrival_time', 'door_closing_time')
    # for col in CATEGORIAL_COLUMNS:
    #     data = convert_col_to_numeric(data, col)
    return data


def preprocess_passengers_data(data):
    data = preprocess_data(data)
    # data = add_30_minute_interval(data, 'arrival_time', '30_min_interval')
    # data = assign_areas(data, 'latitude', 'longitude')
    # data = calculate_time_diff(data, 'arrival_time', 'door_closing_time')
    data = delete_columns(data)
    return data


def delete_columns(X: pd.DataFrame) -> pd.DataFrame:
    columns_dict = COLUMNS_TO_DELETE
    for col in columns_dict.keys():
        if col in X.columns:
            X = X.drop(columns=[col])
    return X


def preprocess_test_data(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    test_df = delete_columns(test_df)
    return test_df


# :#############################################################
################################ - PART A - ################################


#DEBUG:
def print_col(df):
    print("###############################")
    print("\n")
    print("\n")
    column = list(df.columns)
    print(column)
    print("\n")
    print("\n")
    print("###############################")
    return
