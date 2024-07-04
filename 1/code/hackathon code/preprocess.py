import numpy as np
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from datetime import datetime

datetime_format = "%H:%M:%S"
RUSH_OR_SCALE = 0
SCALE = 0
MARK = 1


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






# TODO: trip_id -> int (id)
# TODO: part -> int (id)
# TODO: trip_id_unique_station -> int (id)
# TODO: trip_id_unique -> int (id)
# TODO: line_id id -> int (id)
# IS LINE ID RELEVANT (DEST 9 vs 15 rehavia)
# TODO: alternative -> number of alternatives?
# TODO: cluster -> int(id)
# TODO: station_name -> needed? id and index
# TODO: arrival_time -> hh only? peak time only {0,1}
# DO WE NEED TO ADD time^2
# TODO: door_closing_time - arrival_time ? more passenj.


def add_rush_h_col(X: pd.DataFrame) -> pd.DataFrame:
    rush_hours = get_rush_h(X)
    # print(rush_hours)

    if RUSH_OR_SCALE == SCALE:
        mathod = scale_rush_hours 
    else:
        mathod = add_rush_mark
    X = mathod(X, rush_hours)

    return X



def add_rush_mark(X: pd.DataFrame, rush_hours: list):
    
    def is_rush_hour(hour):
        return 1 if hour in rush_hours else 0
    
    X['hour'] = X['arrival_time'].dt.hour
    X['rush_hour'] = X['hour'].apply(is_rush_hour)
    return X




def scale_rush_hours(df: pd.DataFrame, rush_hours: list) -> pd.DataFrame:
    
    passenger_counts = dict(zip(df['hour'], df['passengers_up']))

    # Find maximum and minimum passenger counts to normalize grades
    max_passengers = max(passenger_counts.values())
    min_passengers = min(passenger_counts.values())

    # Scaling factor
    scale_factor = 9 / (max_passengers - min_passengers) if max_passengers != min_passengers else 1

    # Initialize list for scaled rush hour grades
    scaled_rush_hours = []

    # Assign grades based on passenger counts
    for hour in df['hour']:
        if hour in passenger_counts:
            passengers = passenger_counts[hour]
            # Scale passenger count to a grade between 1 and 10
            grade = 1 + scale_factor * (passengers - min_passengers)
            scaled_rush_hours.append(round(grade))
        else:
            scaled_rush_hours.append(0)  # Handle missing hour data

    
    df['rush_hour_grade'] = scaled_rush_hours
    return df  # Return None if inplace=True (to match pandas convention)


def get_rush_h(X: pd.DataFrame):
    
    X['arrival_time'] = pd.to_datetime(X['arrival_time'], format=datetime_format)
    X['hour'] = X['arrival_time'].dt.hour
    hourly_passenger_counts = X.groupby('hour')["passengers_up"].sum()

    rush_hours = hourly_passenger_counts.nlargest(6).index.tolist()
    return rush_hours



def mult_cul(X: pd.DataFrame, col_1, col_2):
    col_name = "{}_vs_{}".format(col_1, col_2)
    X[col_name] = X[col_1] * X [col_2]
    return X




# TODO PASSENGERS---------------------------------------------------------------:
# TODO: passengers_continue -> threshold? pas_up?
def clean_negative_passengers(X: pd.DataFrame):
    for passeng_fitch in PASSENGER_PRE_PRO_COLUMNS:
        X = X[X[passeng_fitch] >= 0]
    return X


#   -> no floats (clean_half_pepole())
def clean_half_persons(X: pd.DataFrame):
    for passeng_fitch in PASSENGER_PRE_PRO_COLUMNS:
        X[passeng_fitch] = pd.to_numeric(X[passeng_fitch], errors='coerce')
    X.dropna(subset=PASSENGER_PRE_PRO_COLUMNS)
    return X


# PASSENGERS---------------------------------------------------------------:


#TODO station---------------------------------------------------------------::
def clean_time_in_station(X: pd.DataFrame) -> pd.DataFrame:
    # Ensure columns exist
    if 'door_closing_time' not in X.columns or 'arrival_time' not in X.columns:
        raise ValueError("DataFrame must contain 'door_closing_time' and 'arrival_time' columns")
    
    # Convert columns to datetime if they are not already
    X['door_closing_time'] = pd.to_datetime(X['door_closing_time'])
    X['arrival_time'] = pd.to_datetime(X['arrival_time'])
    
    # Ensure no missing values in the required columns
    X = X.dropna(subset=['door_closing_time', 'arrival_time'])
    
    # Calculate the time difference and filter rows
    X = X[(X['door_closing_time'] - X['arrival_time']) >= pd.Timedelta(0)]
    
    return X
# time------------------------------------------------------------------::

def convert_to_datetime(df, columns = ['arrival_time', 'door_closing_time']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')
    return df

def calculate_time_diff(df, start_column, end_column, diff_column_name = "diff time"):
    df = convert_to_datetime(df, [start_column, end_column])
    df[start_column] = pd.to_datetime(df[start_column], errors='coerce')
    df[end_column] = pd.to_datetime(df[end_column], errors='coerce')
    df[end_column] = df[end_column].fillna(df[start_column])

    mask = df[end_column] < df[start_column]
    df.loc[mask, [start_column, end_column]] = df.loc[mask, [end_column, start_column]].values

    df[diff_column_name] = (df[end_column] - df[start_column]).dt.total_seconds() / 60
    df[diff_column_name].fillna(0, inplace=True)
    return df


# station---------------------------------------------------------------::

def add_last_station_column(data):
    # Determine the last station for each trip_id_unique
    data['last_station'] = data.groupby('trip_id_unique')['station_index'].transform(max) == data['station_index']
    data['last_station'] = data['last_station'].astype(int)  # Convert boolean to binary (0/1)
    return data


def add_square_station_index_column(data):
    # Create a column for the square of station_index
    data['station_index_squared'] = data['station_index'] ** 2
    return data


def add_30_minute_interval(df, time_column, interval_column_name):
    # Convert column to datetime
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    # Create a new column with 10-minute intervals
    df[interval_column_name] = df[time_column].dt.floor('30T')
    return df


def assign_areas(df, lat_column, lon_column):
    # Define your own logic for areas, here we use a simple grid approach
    df['area'] = df.apply(lambda row: f"Area_{int(row[lat_column])}_{int(row[lon_column])}", axis=1)
    return df


def add_area_grade_column(data):
    # Calculate the average number of passengers up for each area (cluster)
    avg_passengers_per_area = data.groupby('cluster')['passengers_up'].mean().reset_index()
    avg_passengers_per_area.rename(columns={'passengers_up': 'avg_passengers_up'}, inplace=True)

    # Assign numeric grades based on quantiles
    avg_passengers_per_area['area_grade'] = pd.qcut(avg_passengers_per_area['avg_passengers_up'],
                                                    q=3, labels=[1, 2, 3])  # 1: low, 2: medium, 3: high

    # Merge the area grade back to the main data
    data = data.merge(avg_passengers_per_area[['cluster', 'area_grade']], on='cluster', how='left')
    return data

# TODO ALL---------------------------------------------------------------:::
def delete_null(X: pd.DataFrame):
    df = X.copy()
    X = df.dropna()  # .drop_duplicates()
    return X


def delete_outliers(X: pd.DataFrame):
    # WHAT ARE THE OUTLIERS?
    for lier in OUTLIERS_KEYS:
        OUTLIERS_FUNC[lier](X)
    return X
# ALL---------------------------------------------------------------:::


def numeric_cols(X: pd.DataFrame):
    columns = [["passengers_up"]]
    for col in columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).round().astype(int)
    return X








# ------------------------------------------PART B----------------------------------------#

def calculate_trip_duration(grouped):
    trip_duration_in_minutes = grouped['arrival_time'].max() - grouped['arrival_time'].min()
    return trip_duration_in_minutes.dt.total_seconds() / 60


def calculate_number_of_stations(grouped):
    return grouped.size()


def extract_station_info(grouped):
    source_station = grouped.first()['station_name']
    dest_station = grouped.last()['station_name']
    return source_station, dest_station


def calculate_departure_time(grouped):
    return grouped['arrival_time'].min()


def calculate_total_passengers(grouped):
    total_passengers = grouped['passengers_up'].sum() + grouped['passengers_continue'].sum()
    return total_passengers


def calculate_avg_passengers_per_station(total_passengers, number_of_stations):
    return total_passengers / number_of_stations


def calculate_distance(grouped):
    source_lat = grouped.first()['latitude']
    source_lon = grouped.first()['longitude']
    dest_lat = grouped.last()['latitude']
    dest_lon = grouped.last()['longitude']
    distance = [geodesic((source_lat[i], source_lon[i]), (dest_lat[i], dest_lon[i])).km for i in range(len(source_lat))]
    return distance


def create_trip_summary(data):
    grouped = data.groupby('trip_id_unique')

    trip_duration_in_minutes = calculate_trip_duration(grouped)
    number_of_stations = calculate_number_of_stations(grouped)
    source_station, dest_station = extract_station_info(grouped)
    departure_time = calculate_departure_time(grouped)
    total_passengers = calculate_total_passengers(grouped)
    avg_passengers_per_station = calculate_avg_passengers_per_station(total_passengers, number_of_stations)
    distance = calculate_distance(grouped)

    trip_summary = pd.DataFrame({
        'trip_id_unique': trip_duration_in_minutes.index,
        'trip_duration_in_minutes': trip_duration_in_minutes.values,
        'number_of_stations': number_of_stations.values,
        'source': source_station.values,
        'dest': dest_station.values,
        'departure_time': departure_time.values,
        'total_passengers': total_passengers.values,
        'avg_passengers_per_station': avg_passengers_per_station.values,
        'distance_km': distance
    })

    return trip_summary


def merge_summary_with_data(data, trip_summary):
    return pd.merge(data, trip_summary, on='trip_id_unique')


# def preprocess_trip_data(file_path):
#     data = read_and_preprocess_data(file_path)
#     trip_summary = create_trip_summary(data)
#     merged_data = merge_summary_with_data(data, trip_summary)
#     return merged_data

def convert_cluster_to_numeric(data):
    data['cluster'], _ = pd.factorize(data['cluster'])
    return data


# -------------------------------------------------------------------------------
PASSENGER_PRE_PRO_COLUMNS = ["passengers_up"  # LABLES
                            ,"passengers_continue"]



FET_HENHECER = [
    ("arrival_time", "passengers_continue")
    ,("direction", "rush_hour")
    ,("direction", "arrival_time")
    ,("rush_hour", "station_id")
    ,("arrival_time", "station_id")
]

ADD_FIT_KEY = [
    "add_rush_h_col"
    ,"add_last_station_column"
]

OUTLIERS_KEYS = [
    "clean_half_persons"
    # ,"clean_time_in_station"
    , "clean_negative_passengers"
]

OUTLIERS_FUNC = {
    "clean_time_in_station": clean_time_in_station
    , "clean_half_persons": clean_half_persons
    , "clean_negative_passengers": clean_negative_passengers
}

def get_hen_fet_cor(X: pd.DataFrame):
    for t in FET_HENHECER:
        fet_1, fet_2 = t
        X = mult_cul(X, fet_1, fet_2)
    return X


PREP_FUNC = {
    "delete_null": delete_null
    , "delete_outliers": delete_outliers
    ,"add_rush_h_col": add_rush_h_col #replace with add method for all
    ,"add_last_station_column" : add_last_station_column
    ,"add_area_grade_column" : add_area_grade_column
    # ,"numeric_cols" : numeric_cols
    
    ,"add_square_station_index_column":add_square_station_index_column
    ,"convert_cluster_to_numeric" : convert_cluster_to_numeric
    
    ,"get_hen_fet_cor" : get_hen_fet_cor
}

# :#############################################################
def preprocess_data(X: pd.DataFrame):
    
    for proc_f in PREP_FUNC.values():
        X = proc_f(X)
    
    return X

def preprocess_passengers_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-8")
    data = preprocess_data(data)
    data = add_30_minute_interval(data, 'arrival_time', '30_min_interval')
    data = assign_areas(data, 'latitude', 'longitude')
    data = calculate_time_diff(data, 'arrival_time', 'door_closing_time')
    return data
# :#############################################################



if __name__ == '__main__':

    
    pass