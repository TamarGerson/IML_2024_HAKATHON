import numpy as np
import pandas as pd
from geopy.distance import geodesic
import plotly.express as px
from scipy.stats import pearsonr

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


PASSENGER_PRE_PRO_COLUMNS = ["passengers_up"  # LABLES
    , "passengers_continue"]


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

PREP_FUNC = {
    "delete_null": delete_null
    , "delete_outliers": delete_outliers
}


def preprocess_passengers_data(file_path):
    data = pd.read_csv(file_path, encoding="ISO-8859-8")
    data = delete_null(data)
    data = delete_outliers(data)
    data = add_square_station_index_column(data)
    data = add_30_minute_interval(data, 'arrival_time', '10_min_interval')
    return data


# TODO:
def read_and_preprocess_data(file_path):
    pass


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


def preprocess_trip_data(file_path):
    data = read_and_preprocess_data(file_path)
    trip_summary = create_trip_summary(data)
    merged_data = merge_summary_with_data(data, trip_summary)
    return merged_data


# -------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

def determine_rush_hours(csv_file, time_column, plot=True, encoding='utf-8'):
    """
    Determine rush hours based on transportation data.

    Parameters:
    csv_file (str): Path to the CSV file containing transportation data.
    time_column (str): The name of the column containing timestamp data.
    plot (bool): Whether to plot the distribution of trips/passengers by hour.

    Returns:
    List of int: Rush hours identified by the data.
    """
    # Load the CSV file into a DataFrame
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file, encoding=encoding)
    except UnicodeDecodeError:
        # If a UnicodeDecodeError occurs, try a different encoding
        print(f"Failed to read CSV file with encoding '{encoding}'. Trying 'latin1' instead.")
        df = pd.read_csv(csv_file, encoding='latin1')

    # Convert the time column to datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Extract the hour from the time column
    df['hour'] = df[time_column].dt.hour

    # Group by hour and count the number of trips/passengers
    hourly_counts = df.groupby('hour').size()

    # Identify the rush hours (e.g., top 3 hours with the highest counts)
    rush_hours = hourly_counts.nlargest(3).index.tolist()

    if plot:
        # Plot the distribution of trips/passengers by hour
        hourly_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Trips/Passengers')
        plt.title('Distribution of Trips/Passengers by Hour')
        plt.axhline(y=hourly_counts.mean(), color='r', linestyle='--', label='Average')
        plt.legend()
        plt.show()

    return rush_hours




if __name__ == '__main__':
    # GET RUSH HOUERS
    csv_file = 'train_data.csv'  # Replace with your CSV file path
    time_column = 'arrival_time'  # Replace with your time column name
    rush_hours = determine_rush_hours(csv_file, time_column)
    print("Rush hours based on the data:", rush_hours)
    
    pass