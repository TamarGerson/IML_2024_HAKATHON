import numpy as np
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
from datetime import datetime
from eda import plot_all_correlations, plot_avg_passengers_per_interval_by_area
import sys


TARGET_COLUMN = 'trip_duration_in_minutes'
datetime_format = "%H:%M:%S"
NUM_OF_ARGUMENTS = 2
# ------------------------------------------PART B----------------------------------------#


def calculate_trip_duration(grouped):
    grouped['arrival_time'] = pd.to_datetime(grouped['arrival_time'], format=datetime_format)
    trip_duration_in_minutes = grouped['arrival_time'].max() - grouped['arrival_time'].min()
    return trip_duration_in_minutes.total_seconds() / 60


def calculate_number_of_stations(grouped):
    return len(grouped)


def extract_station_info(grouped):
    source_station = grouped.iloc[0]['station_name']
    dest_station = grouped.iloc[-1]['station_name']
    return source_station, dest_station


def calculate_departure_time(grouped):
    return grouped['arrival_time'].min()


def calculate_total_passengers(grouped):
    total_passengers = grouped['passengers_up'].sum() + grouped['passengers_continue'].sum()
    return total_passengers


def calculate_avg_passengers_per_station(total_passengers, number_of_stations):
    return total_passengers / number_of_stations


def calculate_distance(grouped):
    source_lat = grouped.iloc[0]['latitude']
    source_lon = grouped.iloc[0]['longitude']
    dest_lat = grouped.iloc[-1]['latitude']
    dest_lon = grouped.iloc[-1]['longitude']
    distance = geodesic((source_lat, source_lon), (dest_lat, dest_lon)).km
    return distance


def create_trip_summary(data):
    grouped = data.groupby('trip_id_unique')

    trip_duration_in_minutes = grouped.apply(calculate_trip_duration)
    number_of_stations = grouped.apply(calculate_number_of_stations)
    source_station, dest_station = zip(*grouped.apply(extract_station_info))
    departure_time = grouped.apply(calculate_departure_time)
    total_passengers = grouped.apply(calculate_total_passengers)
    avg_passengers_per_station = total_passengers / number_of_stations
    distance = grouped.apply(calculate_distance)

    trip_summary = pd.DataFrame({
        'trip_id_unique': trip_duration_in_minutes.index,
        'trip_duration_in_minutes': trip_duration_in_minutes.values,
        'number_of_stations': number_of_stations.values,
        'source': source_station,
        'dest': dest_station,
        'departure_time': departure_time.values,
        'total_passengers': total_passengers.values,
        'avg_passengers_per_station': avg_passengers_per_station.values,
        'distance_km': distance.values
    })

    return trip_summary


def merge_summary_with_data(data, trip_summary):
    return pd.merge(data, trip_summary, on='trip_id_unique')


def preprocess_passengers_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, encoding="ISO-8859-8")

    # Ensure 'arrival_time' is in datetime format
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format=datetime_format)

    # Create a summary of each trip
    trip_summary = create_trip_summary(data)

    # Merge the summary with the original data
    # data = merge_summary_with_data(data, trip_summary)

    return trip_summary


if __name__ == "__main__":

    # if len(sys.argv) != NUM_OF_ARGUMENTS:
    #     print("Usage: python evaluate_passengers.py <file_path>")
    # else:

    # file_path = sys.argv[1]
    file_path = '/Users/ofirhol/Documents/year2/ב/iml/hackathon/IML_2024_HAKATHON/train_data.csv'
    train_data = preprocess_passengers_data(file_path)
    print(train_data.columns)

    # plot_avg_passengers_per_interval_by_area(train_data, TARGET_COLUMN)
    # plot_all_correlations(train_data, TARGET_COLUMN, "part_b_first_try")

    # perform_linear_regression(train_data, test_data, target_column)
