import pandas as pd
import plotly.express as px
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def save_plotly_fig(fig, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, file_name)
    fig.write_html(file_path)


def plot_correlation(df, column1, column2):
    if df[column1].nunique() == 1:
        print(f"Skipping column {column1} because it has constant values.")
        return None

    correlation, _ = pearsonr(df[column1], df[column2])
    title = f'Correlation Plot between {column1} and {column2} (r = {correlation:.2f})'
    fig = px.scatter(df, x=column1, y=column2, trendline="ols",
                     title=title, labels={column1: column1, column2: column2})
    return fig


def plot_avg_passengers_per_interval_by_area(df, passengers_column):
    # Create 10-minute intervals
    df['10_min_interval'] = df['arrival_time'].dt.floor('30T')

    # Calculate the average passengers up per 10-minute interval for each cluster
    avg_passengers = df.groupby(['cluster', '30_min_interval'])[passengers_column].mean().reset_index()

    # Plot for each cluster
    clusters = avg_passengers['cluster'].unique()
    for cluster in clusters:
        cluster_data = avg_passengers[avg_passengers['cluster'] == cluster]
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_data['30_min_interval'].astype(str), cluster_data[passengers_column])
        plt.xlabel('Time Interval')
        plt.ylabel('Average Passengers Up')
        plt.title(f'Average Passengers Up per 10-Minute Interval - Cluster {cluster}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'avg_passengers_up_cluster_{cluster}.png')
        plt.close()
        print(f"Graph for cluster {cluster} saved as avg_passengers_up_cluster_{cluster}.png")


def plot_all_correlations(df, target_column, save_folder):
    for column in df.columns:
        if column != target_column and pd.api.types.is_numeric_dtype(df[column]):
            print("plotting correlation")
            fig = plot_correlation(df, column, target_column)
            if fig is not None:
                file_name = f'correlation_{column}_vs_{target_column}.html'
                save_plotly_fig(fig, save_folder, file_name)
                
                
# def determine_rush_hours(csv_file, time_column, plot=True, encoding='utf-8'):
#     """
#     Determine rush hours based on transportation data.

#     Parameters:
#     csv_file (str): Path to the CSV file containing transportation data.
#     time_column (str): The name of the column containing timestamp data.
#     passenger_column (str): The name of the column containing the number of passengers.
#     plot (bool): Whether to plot the distribution of trips/passengers by hour.
#     encoding (str): The encoding used to read the CSV file.
#     datetime_format (str): The format of the datetime strings in the time column.

#     Returns:
#     List of int: Rush hours identified by the data.
#     """

#     # Convert the time column to datetime
#     df[time_column] = pd.to_datetime(df[time_column], format=datetime_format)

#     # Extract the hour from the time column
#     df['hour'] = df[time_column].dt.hour

#     # Group by hour and sum the number of passengers
#     hourly_passenger_counts = df.groupby('hour')[passenger_column].sum()

#     # Identify the rush hours (e.g., top 3 hours with the highest counts)
#     rush_hours = hourly_passenger_counts.nlargest(3).index.tolist()

#     if plot:
#         # Plot the distribution of passengers by hour
#         hourly_passenger_counts.plot(kind='bar', color='skyblue')
#         plt.xlabel('Hour of the Day')
#         plt.ylabel('Number of Passengers')
#         plt.title('Distribution of Passengers by Hour')
#         plt.axhline(y=hourly_passenger_counts.mean(), color='r', linestyle='--', label='Average')
#         plt.legend()
#         plt.show()

#     return rush_hours

def determine_rush_hours(csv_file, time_column, passenger_column, plot=True, encoding='utf-8', datetime_format=None):
    """
    Determine rush hours based on the sum of passengers in the transportation data.

    Parameters:
    csv_file (str): Path to the CSV file containing transportation data.
    time_column (str): The name of the column containing timestamp data.
    passenger_column (str): The name of the column containing the number of passengers.
    plot (bool): Whether to plot the distribution of trips/passengers by hour.
    encoding (str): The encoding used to read the CSV file.
    datetime_format (str): The format of the datetime strings in the time column.

    Returns:
    List of int: Rush hours identified by the data.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file, encoding=encoding)
    except UnicodeDecodeError:
        # If a UnicodeDecodeError occurs, try a different encoding
        print(f"Failed to read CSV file with encoding '{encoding}'. Trying 'latin1' instead.")
        df = pd.read_csv(csv_file, encoding='latin1')

    # Convert the time column to datetime
    df[time_column] = pd.to_datetime(df[time_column], format=datetime_format)

    # Extract the hour from the time column
    df['hour'] = df[time_column].dt.hour

    # Group by hour and sum the number of passengers
    hourly_passenger_counts = df.groupby('hour')[passenger_column].sum()

    # Identify the rush hours (e.g., top 3 hours with the highest counts)
    rush_hours = hourly_passenger_counts.nlargest(3).index.tolist()

    if plot:
        # Plot the distribution of passengers by hour
        hourly_passenger_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Passengers')
        plt.title('Distribution of Passengers by Hour')
        plt.axhline(y=hourly_passenger_counts.mean(), color='r', linestyle='--', label='Average')
        plt.legend()
        plt.show()

    return rush_hours



if __name__ == '__main__':
    csv_file = 'train_data.csv'  # Replace with your CSV file path
    time_column = 'arrival_time'  # Replace with your time column name
    passenger_column = 'passengers_up'  # Replace with the actual passenger column name in your CSV file
    datetime_format = "%H:%M:%S"  # Adjust the format based on the actual format of your timestamps
    rush_hours = determine_rush_hours(csv_file, time_column, passenger_column, datetime_format=datetime_format)
    print("Rush hours based on the data:", rush_hours)
    
    pass