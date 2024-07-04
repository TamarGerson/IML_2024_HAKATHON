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


def plot_avg_passengers_per_interval(df, time_column, passengers_column):
    # Calculate the average passengers up per 10-minute interval
    avg_passengers = df.groupby('10_min_interval')[passengers_column].mean().reset_index()

    # Create the bar chart
    fig = px.bar(avg_passengers, x='10_min_interval', y=passengers_column,
                 title='Average Passengers Up per 10-Minute Interval')
    fig.update_layout(xaxis_title='Time Interval', yaxis_title='Average Passengers Up')
    fig.show()


def plot_all_correlations(df, target_column, save_folder):
    for column in df.columns:
        if column != target_column and pd.api.types.is_numeric_dtype(df[column]):
            fig = plot_correlation(df, column, target_column)
            if fig is not None:
                file_name = f'correlation_{column}_vs_{target_column}.html'
                save_plotly_fig(fig, save_folder, file_name)
                
                
def determine_rush_hours(csv_file, time_column, plot=True, encoding='utf-8'):
    """
    Determine rush hours based on transportation data.

    Parameters:
    csv_file (str): Path to the CSV file containing transportation data.
    time_column (str): The name of the column containing timestamp data.
    plot (bool): Whether to plot the distribution of trips/passengers by hour.
    encoding (str): The encoding used to read the CSV file.

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
    csv_file = 'train_data.csv'  # Replace with your CSV file path
    time_column = 'timestamp'  # Replace with your time column name
    rush_hours = determine_rush_hours(csv_file, time_column)
    print("Rush hours based on the data:", rush_hours)
    
    pass