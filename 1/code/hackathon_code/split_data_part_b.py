import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = 'trip_duration_in_minutes'
datetime_format = "%H:%M:%S"
NUM_OF_ARGUMENTS = 2

def calculate_trip_duration(grouped):
    grouped['arrival_time'] = pd.to_datetime(grouped['arrival_time'], format=datetime_format)
    trip_duration_in_minutes = grouped['arrival_time'].max() - grouped['arrival_time'].min()
    return trip_duration_in_minutes.total_seconds() / 60

def create_trip_summary(data):
    grouped = data.groupby('trip_id_unique')

    trip_duration_in_minutes = grouped.apply(calculate_trip_duration)

    trip_summary = pd.DataFrame({
        'trip_id_unique': trip_duration_in_minutes.index,
        'trip_duration_in_minutes': trip_duration_in_minutes.values
    })

    return trip_summary

def split_data_to_csv(file_path, test_size_percentage=80, seed=42):
    # Load the data
    data = pd.read_csv(file_path, encoding="ISO-8859-8")
    
    # Split the data into training and test sets
    test_size = test_size_percentage / 100.0
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    
    # Save the training data to a CSV file
    train_file_path = 'train_data.csv'
    train_data.to_csv(train_file_path, index=False, encoding="ISO-8859-8")
    
    create_trip_summary(test_data).to_csv('test_data_gold.csv', index=False, encoding="ISO-8859-8")

    # Prepare the test data by excluding arrival_time except for the first station in each trip
    test_data['arrival_time'] = test_data.apply(
        lambda row: row['arrival_time'] if row['station_index'] == 0 else None, axis=1
    )
    
    # Save the test data to a CSV file
    test_file_path = 'test_data.csv'
    test_data.to_csv(test_file_path, index=False, encoding="ISO-8859-8")
    
    print(f"Data split completed. Training data saved to '{train_file_path}' and test data saved to '{test_file_path}'.")

if __name__ == "__main__":
    split_data_to_csv("/Users/ofirhol/Documents/year2/ב/iml/hackathon/IML_2024_HAKATHON/train_bus_schedule.csv")
