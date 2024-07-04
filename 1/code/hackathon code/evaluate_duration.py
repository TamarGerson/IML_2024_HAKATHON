import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess_part_b import *
from eda import *
from split_data import *
import sys


TARGET_COLUMN = 'trip_duration_in_minutes'
datetime_format = "%H:%M:%S"
NUM_OF_ARGUMENTS = 2


if __name__ == "__main__":

    if len(sys.argv) != NUM_OF_ARGUMENTS:
        print("Usage: python evaluate_passengers.py <file_path>")
    else:
        # file_path = sys.argv[1]
        file_path = '/Users/ofirhol/Documents/year2/ב/iml/hackathon/IML_2024_HAKATHON/train_data.csv'
        train_data = preprocess_duration_train(file_path)
        print(train_data.columns)

    # plot_avg_passengers_per_interval_by_area(train_data, TARGET_COLUMN)
    # plot_all_correlations(train_data, TARGET_COLUMN, "part_b_first_try")

    # perform_linear_regression(train_data, test_data, target_column)
