import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess import *
from eda import *
from split_data import *
import sys

NUM_OF_ARGUMENTS = 2
TARGET_COLUMN = 'passengers_up'
TIME_COLUMN = 'arrival_time'


def perform_linear_regression(train_data, test_data, target_column):
    # Separate features and target variable from training data
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    # Separate features and target variable from test data
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error on the test data: {mse}")


def evaluate_preprocess(file_path, test_size_percentage, seed):
    train_data, test_data = split_data(file_path, test_size_percentage, seed)
    # basic preprocess for train and test
    # advanced preprocess
    # linear regression for both

if __name__ == "__main__":
    if len(sys.argv) != NUM_OF_ARGUMENTS:
        print("Usage: python evaluate_passengers.py <file_path>")
    else:
        file_path = sys.argv[1]
        train_data = preprocess_passengers_data(file_path)
        plot_avg_passengers_per_interval_by_area(train_data, TARGET_COLUMN)
        plot_all_correlations(train_data, TARGET_COLUMN, "first_try")


        # perform_linear_regression(train_data, test_data, target_column)