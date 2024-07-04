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


def train_passengers_forest_model(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    return best_rf


def train_passengers_linear_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


if __name__ == "__main__":
    if len(sys.argv) != NUM_OF_ARGUMENTS:
        print("Usage: python evaluate_passengers.py <file_path>")
    else:
        file_path = sys.argv[1]
        train_data = preprocess_passengers_data(file_path)
        plot_avg_passengers_per_interval_by_area(train_data, TARGET_COLUMN)
        #plot_all_correlations(train_data, TARGET_COLUMN, "second_try")


        # perform_linear_regression(train_data, test_data, target_column)
