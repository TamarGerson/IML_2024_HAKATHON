import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preprocess import *


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


if __name__ == "__main__":
    # Get the train and test data
    train_data, test_data = passengers_data()

    # Specify the target column
    target_column = 'your_target_column'  # Replace with your target column name

    # Perform linear regression and evaluate the loss
    perform_linear_regression(train_data, test_data, target_column)