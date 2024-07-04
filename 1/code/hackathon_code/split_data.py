import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import numpy as np


def split_data_to_csv(file_path, test_size_percentage, seed):
    data = pd.read_csv(file_path, encoding="ISO-8859-8")
    test_size = test_size_percentage / 100.0
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    train_data.to_csv('test10_data.csv', index=False, encoding="ISO-8859-8")
    test_data.to_csv('test_data.csv', index=False, encoding="ISO-8859-8")

    print(f"Data split completed. Training data saved to 'train_data.csv' and test data saved to 'test_data.csv'.")


def split_data(file_path, test_size_percentage, seed):
    data = pd.read_csv(file_path, encoding="ISO-8859-8")
    test_size = test_size_percentage / 100.0
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    cur_test, test_data = train_test_split(data, test_size=test_size, random_state=seed)
    return train_data, cur_test


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_data.py <file_path> <test_size_percentage> <seed>")
    else:
        file_path = sys.argv[1]
        test_size_percentage = float(sys.argv[2])
        seed = int(sys.argv[3])
        split_data_to_csv(file_path, test_size_percentage, seed)
        # data = pd.read_csv(file_path, encoding="ISO-8859-8")
        # data = data.drop("passengers_up", axis=1)
        # data.to_csv('test10_data.csv', index=False, encoding="ISO-8859-8")
