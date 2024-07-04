from argparse import ArgumentParser
import logging
import pandas as pd
from hackathon_code.evaluate_passengers.py import *
from hackathon_code.preprocess.py import *
"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


def preprocess_passengers_train(training_data):
    training_data = preprocess_passengers_data(training_data)
    return training_data

def preprocess_passengers_test(test_data, training_data):
    return preprocess_test_data(test_data, training_data)

def load_data(set_path):
    return pd.read_csv(set_path, encoding="ISO-8859-8")


def save_predictions(predictions, trip_id_unique_station, output_path):
    output_df = pd.DataFrame({
        'trip_id_unique_station': trip_id_unique_station,
        'passengers_up': predictions
    })
    output_df.to_csv(output_path, index=False, encoding="ISO-8859-8")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    # 2. preprocess the training set
    logging.info("preprocessing train...")
    training_data = load_data(args.training_set)
    training_data = preprocess_passengers_train(training_data)
    X_train, y_train = training_data.drop("passengers_up", axis=1), training_data(["passengers_up"])
    # 3. train a model
    logging.info("training...")
    model = train_passengers_linear_model(X_train, y_train)
    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")
    test_data = load_data(args.test_set)
    X_test = preprocess_passengers_test(args.test_data, training_data)
    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)
    # 7. save the predictions to args.out
    save_predictions(predictions, test_data['trip_id_unique_station'], args.out)
    logging.info("predictions saved to {}".format(args.out))
