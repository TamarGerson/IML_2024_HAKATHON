# from argparse import ArgumentParser
# import logging
# import pandas as pd

# """
# usage:
#     python code/main.py --training_set PATH --test_set PATH --out PATH

# for example:
#     python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

# """


# def load_data(set_path):
#     return pd.read_csv(set_path, encoding="ISO-8859-8")


# def save_predictions(predictions, trip_id_unique, output_path):
#     output_df = pd.DataFrame({
#         'trip_id_unique_station': trip_id_unique,
#         'trip_duration_in_minutes': predictions
#     })
#     output_df.to_csv(output_path, index=False,  encoding="ISO-8859-8")


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--training_set', type=str, required=True,
#                         help="path to the training set")
#     parser.add_argument('--test_set', type=str, required=True,
#                         help="path to the test set")
#     parser.add_argument('--out', type=str, required=True,
#                         help="path of the output file as required in the task description")
#     args = parser.parse_args()

#     # 1. load the training set (args.training_set)
#     # 2. preprocess the training set
#     logging.info("preprocessing train...")
#     training_data = load_data(args.training_set)
#     X_train, y_train = preprocess_duration_train(training_data)
#     # 3. train a model
#     logging.info("training...")
#     model = train_duration_model(X_train, y_train)
#     # 4. load the test set (args.test_set)
#     # 5. preprocess the test set
#     logging.info("preprocessing test...")
#     test_data = load_data(args.test_set)
#     X_test = preprocess_duration_test(test_data)
#     # 6. predict the test set using the trained model
#     logging.info("predicting...")
#     predictions = model.predict(X_test)
#     # 7. save the predictions to args.out
#     save_predictions(predictions, test_data['trip_id_unique'], args.out)
#     logging.info("predictions saved to {}".format(args.out))

# ----------------------------- above this line is what neomi wrote  -----------------------------

# ----------------------------- below this line is what gpt wrote  -----------------------------
from argparse import ArgumentParser
import logging
import pandas as pd
from xgboost import XGBRegressor

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


def load_data(set_path):
    return pd.read_csv(set_path, encoding="ISO-8859-8")


def save_predictions(predictions, trip_id_unique, output_path):
    output_df = pd.DataFrame({
        'trip_id_unique': trip_id_unique,
        'trip_duration_in_minutes': predictions
    })
    output_df.to_csv(output_path, index=False, encoding="ISO-8859-8")


def train_duration_model(X_train, y_train):
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_train, y_train)], verbose=False)
    return model


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
    training_data = load_data(args.training_set)

    # Assume that 'trip_duration_in_minutes' is the target column in the preprocessed training data
    features = ['number_of_stops', 'distance', 'total_passengers']
    target = 'trip_duration_in_minutes'
    X_train = training_data[features]
    y_train = training_data[target]

    # 3. train a model
    logging.info("training...")
    model = train_duration_model(X_train, y_train)

    # 4. load the test set (args.test_set)
    test_data = load_data(args.test_set)

    # Assume that the test data is preprocessed and ready for prediction
    X_test = test_data[features]

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)

    # 7. save the predictions to args.out
    save_predictions(predictions, test_data['trip_id_unique'], args.out)
    logging.info("predictions saved to {}".format(args.out))
