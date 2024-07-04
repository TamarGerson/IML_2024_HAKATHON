import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report

def clean_time_format(time_str):
    """ Clean and ensure time format is HH:MM """
    try:
        if pd.isnull(time_str) or time_str.strip() == '':
            return None
        time_str = time_str.strip()
        parts = time_str.split(':')
        if len(parts) != 2:
            return None
        hour, minute = parts
        if len(hour) == 1:
            hour = f'0{hour}'
        if len(minute) == 1:
            minute = f'0{minute}'
        return f'{hour}:{minute}'
    except Exception as e:
        return None

def load_and_preprocess_data(file_path):
    # Load the data from a CSV file
    df = pd.read_csv(file_path)

    # Clean the arrival_time and door_closing_time columns
    df['arrival_time'] = df['arrival_time'].apply(clean_time_format)
    df['door_closing_time'] = df['door_closing_time'].apply(clean_time_format)

    # Drop rows with invalid time formats
    df.dropna(subset=['arrival_time', 'door_closing_time'], inplace=True)

    # Drop unused columns
    unused_columns = ['door_closing_time', 'station_id', 'cluster']
    df.drop(columns=unused_columns, inplace=True)

    # Extract hour and minute from arrival_time
    df['hour'] = pd.to_datetime(df['arrival_time'], format='%H:%M').dt.hour
    df['minute'] = pd.to_datetime(df['arrival_time'], format='%H:%M').dt.minute

    # Create direction dummies
    df = pd.get_dummies(df, columns=['direction'])

    # Drop the original arrival_time column as it's no longer needed
    df.drop(columns=['arrival_time'], inplace=True)

    return df

def train_decision_tree_model(df):
    # Define features and target
    features = ['hour'
                , 'minute'
                # , 'rush_hour'
                # , 'direction'
                , 'passengers_continue'
                , 'passengers_continue_menupach']
    target = 'passengers_up'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Train the decision tree model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate absolute errors
    abs_errors = abs(y_test - y_pred)
    sum_abs_errors = sum(abs_errors)

    # Calculate total number of passengers that got up
    total_passengers_up = sum(y_test)

    # Calculate error percentage
    if total_passengers_up > 0:
        error_percentage = sum_abs_errors / total_passengers_up * 100
    else:
        error_percentage = float('nan')  # Handle case where total_passengers_up is zero

    # Print results
    print("\nSum of Absolute Errors:", sum_abs_errors)
    print("Sum of total_passengers_up:", total_passengers_up)
    print("Error Percentage:", error_percentage)

    # Print predictions and true values
    print("\nTrue vs Predicted values:")
    comparison = pd.DataFrame({'y_test_true_val': y_test, 'y_test_pred': y_pred})
    print(comparison)

    # Evaluate the model
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf

# Specify the path to your CSV file
csv_file_path = '/Users/ofirhol/Documents/year2/ב/iml/hackathon/IML_2024_HAKATHON/train_data.csv'

# Load and preprocess the data
df = load_and_preprocess_data(csv_file_path)

# Train the model using the preprocessed data
trained_model = train_decision_tree_model(df)
