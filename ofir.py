import pandas as pd

def convert_to_datetime(df, columns = ['arrival_time', 'door_closing_time']):
    for col in columns:
        df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')
    return df

def calculate_time_diff(df, start_column, end_column, diff_column_name = "diff time"):
    df[start_column] = pd.to_datetime(df[start_column], errors='coerce')
    df[end_column] = pd.to_datetime(df[end_column], errors='coerce')
    df[end_column] = df[end_column].fillna(df[start_column])
    
    mask = df[end_column] < df[start_column]
    df.loc[mask, [start_column, end_column]] = df.loc[mask, [end_column, start_column]].values
    
    df[diff_column_name] = (df[end_column] - df[start_column]).dt.total_seconds() / 60
    df[diff_column_name].fillna(0, inplace=True)
    return df


df = pd.read_csv('/Users/ofirhol/Documents/year2/ב/iml/hackathon/IML_2024_HAKATHON/test_data.csv', encoding="ISO-8859-8")
df = convert_to_datetime(df)
df = calculate_time_diff(df, 'arrival_time', 'door_closing_time')
filtered_df = df[df["diff time"] != 0 ]
filtered_df = filtered_df[df["diff time"] != 1 ]          
print(filtered_df.head())

