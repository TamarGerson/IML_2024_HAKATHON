import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("train_data.csv", encoding="ISO-8859-8")

# Feature Engineering
data['arrival_time'] = pd.to_datetime(data['arrival_time'])
data['arrival_hour'] = data['arrival_time'].dt.hour
data['arrival_day_of_week'] = data['arrival_time'].dt.dayofweek

# Aggregate data
hourly_passengers = data.groupby(['arrival_hour'])['passengers_up'].mean().reset_index()

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(hourly_passengers.pivot('arrival_hour', 'arrival_day_of_week', 'passengers_up'), cmap="YlGnBu", annot=True)
plt.title('Average Number of Passengers Up by Hour and Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Hour of the Day')
plt.xticks(np.arange(7) + 0.5, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
plt.show()