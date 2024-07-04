import pandas as pd
import plotly.express as px
import os
from scipy.stats import pearsonr

def save_plotly_fig(fig, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = os.path.join(folder, file_name)
    fig.write_html(file_path)

def plot_correlation(df, column1, column2):
    if df[column1].nunique() == 1:
        print(f"Skipping column {column1} because it has constant values.")
        return None

    correlation, _ = pearsonr(df[column1], df[column2])
    title = f'Correlation Plot between {column1} and {column2} (r = {correlation:.2f})'
    fig = px.scatter(df, x=column1, y=column2, trendline="ols",
                     title=title, labels={column1: column1, column2: column2})
    return fig

def plot_all_correlations(df, target_column, save_folder):
    for column in df.columns:
        if column != target_column and pd.api.types.is_numeric_dtype(df[column]):
            fig = plot_correlation(df, column, target_column)
            if fig is not None:
                file_name = f'correlation_{column}_vs_{target_column}.html'
                save_plotly_fig(fig, save_folder, file_name)





