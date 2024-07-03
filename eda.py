import pandas as pd
def plot_correlation(df, column1, column2):
    # Calculate the correlation coefficient
    correlation, _ = pearsonr(df[column1].dropna(), df[column2].dropna())
    
    # Create the plot with the correlation coefficient in the title
    title = f'Correlation Plot between {column1} and {column2} (r = {correlation:.2f})'
    fig = px.scatter(df, x=column1, y=column2, trendline="ols",
                     title=title, labels={column1: column1, column2: column2})
    fig.show()

