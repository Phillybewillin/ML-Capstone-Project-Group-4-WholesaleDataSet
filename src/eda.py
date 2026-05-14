import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_summary_statistics(df):
    """
    Generate summary statistics (df.describe) for the dataset.
    """
    return df.describe()

def plot_correlation_heatmap(df):
    """
    Generate the initial correlation heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
