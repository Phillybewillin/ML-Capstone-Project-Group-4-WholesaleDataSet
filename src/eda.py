import seaborn as sns
import matplotlib.pyplot as plt

def perform_full_eda(df):
    """
    Data Health Check. 
    Finds nulls, duplicates, and data types to prevent pipeline errors.
    """
    print("--- DATA HEALTH REPORT ---")
    
    # 1. Check for Nulls/Empty Cells
    null_counts = df.isnull().sum()
    print(f"\nMissing Values per Column:\n{null_counts[null_counts > 0] if null_counts.any() else 'None found'}")
    
    # 2. Check for Duplicates
    dupes = df.duplicated().sum()
    print(f"\nDuplicate Rows Found: {dupes}")
    
    # 3. Data Type Check (Ensures all are numeric for clustering)
    print(f"\nData Types:\n{df.dtypes}")
    
    print("\nSummary Statistics:")
    print(df.describe())

def plot_visuals(df):
    """
    Generates the visual EDA required for the 5-page report.
    """
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show()

    # Boxplots for Outliers (Vital for DBSCAN/K-Means choice)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title('Outlier Detection (Boxplots)')
    plt.show()
