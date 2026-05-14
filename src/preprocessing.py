from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_pipeline():
    """
    Returns a scikit-learn pipeline for data preprocessing.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler())
        # Add more steps if needed
    ])
    return pipeline
