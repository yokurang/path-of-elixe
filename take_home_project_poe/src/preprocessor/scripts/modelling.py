from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def split_data_sklearn(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split datasets using sklearn's train_test_split.
    Returns:
    --------
    tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
