import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    test_values = pd.read_csv('data/test_values.csv', header=None, index_col=0)
    predict_values = pd.read_csv('data/predict_values.csv', header=None, index_col=0)
    assert accuracy_score(test_values, predict_values) >= 0.9, "Should be more than 0.9"