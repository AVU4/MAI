from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('data/iris.csv', header=None)
    train_x, test_x, train_y, test_y = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)
    random_forest = RandomForestClassifier(max_depth=2, random_state=0)
    random_forest.fit(train_x, train_y)
    pd.DataFrame(random_forest.predict(test_x)).to_csv('data/predict_values.csv', header=False)
    pd.DataFrame(test_y).to_csv('data/test_values.csv', header=False)