# src/data_loader.py

import pandas as pd
from sklearn.datasets import load_iris

def save_iris_dataset(output_path="data/iris.csv"):
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv(output_path, index=False)
    print(f"Iris dataset saved to {output_path}")

if __name__ == "__main__":
    save_iris_dataset()
