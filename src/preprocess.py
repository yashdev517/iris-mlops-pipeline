# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(input_path="data/iris.csv", output_path="data/iris_processed.csv"):
    df = pd.read_csv(input_path)
    features = df.drop('target', axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    scaled_df = pd.DataFrame(scaled, columns=features.columns)
    scaled_df['target'] = df['target']
    scaled_df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")

if __name__ == "__main__":
    preprocess()
