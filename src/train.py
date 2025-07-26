# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_data(path="data/iris_processed.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Log params, metrics, and model
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"{model_name} accuracy: {accuracy}")

        return accuracy, run_id

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    results = []

    # Train and evaluate Logistic Regression
    lr = LogisticRegression(max_iter=200)
    acc_lr, run_id_lr = train_and_log_model(lr, "LogisticRegression", X_train, X_test, y_train, y_test)
    results.append(("LogisticRegression", acc_lr, run_id_lr))

    # Train and evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    acc_rf, run_id_rf = train_and_log_model(rf, "RandomForest", X_train, X_test, y_train, y_test)
    results.append(("RandomForest", acc_rf, run_id_rf))

    # Register best model
    best_model = max(results, key=lambda x: x[1])  # x[1] is accuracy
    best_name, best_acc, best_run_id = best_model

    model_uri = f"runs:/{best_run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="iris-best-model"
    )

    print(f"Best model '{best_name}' registered as 'iris-best-model' with accuracy: {best_acc}")
