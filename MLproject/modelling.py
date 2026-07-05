import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
 
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_preprocessing/heart_preprocessing.csv")
    data = pd.read_csv(file_path)
    
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.3, stratify=y
    )
    input_example = X_train[0:5]

    solver = "liblinear"
    max_iter = 1000 
    random_state = 42

    C = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    penalty = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ["l1", "l2"] else "l2"

    with mlflow.start_run(nested=True):
        model = LogisticRegression(
            C=C, 
            penalty=penalty, 
            solver=solver, 
            max_iter=max_iter, 
            random_state=random_state
        )

        model.fit(X_train, y_train)
 
        predicted_qualities = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        # Mencatat parameter eksperimen ke dalam MLflow Dashboard
        mlflow.log_param("C", C)
        mlflow.log_param("penalty", penalty)

        # Mencatat metrik hasil evaluasi ke dalam MLflow Dashboard
        mlflow.log_metric("accuracy", accuracy)
 
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print(f"Eksperimen Berhasil! Model disimpan ke MLflow.")
        print(f"Parameter -> C: {C}, Penalty: {penalty}")
        print(f"Hasil Akurasi Model: {accuracy:.2f}")