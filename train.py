import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Model trained. Accuracy: {acc}")


if __name__ == "__main__":
    main()
