import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from data_preprocessing import load_data
from mlflow.tracking import MlflowClient

# Tell MLflow where the tracking/registry server is (local default).
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
mlflow.set_registry_uri(mlflow.get_tracking_uri())

# Load and rename data for consistent schema
df = load_data()

# Rename columns to match API input schema
df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width"
}, inplace=True)

# Split features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLflow experiment
mlflow.set_experiment("iris_experiment")

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier()
}

# Track best model
best_accuracy = 0
best_model_name = None
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            input_example=X_train.iloc[:1],
            signature=signature
        )

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_run_id = run.info.run_id

# Register best model
if best_run_id:
    model_uri = f"runs:/{best_run_id}/{best_model_name}"
    registered_model_name = "iris-best-model"

    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    print(f"Registering model from run ID: {best_run_id}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name
    )

    client = MlflowClient()

    # Wait until model is registered before aliasing
    latest_version = result.version

    # Set alias 'champion' to latest version
    client.set_registered_model_alias(
        name=registered_model_name,
        alias="champion",
        version=latest_version
    )

    print(f"Model registered as '{registered_model_name}' and tagged with alias '@champion'.")
