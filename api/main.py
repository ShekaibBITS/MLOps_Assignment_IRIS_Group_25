from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
class_names = iris.target_names

MODEL_NAME = "iris-best-model"
ALIAS = "champion"

app = FastAPI()
model = None  # will be set on startup

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    except Exception as e:
        # Fail fast with a clear message (surfaces in docker logs)
        raise RuntimeError(
            f"Failed to load MLflow model '{MODEL_NAME}@{ALIAS}'. "
            f"Check MLFLOW_TRACKING_URI/REGISTRY_URI and that the alias exists. Error: {e}"
        )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/")
def read_root():
    return {"message": "Iris model is ready for prediction."}

@app.post("/predict")
def predict(input_data: IrisInput):
    df = pd.DataFrame([input_data.model_dump()])
    pred = model.predict(df)
    class_name = class_names[int(pred[0])]
    return {"input_features": input_data.model_dump(), "predicted_class": class_name}
