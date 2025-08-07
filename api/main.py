from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

# Load the model from MLflow Model Registry
MODEL_NAME = "iris-best-model"
STAGE = "Production"  # or "Staging"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")

app = FastAPI()

# Input schema for prediction
class IrisInput(BaseModel):
    features: list  # e.g., [5.1, 3.5, 1.4, 0.2]

@app.get("/")
def read_root():
    return {"message": "Iris model is ready for prediction."}

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        df = pd.DataFrame([input_data.features])
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
