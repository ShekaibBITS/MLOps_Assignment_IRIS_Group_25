from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
from sklearn.datasets import load_iris

# Load class names from sklearn
iris = load_iris()
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# Load the MLflow registered model using ALIAS (not deprecated stages)
MODEL_NAME = "iris-best-model"
ALIAS = "champion"  # Use the alias set in train.py

# Load the model using the alias
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

app = FastAPI()

# Define input schema with named features
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Iris model is ready for prediction."}

@app.post("/predict")
def predict(input_data: IrisInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.model_dump()])
        prediction = model.predict(df)
        class_index = int(prediction[0])
        class_name = class_names[class_index]
        return {
            "input_features": input_data.model_dump(),
            "predicted_class": class_name
        }
    except Exception as e:
        return {"error": str(e)}
