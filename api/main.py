from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
import os
import sqlite3
import time

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from sklearn.datasets import load_iris


# -------------------- Config & constants --------------------

MODEL_NAME = "iris-best-model"
ALIAS = "champion"

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "app.log")
DB_PATH = os.path.join(LOG_DIR, "predictions.db")

iris = load_iris()
CLASS_NAMES = iris.target_names


# -------------------- Logging setup --------------------

os.makedirs(LOG_DIR, exist_ok=True)

LOGGER = logging.getLogger("api")
LOGGER.setLevel(logging.INFO)

_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)
_handler.setFormatter(_formatter)

if not LOGGER.handlers:
    LOGGER.addHandler(_handler)


# -------------------- SQLite helpers --------------------

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            sepal_length REAL,
            sepal_width REAL,
            petal_length REAL,
            petal_width REAL,
            predicted_class TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_row(sl: float, sw: float, pl: float, pw: float, pred: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        (
            "INSERT INTO requests "
            "(ts, sepal_length, sepal_width, petal_length, petal_width, "
            "predicted_class) VALUES (datetime('now'), ?, ?, ?, ?, ?)"
        ),
        (sl, sw, pl, pw, pred),
    )
    conn.commit()
    conn.close()


# -------------------- Prometheus metrics --------------------

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)


# -------------------- FastAPI app (lifespan) --------------------

class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)


model = None  # set in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    init_db()
    LOGGER.info("Startup: model loaded and SQLite initialized.")
    try:
        yield
    finally:
        LOGGER.info("Shutdown: app stopping.")


app = FastAPI(title="Iris Classifier API", lifespan=lifespan)


# -------------------- Middleware --------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.time() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(
            endpoint=endpoint, method=request.method, status=status
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)


# -------------------- Routes --------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {"message": "Iris model is ready for prediction."}


@app.post("/predict")
def predict(input_data: IrisInput):
    df = pd.DataFrame([input_data.model_dump()])
    pred = model.predict(df)
    class_name = CLASS_NAMES[int(pred[0])]

    LOGGER.info("INPUT=%s PRED=%s", input_data.model_dump(), class_name)
    insert_row(
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
        class_name,
    )

    return {
        "input_features": input_data.model_dump(),
        "predicted_class": class_name,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
