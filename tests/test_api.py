# tests/test_api.py
import os
os.environ["USE_LOCAL_MODEL"] = "true"  # noqa: E402

from fastapi.testclient import TestClient # noqa: E402
from api.main import app # noqa: E402


def test_health():
    # Use context manager so FastAPI lifespan (startup) runs
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True


def test_predict_shape():
    with TestClient(app) as client:
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert "predicted_class" in r.json()
