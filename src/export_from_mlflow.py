# src/export_from_mlflow.py
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.artifacts import download_artifacts

MODEL_NAME = os.getenv("MODEL_NAME", "iris-best-model")
ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_registry_uri(MLFLOW_URI)

    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    print(f"Resolving model from registry: {model_uri}")

    # 1) Download the registered model's artifact directory to a temp dir
    tmp_dir = download_artifacts(model_uri)
    print(f"Downloaded model artifacts to temp dir: {tmp_dir}")

    # 2) Copy to project-local folder exported_model/
    export_dir = Path("exported_model")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(tmp_dir, export_dir)
    print(f"Copied model artifacts to ./{export_dir}")

    # 3) Optional: verify we can load the local pyfunc model folder
    try:
        _ = mlflow.pyfunc.load_model(str(export_dir))
        print("Verified: local pyfunc model loads from ./exported_model")
    except Exception as e:
        raise RuntimeError(
            f"Exported model folder is not loadable as pyfunc: {e}"
        )

if __name__ == "__main__":
    main()