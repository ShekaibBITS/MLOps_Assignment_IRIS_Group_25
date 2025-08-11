# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (optional-sometimes needed by packages like numpy, pandas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \    
    #build-essential gcc curl && \ 
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.api.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.api.txt

#Tell the container how to reach your host MLflow server
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV MLFLOW_REGISTRY_URI=${MLFLOW_TRACKING_URI}

# ---------------- ADDED: toggle + baked model copy ----------------
# Build-time toggle to use local exported model (default false for dev)
# docker build -t iris-api-offline --build-arg USE_LOCAL_MODEL=true ## when running from local model in dev.
ARG USE_LOCAL_MODEL=false
ENV USE_LOCAL_MODEL=${USE_LOCAL_MODEL}

# If you want the image to run offline, make sure exported_model/ exists
# before building. This COPY ensures it's inside the image at /app/exported_model
COPY exported_model /app/exported_model
# ------------------------------------------------------------------

# Copy source code
COPY . .

# Expose port (optional - good practice)
EXPOSE 8000
# Default command to run the API

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
