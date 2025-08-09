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

# Copy source code
COPY . .

# Expose port (optional - good practice)
EXPOSE 8000
# Default command to run the API

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
