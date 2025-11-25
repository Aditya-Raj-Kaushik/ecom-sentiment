# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system deps (if needed later you can add build tools here)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + model
COPY src src
COPY models models

# Expose FastAPI port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
