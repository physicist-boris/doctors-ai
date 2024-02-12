import os


# NOTE(Participant): This will eventually change
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_API_HOST = os.getenv("MINIO_API_HOST", "localhost:9000")
SERVER_API_URL = os.getenv("SERVER_API_URL", "http://localhost:3000/predict")