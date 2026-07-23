import os
import boto3
from pathlib import Path

MODEL_PATH = Path("artifacts/training/model.h5")
S3_BUCKET = os.environ.get("MODEL_S3_BUCKET")
S3_KEY = os.environ.get("MODEL_S3_KEY", "model.h5")

def download_model():
    if MODEL_PATH.exists():
        print("Model already present locally, skipping download.")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from s3://{S3_BUCKET}/{S3_KEY} ...")
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_KEY, str(MODEL_PATH))
    print("Model download complete.")

if __name__ == "__main__":
    download_model()