# Use latest maintained Python 3.9 slim image (based on Debian Bookworm or Bullseye)
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install awscli using pip (no apt issues)
RUN pip install --no-cache-dir awscli

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# S3 config for downloading the trained model at startup
ENV MODEL_S3_BUCKET=brain-tumor-model-akash
ENV MODEL_S3_KEY=model.h5
ENV AWS_DEFAULT_REGION=ap-south-1

# Run the application
CMD ["python3", "app.py"]