FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_fixed6_prod_ready.py .

# Create directory for data files
RUN mkdir -p /app/data

# Set environment variables for S3
ENV S3_BUCKET_NAME=product-matching-model-bucket
ENV S3_REGION=us-east-1

# Expose port
EXPOSE 8000

# Run the application (S3 download happens at runtime in the Python code)
CMD ["python", "app_fixed6_prod_ready.py"]
