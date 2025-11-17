# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install spatial-lab in editable mode
RUN pip install -e .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for API (future)
EXPOSE 8000

# Default command
CMD ["python", "-m", "spatial_lab.experiment_runner", "--help"]
