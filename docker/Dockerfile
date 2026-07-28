# Production-Grade Sandbox and Execution Docker Configuration

# Use python-3.12-slim base
FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements and python installation setup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and configurations
COPY agent/ ./agent/
COPY tests/ ./tests/

# Default entry
CMD ["python", "-m", "agent.cli"]
