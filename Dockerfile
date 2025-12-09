# Dockerfile - production-ready for Render (binds to $PORT)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install system deps required by OpenCV/ffmpeg/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Create outputs folder (if your app writes outputs)
RUN mkdir -p /app/outputs

# Expose port (documentation only — Render uses $PORT)
EXPOSE 10000

# Use shell form so $PORT env expands at runtime
CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:${PORT:-10000} --workers 2 --timeout 120"]
