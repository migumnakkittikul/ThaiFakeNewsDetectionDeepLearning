FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libthai0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py model_def.py fake_news_model.pth ./

EXPOSE $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "120", "--workers", "2", "app:app"]