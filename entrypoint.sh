#!/bin/bash
# Production entrypoint script - runs both FastAPI and Celery Beat

set -e

echo "=========================================="
echo "Starting LPR API Production Environment"
echo "=========================================="

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Start Celery worker with beat scheduler in background
echo "Starting Celery worker with beat scheduler..."
celery -A lpr.app.celery_app worker --beat \
    --loglevel=INFO --concurrency=2 \
    --pidfile=/tmp/celery.pid --logfile=/tmp/celery.log &

CELERY_PID=$!
echo "Celery worker started (PID: $CELERY_PID)"

# Wait for celery to initialize
sleep 3

# Start FastAPI server
echo "Starting FastAPI server..."
exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4 --timeout-keep-alive 30

