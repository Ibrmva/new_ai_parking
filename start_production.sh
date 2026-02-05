#!/bin/bash
# Production startup script for LPR API with Celery Beat
# Usage: ./start.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting LPR API Production Setup...${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${YELLOW}Redis not running. Starting Redis...${NC}"
    redis-server --daemonize yes
    echo -e "${GREEN}Redis started.${NC}"
else
    echo -e "${GREEN}Redis is already running.${NC}"
fi

# Check Redis connectivity
echo -e "${GREEN}Checking Redis connectivity...${NC}"
if redis-cli ping | grep -q "PONG"; then
    echo -e "${GREEN}Redis is ready!${NC}"
else
    echo -e "${RED}Redis is not responding. Please check Redis installation.${NC}"
    exit 1
fi

# Start Celery worker with beat scheduler
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Celery worker with beat scheduler...${NC}"
echo -e "${GREEN}========================================${NC}"

# Kill any existing celery processes
pkill -f "celery.*lpr.app.celery_app" 2>/dev/null || true
sleep 2

# Start new Celery worker with beat
celery -A lpr.app.celery_app worker --beat \
    --loglevel=INFO --concurrency=2 \
    --pidfile=/tmp/celery.pid --logfile=/tmp/celery.log

echo -e "${GREEN}Celery worker started with beat scheduler${NC}"
echo -e "${GREEN}Logs available at: /tmp/celery.log${NC}"

# Give Celery time to initialize
echo -e "${YELLOW}Waiting for Celery to initialize...${NC}"
sleep 5

# Check Celery is running
if ps aux | grep -v grep | grep -q "celery.*worker"; then
    echo -e "${GREEN}Celery worker is running!${NC}"
else
    echo -e "${RED}Celery worker failed to start. Check /tmp/celery.log${NC}"
    exit 1
fi

# Start FastAPI server
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo -e "${GREEN}========================================${NC}"

python api.py

