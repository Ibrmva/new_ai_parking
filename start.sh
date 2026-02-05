#!/bin/bash
# Production startup script for LPR API with Celery Beat
# Usage: ./start.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 

echo -e "${GREEN}Starting LPR API Production Setup...${NC}"

if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${YELLOW}Redis not running. Starting Redis...${NC}"
    redis-server --daemonize yes
    echo -e "${GREEN}Redis started.${NC}"
else
    echo -e "${GREEN}Redis is already running.${NC}"
fi

echo -e "${GREEN}Starting Celery worker with beat scheduler...${NC}"
celery -A lpr.app.celery_app worker --beat \
    --loglevel=INFO --concurrency=2 --detach --pidfile=/tmp/celery.pid --logfile=/tmp/celery.log

echo -e "${GREEN}Celery worker started (check /tmp/celery.log)${NC}"

echo -e "${GREEN}Starting FastAPI server...${NC}"
python api.py