FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    build-essential \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

COPY lpr/app ./app
COPY sort ./sort
COPY templates ./templates
COPY alembic ./alembic

COPY lpr.db ./lpr.db


COPY cron_job.sh /app/cron_job.sh
RUN chmod +x /app/cron_job.sh

COPY crontab.txt /etc/cron.d/app-cron
RUN chmod 0644 /etc/cron.d/app-cron
RUN crontab /etc/cron.d/app-cron

EXPOSE 8000

CMD cron && uvicorn app.main:app --host 0.0.0.0 --port 8000
