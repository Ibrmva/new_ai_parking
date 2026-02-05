from celery import Celery
from lpr.app.config import settings
import logging

logger = logging.getLogger(__name__)

celery_app = Celery(
    "lpr_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["lpr.app.tasks.celery_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Bishkek",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.TASK_TIME_LIMIT,
    task_soft_time_limit=settings.TASK_TIME_LIMIT - 60,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)

if settings.ENABLE_AUTO_PROCESSING:
    poll_interval = float(settings.POLLING_INTERVAL_MINUTES * 60)
    
    if poll_interval <= 0:
        celery_app.conf.beat_schedule = {}
        logger.info("Event-driven mode - webhooks only")
    else:
        celery_app.conf.beat_schedule = {
            'process-remote-images-every-second': {
                'task': 'lpr.app.tasks.celery_tasks.process_remote_images_task',
                'schedule': poll_interval,
                'options': {
                    'expires': max(poll_interval - 0.5, 0.5),
                    'time_limit': min(poll_interval, 60)
                }
            },
        }
        logger.info(f"Auto-processing enabled - polling every {poll_interval} seconds")
else:
    celery_app.conf.beat_schedule = {}
    logger.info("Auto-processing disabled")

logger.info(f"Celery configured - Broker: {settings.CELERY_BROKER_URL}")