import asyncio
import logging
from celery import Task
from lpr.app.celery_app import celery_app
from lpr.app.config import settings

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    _db = None
    
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        if self._db is not None:
            try:
                self._db.close()
            except:
                pass


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="lpr.app.tasks.celery_tasks.process_remote_images_task",
    max_retries=1,
    default_retry_delay=1,
    autoretry_for=(Exception,),
    retry_backoff=False,
    time_limit=60,
    soft_time_limit=50
)
def process_remote_images_task(self):
    try:
        from lpr.app.routers.external import process_remote_images_logic
        
        result = asyncio.run(process_remote_images_logic())
        
        if result.get('total', 0) > 0:
            logger.info(f"Processed {result.get('total', 0)} images")
        
        return result
        
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        else:
            return {"total": 0, "processed": [], "error": str(e)}