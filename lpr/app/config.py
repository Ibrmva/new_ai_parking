import os
from typing import Optional
from pathlib import Path

class Settings:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    EXTERNAL_API: str = os.getenv("EXTERNAL_API", "")
    PLATE_IMAGE_URLL: str = os.getenv("PLATE_IMAGE_URLL", "")
    PLATE_RECOGNITION_ADD_URL: str = os.getenv("PLATE_RECOGNITION_ADD_URL", "")
    EXTERNAL_RECOGNITION_URL: str = os.getenv("EXTERNAL_RECOGNITION_URL", "")
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    POLLING_INTERVAL_MINUTES: float = float(os.getenv("POLLING_INTERVAL_MINUTES", "0"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_SECONDS: int = int(os.getenv("RETRY_DELAY_SECONDS", "60"))
    ENABLE_AUTO_PROCESSING: bool = os.getenv("ENABLE_AUTO_PROCESSING", "true").lower() == "true"
    
    WORKER_CONCURRENCY: int = int(os.getenv("WORKER_CONCURRENCY", "2"))
    TASK_TIME_LIMIT: int = int(os.getenv("TASK_TIME_LIMIT", "1800"))
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/lpr.db")
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "celery.log"))

settings = Settings()