import os
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool
from alembic import context
from lpr.app.models import Base

load_dotenv()

DATABASE_URL = f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'lpr.db')}"

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)
target_metadata = Base.metadata
