from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Plate(Base):
    __tablename__ = "plates"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, nullable=True)
    filename = Column(String(255), nullable=False)
    detected_text = Column(String(255), nullable=False)
    confidence = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    track_id = Column(Integer)
    camera_id = Column(String(255))
    image = Column(LONGBLOB)
    bbox = Column(String(255))
