from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from .connection import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(String(36), unique=True)
    sport = Column(String(20))
    confidence = Column(Float)
    processing_time = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)