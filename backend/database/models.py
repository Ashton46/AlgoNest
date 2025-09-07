from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func
from .connection import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(36), unique=True, index=True, nullable=False)
    
    sport = Column(String(20), index=True, nullable=False)
    home_team = Column(String(50))
    away_team = Column(String(50))
    score_home = Column(Integer)
    score_away = Column(Integer)
    
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)
    model_version = Column(String(20), nullable=False)
    
    game_situation = Column(Text)
    predictions = Column(Text)
    
    actual_outcome = Column(String(100))
    feedback_correct = Column(Boolean)
    feedback_received_at = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, sport={self.sport}, confidence={self.confidence})>"

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(20), nullable=False, index=True)
    sport = Column(String(20), nullable=False, index=True)
    
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    
    sample_size = Column(Integer, default=0)
    date_range_start = Column(DateTime)
    date_range_end = Column(DateTime)
    
    recorded_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_version}, metric={self.metric_name}, value={self.metric_value})>"

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(36), nullable=False, index=True)
    
    actual_play_type = Column(String(100), nullable=False)
    was_prediction_correct = Column(Boolean, nullable=False)
    confidence_rating = Column(Integer)
    
    user_notes = Column(Text)
    user_id = Column(String(36))

    submitted_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<UserFeedback(prediction_id={self.prediction_id}, correct={self.was_prediction_correct})>"