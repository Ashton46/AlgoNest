from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
   
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sport = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(JSON, nullable=False)
    predictions = Column(JSON, nullable=False)
    win_probability = Column(JSON, nullable=True)
    confidence = Column(Float, default=0.0)
    processing_time_ms = Column(Integer, default=0)
    model_version = Column(String(50), default="v1.0")
    user_agent = Column(Text, nullable=True)
    client_ip = Column(String(45), nullable=True)
   
    def __repr__(self):
        return f"<PredictionLog(sport='{self.sport}', timestamp='{self.timestamp}')>"

class GameSituation(Base):
    __tablename__ = "game_situations"
   
    id = Column(Integer, primary_key=True)
    sport = Column(String(20), nullable=False)
    situation_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    common_plays = Column(JSON, nullable=False)
    success_rate = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
   
    def __repr__(self):
        return f"<GameSituation(sport='{self.sport}', type='{self.situation_type}')>"

class ModelPerformance(Base):
    __tablename__ = "model_performance"
   
    id = Column(Integer, primary_key=True)
    sport = Column(String(20), nullable=False)
    model_version = Column(String(50), nullable=False)
    accuracy = Column(Float, default=0.0)
    precision = Column(Float, default=0.0)
    recall = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    training_samples = Column(Integer, default=0)
    test_samples = Column(Integer, default=0)
    training_date = Column(DateTime, default=datetime.utcnow)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
   
    def __repr__(self):
        return f"<ModelPerformance(sport='{self.sport}', accuracy='{self.accuracy}')>"