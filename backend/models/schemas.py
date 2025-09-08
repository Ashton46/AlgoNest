from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class SportType(str, Enum):
    football = "football"
    basketball = "basketball"

class GameSituation(BaseModel):
    sport: SportType
    score_home: int = Field(..., ge=0, le=150)
    score_away: int = Field(..., ge=0, le=150)
    time_minutes: int = Field(..., ge=0, le=15)
    time_seconds: int = Field(..., ge=0, le=59)
    down: Optional[int] = Field(None, ge=1, le=4)
    distance: Optional[int] = Field(None, ge=1, le=99)
    yard_line: Optional[int] = Field(None, ge=1, le=99)
    quarter: Optional[int] = Field(None, ge=1, le=4)
    shot_clock: Optional[int] = Field(None, ge=0, le=24)

    @property
    def score_differential(self) -> int:
        return self.score_home - self.score_away

    @property
    def time_remaining(self) -> int:
        return self.time_minutes * 60 + self.time_seconds

    @property
    def total_time_seconds(self) -> int:
        if self.sport == SportType.football:
            return (4 - self.quarter) * 900 + self.time_remaining if self.quarter else 0
        else:  # basketball
            return (4 - self.quarter) * 720 + self.time_remaining if self.quarter else 0

class PlayPrediction(BaseModel):
    play_type: str
    probability: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None
    expected_yards: Optional[float] = None

class WinProbability(BaseModel):
    home_win_prob: float = Field(..., ge=0.0, le=1.0)
    away_win_prob: float = Field(..., ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    predictions: List[PlayPrediction]
    win_probability: Optional[WinProbability] = None
    confidence: float
    processing_time: str
    model_version: str
    timestamp: str
    prediction_id: Optional[str] = None

class HistoricalData(BaseModel):
    date: str
    accuracy: float
    confidence: float
    total_predictions: int

class MetricsResponse(BaseModel):
    model_accuracy: str
    average_confidence: str
    average_processing_time: str
    total_predictions: str
    uptime: str
    last_updated: str

class TrainingResponse(BaseModel):
    status: str
    sport: str
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None
    message: Optional[str] = None