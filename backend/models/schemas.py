from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum

class SportType(str, Enum):
    football = "football"
    basketball = "basketball"

class GameSituation(BaseModel):
    sport: SportType = Field(..., description="Type of sport")

    score_home: int = Field(..., ge=0, le=150, description="Home team score")
    score_away: int = Field(..., ge=0, le=150, description="Away team score")
    time_minutes: int = Field(..., ge=0, le=15, description="Minutes remaining in quarter")
    time_seconds: int = Field(..., ge=0, le=59, description="Seconds remaining in quarter")

    down: Optional[int] = Field(None, ge=1, le=4, description="Current down (1-4)")
    distance: Optional[int] = Field(None, ge=1, le=99, description="Yards to first down")
    yard_line: Optional[int] = Field(None, ge=1, le=99, description="Current yard line position")
    
    quarter: Optional[int] = Field(None, ge=1, le=4, description="Current quarter")
    shot_clock: Optional[int] = Field(None, ge=0, le=24, description="Shot clock seconds remaining")
    
    home_team: Optional[str] = Field(None, description="Home team name/abbreviation")
    away_team: Optional[str] = Field(None, description="Away team name/abbreviation")
    weather: Optional[str] = Field(None, description="Weather conditions (for outdoor sports)")
    temperature: Optional[int] = Field(None, description="Temperature in Fahrenheit")
    wind_speed: Optional[int] = Field(None, description="Wind speed in mph")


    @field_validator('down', 'distance', 'yard_line')
    @classmethod
    def football_fields_required(cls, v, info):
        if info.data.get('sport') == SportType.football:
            if v is None:
                raise ValueError('This field is required for football predictions')
        return v
    
    @field_validator('quarter')
    @classmethod
    def basketball_fields_validation(cls, v, info):
        if info.data.get('sport') == SportType.basketball:
            if v is None:
                raise ValueError('Quarter is required for basketball predictions')
        return v

class PlayPrediction(BaseModel):
    play_type: str = Field(..., description="Type of play predicted")
    probability: float = Field(..., ge=0.0, le=1.1, description="Probability of this play (0-1)")
    description: Optional[str] = Field(None, description="Human-readable description")
    success_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability this play succeeds")
    expected_yards: Optional[float] = Field(None, description="Expected yards gained/lost")

class WinProbability(BaseModel):
    home_win_prob: float = Field(..., ge=0.0, le=1.0, description="Home team win probability")
    away_win_prob: float = Field(..., ge=0.0, le=1.0, description="Away team win probability")
    factors: Optional[List[str]] = Field(None, description="Key factors influencing win probability")

class PredictionResponse(BaseModel):
    predictions: List[PlayPrediction] = Field(..., description="List of play predictions")
    win_probability: Optional[WinProbability] = Field(None, description="Win probability analysis")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall model confidence")
    processing_time: str = Field(..., description="Time taken to process prediction")
    model_version: str = Field(..., description="Version of the prediction model")
    timestamp: str = Field(..., description="When the prediction was made")
    prediction_id: Optional[str] = Field(None, description="Unique identifier for this prediction")

class HistoricalData(BaseModel):
    date: str = Field(..., description="Date of the data point")
    accuracy: float = Field(..., ge=0.0, le=100.0, description="Model accuracy percentage")
    confidence: float = Field(..., ge=0.0, le=100.0, description="Average confidence percentage")
    total_predictions: int = Field(..., ge=0, description="Number of predictions made")
    sport: Optional[str] = Field(None, description="Sport type for this data")

class MetricsResponse(BaseModel):
    model_accuracy: str = Field(..., description="Overall model accuracy")
    average_confidence: str = Field(..., description="Average prediction confidence")
    average_processing_time: str = Field(..., description="Average time per prediction")
    total_predictions: str = Field(..., description="Total predictions made")
    uptime: str = Field(..., description="System uptime percentage")
    last_updated: str = Field(..., description="When metrics were last calculated")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())