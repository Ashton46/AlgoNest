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
