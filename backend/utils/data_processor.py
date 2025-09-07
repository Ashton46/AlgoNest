from models.schemas import GameSituation
from typing import Dict
import random
from datetime import datetime, timedelta

def process_game_situation(game_situation: GameSituation) -> Dict:
    features = {}
    
    features['down'] = game_situation.down or 1
    features['distance'] = game_situation.distance or 10
    features['yard_line'] = game_situation.yard_line or 50
    features['quarter'] = 4
    
    features['score_home'] = game_situation.score_home
    features['score_away'] = game_situation.score_away
    features['score_differential'] = game_situation.score_home - game_situation.score_away
    
    features['time_minutes'] = game_situation.time_minutes
    features['time_seconds'] = game_situation.time_seconds
    features['time_remaining'] = (game_situation.time_minutes * 60) + game_situation.time_seconds
    
    features['field_position_category'] = _categorize_field_position(features['yard_line'])
    features['red_zone'] = 1 if features['yard_line'] <= 20 else 0
    features['two_minute_warning'] = 1 if features['time_remaining'] <= 120 else 0
    features['goal_to_go'] = 1 if features['distance'] >= features['yard_line'] else 0
    features['short_yardage'] = 1 if features['distance'] <= 3 else 0
    features['long_yardage'] = 1 if features['distance'] >= 10 else 0
    
    features['close_game'] = 1 if abs(features['score_differential']) <= 7 else 0
    features['late_game'] = 1 if features['time_remaining'] <= 300 else 0
    features['desperation_time'] = 1 if features['time_remaining'] <= 120 and features['score_differential'] < 0 else 0
    
    return features

def _categorize_field_position(yard_line: int) -> int:
    if yard_line <= 20:
        return 0
    elif yard_line <= 40:
        return 1
    elif yard_line <= 60:
        return 2
    else:
        return 3

def generate_sample_historical_data(limit: int = 20, sport: str = "football"):
    historical_data = []
    base_accuracy = 85.0
    base_confidence = 90.0 
    
    for i in range(limit):
        date = datetime.now() - timedelta(days=i)
        
        accuracy_variation = random.uniform(-8, 12)
        confidence_variation = random.uniform(-6, 8)
        
        is_weekend = date.weekday() >= 5
        base_predictions = 120 if is_weekend else 60
        prediction_variation = random.randint(-30, 50)
        
        historical_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "accuracy": max(70.0, min(98.0, base_accuracy + accuracy_variation)),
            "confidence": max(75.0, min(96.0, base_confidence + confidence_variation)),
            "total_predictions": max(10, base_predictions + prediction_variation),
            "sport": sport
        })
    
    return historical_data

def calculate_trend_metrics(historical_data: list) -> dict:
    if len(historical_data) < 2:
        return {"trend": "stable", "change": 0.0}
    
    recent_week = historical_data[:7]
    previous_week = historical_data[7:14] if len(historical_data) >= 14 else []
    
    if not previous_week:
        return {"trend": "insufficient_data", "change": 0.0}
    
    recent_avg = sum(d["accuracy"] for d in recent_week) / len(recent_week)
    previous_avg = sum(d["accuracy"] for d in previous_week) / len(previous_week)
    
    change = recent_avg - previous_avg
    
    if change > 2.0:
        trend = "improving"
    elif change < -2.0:
        trend = "declining"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "change": round(change, 2),
        "recent_average": round(recent_avg, 1),
        "previous_average": round(previous_avg, 1)
    }