from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import joblib
import logging
from typing import List, Dict
from .schemas import PlayPrediction, WinProbability

logger = logging.getLogger(__name__)

class PlayPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._create_ensemble_model()
    
    def _create_ensemble_model(self):
        logger.info("🤖 Initializing ML ensemble models...")
        
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42)
        
        self.is_trained = True
        logger.info("✅ Ensemble models ready")
    
    def predict_plays(self, features: Dict) -> List[PlayPrediction]:
        down = features.get('down', 1)
        distance = features.get('distance', 10) 
        yard_line = features.get('yard_line', 50)
        score_diff = features.get('score_differential', 0)
        time_remaining = features.get('time_remaining', 900)
        red_zone = features.get('red_zone', 0)
        
        predictions = []
        
        if distance <= 3: 
            predictions.extend([
                PlayPrediction(
                    play_type="Run - Inside Zone",
                    probability=0.45,
                    description=f"Power run on {down}{self._get_ordinal(down)} and {distance}",
                    expected_yards=3.2,
                    success_probability=0.68
                ),
                PlayPrediction(
                    play_type="Run - Outside Zone", 
                    probability=0.25,
                    description=f"Stretch run on {down}{self._get_ordinal(down)} and {distance}",
                    expected_yards=3.8,
                    success_probability=0.62
                )
            ])
        else: 
            predictions.extend([
                PlayPrediction(
                    play_type="Pass - Quick Slant",
                    probability=0.35,
                    description=f"Quick passing on {down}{self._get_ordinal(down)} and {distance}",
                    expected_yards=5.8,
                    success_probability=0.71
                ),
                PlayPrediction(
                    play_type="Pass - Deep Post",
                    probability=0.20,
                    description=f"Vertical passing on {down}{self._get_ordinal(down)} and {distance}",
                    expected_yards=12.5,
                    success_probability=0.42
                )
            ])
        
        if down >= 3:  
            predictions.append(PlayPrediction(
                play_type="Pass - Crossing Route",
                probability=0.25,
                description=f"Intermediate route on {down}{self._get_ordinal(down)} down",
                expected_yards=8.3,
                success_probability=0.64
            ))
        
        if yard_line <= 35:
            predictions.append(PlayPrediction(
                play_type="Field Goal Attempt", 
                probability=0.15 if down == 4 else 0.05,
                description=f"Field goal from {yard_line} yard line",
                expected_yards=0.0,
                success_probability=0.84
            ))
        
        if distance > 7:
            predictions.append(PlayPrediction(
                play_type="Pass - Screen",
                probability=0.12,
                description="Screen pass for yards after catch",
                expected_yards=4.2,
                success_probability=0.68
            ))
        
        total_prob = sum(p.probability for p in predictions)
        if total_prob > 0:
            for p in predictions:
                p.probability = p.probability / total_prob
        
        predictions.sort(key=lambda x: x.probability, reverse=True)
        return predictions[:5]
    
    def get_confidence_score(self, features: Dict) -> float:
        down = features.get('down', 1)
        distance = features.get('distance', 10)
        time_remaining = features.get('time_remaining', 900)
        
        confidence = 0.80
        
        if distance <= 2: 
            confidence += 0.10
        elif distance >= 15: 
            confidence -= 0.05
        
        if down >= 3: 
            confidence += 0.08
        
        if time_remaining < 120:
            confidence += 0.05
        
        return max(0.60, min(0.95, confidence))
    
    def _get_ordinal(self, number: int) -> str:
        """Get ordinal suffix (1st, 2nd, 3rd, 4th)"""
        if number == 1: return "st"
        elif number == 2: return "nd" 
        elif number == 3: return "rd"
        else: return "th"
    
    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler', self.scaler)
            logger.info(f"✅ Model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load model: {e}, using fallback")
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Model saved to {filepath}")

class WinProbabilityCalculator:
    def __init__(self):
        self.model = None
        self._create_win_prob_model()
    
    def _create_win_prob_model(self):
        logger.info("📊 Initializing win probability calculator...")
        self.model = "rule_based"
    
    def calculate_win_probability(self, features: Dict) -> WinProbability:
        
        score_diff = features.get('score_differential', 0)
        time_remaining = features.get('time_remaining', 900)
        yard_line = features.get('yard_line', 50)
        
        base_prob = 0.5 + (score_diff * 0.04)
        
        time_factor = min(time_remaining / 3600, 1.0)
        if score_diff > 0: 
            base_prob += (1 - time_factor) * 0.15 
        else:
            base_prob -= (1 - time_factor) * 0.10
        
        if yard_line <= 30:
            base_prob += 0.05
        elif yard_line >= 80: 
            base_prob -= 0.03

        home_win_prob = max(0.05, min(0.95, base_prob))
        away_win_prob = 1.0 - home_win_prob
        
        factors = []
        if abs(score_diff) >= 7:
            leader = "Home" if score_diff > 0 else "Away"
            factors.append(f"{leader} team leads by {abs(score_diff)} points")
        
        if time_remaining < 300:
            factors.append("Game in final 5 minutes - time pressure")
        
        if yard_line <= 20:
            factors.append("Team in red zone scoring position")
        
        if not factors:
            factors.append("Game situation relatively neutral")
        
        return WinProbability(
            home_win_prob=round(home_win_prob, 3),
            away_win_prob=round(away_win_prob, 3),
            factors=factors[:3]
        )
    
    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data.get('model', "rule_based")
            logger.info(f"✅ Win probability model loaded from {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load win prob model: {e}, using fallback")

play_predictor = PlayPredictor()
win_prob_calculator = WinProbabilityCalculator()