import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from models.schemas import GameSituation, PlayPrediction, WinProbability

logger = logging.getLogger(__name__)

class PlayPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'down', 'distance', 'yard_line', 'score_differential',
            'time_remaining', 'field_position_category', 'quarter',
            'red_zone', 'two_minute_warning', 'goal_to_go'
        ]
        self.play_types = [
            'Run - Inside Zone',
            'Run - Outside Zone',
            'Run - Draw Play',
            'Pass - Quick Slant',
            'Pass - Screen',
            'Pass - Deep Post',
            'Pass - Crossing Route',
            'Field Goal Attempt',
            'Punt',
            'Spike/Kneel'
        ]
       
    def _engineer_features(self, game_situation: GameSituation) -> Dict:
        features = {}
        features['down'] = game_situation.down or 1
        features['distance'] = game_situation.distance or 10
        features['yard_line'] = game_situation.yard_line or 50
        features['quarter'] = 4 
        features['score_differential'] = game_situation.score_home - game_situation.score_away
        features['time_remaining'] = (game_situation.time_minutes * 60) + game_situation.time_seconds
        features['field_position_category'] = self._get_field_position_category(features['yard_line'])
        features['red_zone'] = 1 if features['yard_line'] <= 20 else 0
        features['two_minute_warning'] = 1 if features['time_remaining'] <= 120 else 0
        features['goal_to_go'] = 1 if features['distance'] >= features['yard_line'] else 0
       
        return features
   
    def _get_field_position_category(self, yard_line: int) -> int:
        if yard_line <= 20:
            return 0
        elif yard_line <= 40:
            return 1
        elif yard_line <= 60:
            return 2
        else:
            return 3
   
    def train_model(self, training_data: pd.DataFrame):
        logger.info("Training play prediction model...")
       
        try:
            X = training_data[self.feature_columns]
            y = training_data['play_type']

            X = X.fillna(X.mean())
           
            X_scaled = self.scaler.fit_transform(X)

            y_encoded = self.label_encoder.fit_transform(y)
           
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
           
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
           
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
           
            lr_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
           
            self.model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('xgb', xgb_model),
                    ('lr', lr_model)
                ],
                voting='soft'
            )
           
            self.model.fit(X_train, y_train)
           
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
           
            logger.info(f"Model training completed. Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
           
            cv_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=5)
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
           
            return {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
           
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
   
    def predict_plays(self, game_situation: GameSituation) -> List[PlayPrediction]:
        if not self.model:
            raise ValueError("Model not trained or loaded")
       
        try:
            features = self._engineer_features(game_situation)
           
            feature_vector = np.array([[features[col] for col in self.feature_columns]])
           
            feature_vector_scaled = self.scaler.transform(feature_vector)
           
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            play_predictions = []
            for i, prob in enumerate(probabilities):
                if prob > 0.01:
                    play_type = self.label_encoder.inverse_transform([i])[0]
                   
                    expected_yards = self._calculate_expected_yards(play_type, features)
                    success_prob = self._calculate_success_probability(play_type, features)
                   
                    play_predictions.append(PlayPrediction(
                        play_type=play_type,
                        probability=float(prob),
                        description=self._get_play_description(play_type, features),
                        success_probability=success_prob,
                        expected_yards=expected_yards
                    ))
            play_predictions.sort(key=lambda x: x.probability, reverse=True)
           
            return play_predictions[:5]
           
        except Exception as e:
            logger.error(f"Play prediction failed: {e}")
            raise
   
    def _calculate_expected_yards(self, play_type: str, features: Dict) -> float:
        base_yards = {
            'Run - Inside Zone': 3.2,
            'Run - Outside Zone': 3.8,
            'Run - Draw Play': 4.1,
            'Pass - Quick Slant': 5.8,
            'Pass - Screen': 4.2,
            'Pass - Deep Post': 12.5,
            'Pass - Crossing Route': 8.3,
            'Field Goal Attempt': 0.0,
            'Punt': 0.0,
            'Spike/Kneel': 0.0
        }.get(play_type, 4.0)

        if features['red_zone']:
            base_yards *= 0.8
       
        if features['down'] >= 3:
            base_yards *= 1.1
       
        return round(base_yards, 1)
   
    def _calculate_success_probability(self, play_type: str, features: Dict) -> float:
        base_success = {
            'Run - Inside Zone': 0.62,
            'Run - Outside Zone': 0.58,
            'Run - Draw Play': 0.65,
            'Pass - Quick Slant': 0.71,
            'Pass - Screen': 0.68,
            'Pass - Deep Post': 0.42,
            'Pass - Crossing Route': 0.64,
            'Field Goal Attempt': 0.84,
            'Punt': 0.95,
            'Spike/Kneel': 0.99
        }.get(play_type, 0.60)
        if features['distance'] <= 3:
            base_success *= 1.15
        elif features['distance'] >= 10:
            base_success *= 0.85
       
        if features['red_zone']:
            base_success *= 0.9
       
        return min(0.99, round(base_success, 2))
   
    def _get_play_description(self, play_type: str, features: Dict) -> str:
        descriptions = {
            'Run - Inside Zone': f"Rushing attempt up the middle on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Run - Outside Zone': f"Outside rushing attempt on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Run - Draw Play': f"Draw play on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Pass - Quick Slant': f"Quick passing play on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Pass - Screen': f"Screen pass on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Pass - Deep Post': f"Deep passing attempt on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Pass - Crossing Route': f"Intermediate passing play on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Field Goal Attempt': f"Field goal attempt from the {features['yard_line']} yard line",
            'Punt': f"Punt on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}",
            'Spike/Kneel': f"Clock management play"
        }
       
        return descriptions.get(play_type, f"Play attempt on {features['down']}{self._get_down_suffix(features['down'])} and {features['distance']}")
   
    def _get_down_suffix(self, down: int) -> str:
        if down == 1:
            return "st"
        elif down == 2:
            return "nd"
        elif down == 3:
            return "rd"
        else:
            return "th"
   
    def get_confidence_score(self, game_situation: GameSituation) -> float:
        if not self.model:
            return 0.0
       
        try:
            features = self._engineer_features(game_situation)
            feature_vector = np.array([[features[col] for col in self.feature_columns]])
            feature_vector_scaled = self.scaler.transform(feature_vector)
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            confidence = 1.0 - (entropy / max_entropy)
           
            return round(confidence, 3)
           
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
   
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
   
    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._create_fallback_model()
   
    def _create_fallback_model(self):
        logger.info("Creating fallback model...")
        self.model = "fallback"


class WinProbabilityCalculator:
   
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'score_differential', 'time_remaining', 'field_position',
            'down', 'distance', 'timeouts_home', 'timeouts_away'
        ]
   
    def calculate_win_probability(self, game_situation: GameSituation) -> WinProbability:
        """Calculate win probability for current game state"""
        try:
            features = self._engineer_win_prob_features(game_situation)
           
            if self.model and self.model != "fallback":
                feature_vector = np.array([[features[col] for col in self.feature_columns]])
                feature_vector_scaled = self.scaler.transform(feature_vector)
                home_win_prob = float(self.model.predict_proba(feature_vector_scaled)[0][1])
            else:
                home_win_prob = self._calculate_simple_win_prob(features)
           
            away_win_prob = 1.0 - home_win_prob
           
            factors = self._identify_win_prob_factors(features, home_win_prob)
           
            return WinProbability(
                home_win_prob=round(home_win_prob, 3),
                away_win_prob=round(away_win_prob, 3),
                factors=factors
            )
           
        except Exception as e:
            logger.error(f"Win probability calculation failed: {e}")
            return WinProbability(
                home_win_prob=0.5,
                away_win_prob=0.5,
                factors=["Unable to calculate factors"]
            )
   
    def _engineer_win_prob_features(self, game_situation: GameSituation) -> Dict:
        """Engineer features for win probability calculation"""
        features = {}
       
        features['score_differential'] = game_situation.score_home - game_situation.score_away
        features['time_remaining'] = (game_situation.time_minutes * 60) + game_situation.time_seconds
        features['field_position'] = game_situation.yard_line or 50
        features['down'] = game_situation.down or 1
        features['distance'] = game_situation.distance or 10
        features['timeouts_home'] = 3
        features['timeouts_away'] = 3
       
        return features
   
    def _calculate_simple_win_prob(self, features: Dict) -> float:
        """Simple win probability calculation"""
        base_prob = 0.5
        score_diff = features['score_differential']
        base_prob += (score_diff * 0.05)
       
        time_remaining = features['time_remaining']
        if time_remaining < 120:
            if score_diff > 0:
                base_prob += 0.1
            else:
                base_prob -= 0.1
       
        # Field position impact
        field_pos = features['field_position']
        if field_pos < 30:
            base_prob += 0.05
        return max(0.01, min(0.99, base_prob))
   
    def _identify_win_prob_factors(self, features: Dict, home_win_prob: float) -> List[str]:
        """Identify key factors affecting win probability"""
        factors = []
       
        score_diff = features['score_differential']
        time_remaining = features['time_remaining']
       
        if abs(score_diff) > 7:
            leader = "Home" if score_diff > 0 else "Away"
            factors.append(f"{leader} team leads by {abs(score_diff)} points")
       
        if time_remaining < 300:
            factors.append("Game in final 5 minutes")
       
        if features['field_position'] < 20:
            factors.append("Team in red zone scoring position")
       
        if features['down'] >= 3:
            factors.append("Critical down situation")
       
        if not factors:
            factors.append("Game situation relatively neutral")
       
        return factors[:3]
   
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Win probability model saved to {filepath}")
   
    def load_model(self, filepath: str):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info(f"Win probability model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load win probability model: {e}")
            self.model = "fallback"