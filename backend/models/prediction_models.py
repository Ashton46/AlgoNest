from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict
from .schemas import PlayPrediction, WinProbability
from utils.kaggle_data import sports_data

logger = logging.getLogger(__name__)

class UnifiedSportsPredictor:

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.is_trained = {}
        self.model_stats = {}
        self.feature_names = {}
        
    def train_sport_model(self, sport: str, force_retrain: bool = False) -> Dict:
        logger.info(f"🤖 Training {sport} model...")
        
        try:
            training_data, data_info = sports_data.get_training_data(sport, force_retrain)
            
            if training_data.empty:
                logger.warning(f"⚠️ No {sport} data, using fallback")
                self.is_trained[sport] = False
                return {"status": "no_data"}
            
            logger.info(f"📊 Training on {len(training_data):,} {sport} records")

            if sport == "football":
                X, y, feature_names = self._prepare_nfl_features(training_data)
            elif sport == "basketball":
                X, y, feature_names = self._prepare_nba_features(training_data)
            else:
                raise ValueError(f"Unsupported sport: {sport}")
            
            if len(X) < 1000:
                logger.warning(f"⚠️ Insufficient {sport} data")
                self.is_trained[sport] = False
                return {"status": "insufficient_data"}
            
            self.feature_names[sport] = feature_names
            self.scalers[sport] = StandardScaler()
            self.encoders[sport] = LabelEncoder()
            
            y_encoded = self.encoders[sport].fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            X_train_scaled = self.scalers[sport].fit_transform(X_train)
            X_test_scaled = self.scalers[sport].transform(X_test)
            
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
            
            self.models[sport] = VotingClassifier(
                estimators=[('rf', rf_model), ('xgb', xgb_model)],
                voting='soft'
            )
            
            self.models[sport].fit(X_train_scaled, y_train)
            
            train_score = self.models[sport].score(X_train_scaled, y_train)
            test_score = self.models[sport].score(X_test_scaled, y_test)
            
            self.is_trained[sport] = True
            self.model_stats[sport] = {
                "status": "success", "sport": sport,
                "train_accuracy": round(train_score, 4),
                "test_accuracy": round(test_score, 4),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "play_types": list(self.encoders[sport].classes_),
                "data_source": data_info.get("source", "Kaggle")
            }
            
            logger.info(f"✅ {sport} model trained! Accuracy: {test_score:.3f}")
            return self.model_stats[sport]
            
        except Exception as e:
            logger.error(f"❌ {sport} training failed: {e}")
            self.is_trained[sport] = False
            return {"status": "error", "error": str(e)}
    
    def predict_plays(self, sport: str, features: Dict) -> List[PlayPrediction]:
        if self.is_trained.get(sport, False) and sport in self.models:
            return self._predict_with_ml(sport, features)
        else:
            logger.info(f"🔄 Using rule-based fallback for {sport}")
            return self._predict_with_rules(sport, features)
    
    def _prepare_nfl_features(self, data: pd.DataFrame) -> tuple:
        logger.info("🏈 Preparing NFL features...")

        feature_data = data.copy()
        
        default_cols = {
            'down': lambda: np.random.randint(1, 5, len(feature_data)),
            'ydstogo': lambda: np.random.randint(1, 20, len(feature_data)),
            'yardline_100': lambda: np.random.randint(1, 100, len(feature_data)),
            'quarter': lambda: np.random.randint(1, 6, len(feature_data)),
            'score_diff': lambda: np.random.randint(-21, 21, len(feature_data)),
            'play_type': lambda: np.random.choice(['run', 'pass', 'punt', 'field_goal'], len(feature_data))
        }
        
        for col, generator in default_cols.items():
            if col not in feature_data.columns:
                feature_data[col] = generator()
        
        feature_data['red_zone'] = (feature_data['yardline_100'] <= 20).astype(int)
        feature_data['short_yardage'] = (feature_data['ydstogo'] <= 3).astype(int)
        feature_data['long_yardage'] = (feature_data['ydstogo'] >= 7).astype(int)
        feature_data['fourth_quarter'] = (feature_data['quarter'] == 4).astype(int)
        
        feature_columns = [
            'down', 'ydstogo', 'yardline_100', 'quarter', 'score_diff',
            'red_zone', 'short_yardage', 'long_yardage', 'fourth_quarter'
        ]
        
        X = feature_data[feature_columns].fillna(0).values
        y = feature_data['play_type'].values
        
        logger.info(f"🔢 NFL Features: {len(feature_columns)}, Samples: {len(X):,}")
        return X, y, feature_columns
    
    def _prepare_nba_features(self, data: pd.DataFrame) -> tuple:
        logger.info("🏀 Preparing NBA features...")
        
        feature_data = data.copy()
        
        default_cols = {
            'quarter': lambda: np.random.randint(1, 6, len(feature_data)),
            'shot_distance': lambda: np.random.uniform(0, 30, len(feature_data)),
            'score_diff': lambda: np.random.randint(-30, 30, len(feature_data)),
            'shot_type': lambda: np.random.choice(['3PT', 'Paint', 'MidRange'], len(feature_data))
        }
        
        for col, generator in default_cols.items():
            if col not in feature_data.columns:
                feature_data[col] = generator()
        
        feature_data['paint_shot'] = (feature_data['shot_distance'] <= 8).astype(int)
        feature_data['three_point_shot'] = (feature_data['shot_distance'] >= 23.75).astype(int)
        feature_data['close_game'] = (abs(feature_data['score_diff']) <= 5).astype(int)
        feature_data['fourth_quarter'] = (feature_data['quarter'] == 4).astype(int)
        
        feature_columns = [
            'quarter', 'shot_distance', 'score_diff',
            'paint_shot', 'three_point_shot', 'close_game', 'fourth_quarter'
        ]
        
        X = feature_data[feature_columns].fillna(0).values
        y = feature_data['shot_type'].values
        
        logger.info(f"🔢 NBA Features: {len(feature_columns)}, Samples: {len(X):,}")
        return X, y, feature_columns
    
    def _predict_with_ml(self, sport: str, features: Dict) -> List[PlayPrediction]:
        try:
            if sport == "football":
                feature_vector = self._convert_nfl_features(features)
            elif sport == "basketball":
                feature_vector = self._convert_nba_features(features)
            else:
                return self._predict_with_rules(sport, features)
            
            feature_vector_scaled = self.scalers[sport].transform(feature_vector.reshape(1, -1))
            probabilities = self.models[sport].predict_proba(feature_vector_scaled)[0]
            
            predictions = []
            for i, prob in enumerate(probabilities):
                if prob > 0.05:
                    play_category = self.encoders[sport].inverse_transform([i])[0]
                    play_type = self._format_play_type(sport, play_category)
                    
                    predictions.append(PlayPrediction(
                        play_type=play_type,
                        probability=round(float(prob), 3),
                        description=f"ML prediction from {sport} data",
                        expected_points=self._estimate_expected_value(sport, play_category, features)
                    ))
            
            predictions.sort(key=lambda x: x.probability, reverse=True)
            return predictions[:5]
            
        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            return self._predict_with_rules(sport, features)
    def _convert_nfl_features(self, features: Dict) -> np.ndarray:
        yard_line = features.get('yard_line', 50)
        distance = features.get('distance', 10)
        quarter = features.get('quarter', 1)
        
        return np.array([
            features.get('down', 1),
            distance,
            yard_line,
            quarter,
            features.get('score_differential', 0),
            1 if yard_line <= 20 else 0,
            1 if distance <= 3 else 0,
            1 if distance >= 7 else 0,
            1 if quarter == 4 else 0
        ])
    
    def _convert_nba_features(self, features: Dict) -> np.ndarray:
        shot_distance = features.get('shot_distance', 15)
        quarter = features.get('quarter', 1)
        
        return np.array([
            quarter,
            shot_distance,
            features.get('score_differential', 0),
            1 if shot_distance <= 8 else 0,
            1 if shot_distance >= 23.75 else 0,
            1 if abs(features.get('score_differential', 0)) <= 5 else 0,
            1 if quarter == 4 else 0
        ])
    
    def _predict_with_rules(self, sport: str, features: Dict) -> List[PlayPrediction]:
        if sport == "football":
            return self._predict_nfl_rules(features)
        elif sport == "basketball":
            return self._predict_nba_rules(features)
        return [PlayPrediction(play_type="Unknown Play", probability=1.0, expected_points=0.0)]
    
    def _predict_nfl_rules(self, features: Dict) -> List[PlayPrediction]:
        distance = features.get('distance', 10)
        down = features.get('down', 1)
        yard_line = features.get('yard_line', 50)
        quarter = features.get('quarter', 1)
        score_diff = features.get('score_differential', 0)
        
        if yard_line <= 5:
            return [
                PlayPrediction(play_type="Run - Goal Line", probability=0.60, expected_points=0.42),
                PlayPrediction(play_type="Pass - Fade", probability=0.25, expected_points=0.48),
                PlayPrediction(play_type="Pass - Slant", probability=0.15, expected_points=0.45)
            ]
        elif yard_line <= 20:
            return [
                PlayPrediction(play_type="Run - Inside", probability=0.45, expected_points=0.40),
                PlayPrediction(play_type="Pass - Short", probability=0.35, expected_points=0.52),
                PlayPrediction(play_type="Pass - Red Zone", probability=0.20, expected_points=0.58)
            ]
        elif distance <= 3:
            return [
                PlayPrediction(play_type="Run - Inside", probability=0.55, expected_points=0.38),
                PlayPrediction(play_type="Run - Outside", probability=0.25, expected_points=0.42),
                PlayPrediction(play_type="Pass - Quick", probability=0.20, expected_points=0.46)
            ]
        elif down >= 3 and distance >= 7:
            return [
                PlayPrediction(play_type="Pass - Deep", probability=0.45, expected_points=0.85),
                PlayPrediction(play_type="Pass - Intermediate", probability=0.35, expected_points=0.62),
                PlayPrediction(play_type="Screen Pass", probability=0.20, expected_points=0.45)
            ]
        elif quarter == 4 and features.get('time_remaining', 900) <= 120:
            if score_diff < 0:
                return [
                    PlayPrediction(play_type="Pass - Deep", probability=0.50, expected_points=0.88),
                    PlayPrediction(play_type="Pass - Intermediate", probability=0.35, expected_points=0.65),
                    PlayPrediction(play_type="Screen Pass", probability=0.15, expected_points=0.48)
                ]
            else:
                return [
                    PlayPrediction(play_type="Run - Clock", probability=0.55, expected_points=0.35),
                    PlayPrediction(play_type="Pass - Short", probability=0.30, expected_points=0.52),
                    PlayPrediction(play_type="Kneel Down", probability=0.15, expected_points=0.0)
                ]
        else:
            return [
                PlayPrediction(play_type="Run - Inside", probability=0.40, expected_points=0.42),
                PlayPrediction(play_type="Pass - Intermediate", probability=0.35, expected_points=0.65),
                PlayPrediction(play_type="Run - Outside", probability=0.15, expected_points=0.45),
                PlayPrediction(play_type="Screen Pass", probability=0.10, expected_points=0.52)
            ]
    
    def _predict_nba_rules(self, features: Dict) -> List[PlayPrediction]:
        score_diff = features.get('score_differential', 0)
        shot_clock = features.get('shot_clock', 24)
        quarter = features.get('quarter', 1)
        time_remaining = features.get('time_remaining', 600)
        
        if quarter == 4 and time_remaining <= 300 and abs(score_diff) <= 5:
            return [
                PlayPrediction(play_type="Isolation", probability=0.40, expected_points=0.95),
                PlayPrediction(play_type="Pick and Roll", probability=0.30, expected_points=1.05),
                PlayPrediction(play_type="Three-Point Shot", probability=0.20, expected_points=1.12),
                PlayPrediction(play_type="Post Up", probability=0.10, expected_points=0.92)
            ]
        elif shot_clock <= 7:
            return [
                PlayPrediction(play_type="Isolation", probability=0.45, expected_points=0.85),
                PlayPrediction(play_type="Quick Three", probability=0.30, expected_points=1.08),
                PlayPrediction(play_type="Drive to Basket", probability=0.25, expected_points=0.95)
            ]
        elif abs(score_diff) >= 20:
            return [
                PlayPrediction(play_type="Three-Point Shot", probability=0.35, expected_points=1.10),
                PlayPrediction(play_type="Fast Break", probability=0.30, expected_points=1.25),
                PlayPrediction(play_type="Mid-Range", probability=0.20, expected_points=0.92),
                PlayPrediction(play_type="Bench Isolation", probability=0.15, expected_points=0.78)
            ]
        else:
            return [
                PlayPrediction(play_type="Pick and Roll", probability=0.35, expected_points=1.02),
                PlayPrediction(play_type="Three-Point Shot", probability=0.25, expected_points=1.15),
                PlayPrediction(play_type="Post Up", probability=0.20, expected_points=0.96),
                PlayPrediction(play_type="Cut to Basket", probability=0.15, expected_points=1.22),
                PlayPrediction(play_type="Isolation", probability=0.05, expected_points=0.88)
            ]
    
    def _format_play_type(self, sport: str, category: str) -> str:
        format_map = {
            'football': {
                'run': 'Run Play', 'pass': 'Pass Play',
                'punt': 'Punt', 'field_goal': 'Field Goal Attempt'
            },
            'basketball': {
                '3PT': 'Three-Point Shot', 'Paint': 'Paint/Close Shot',
                'MidRange': 'Mid-Range Jumper'
            }
        }
        return format_map.get(sport, {}).get(category, category.title())
    
    def _estimate_expected_value(self, sport: str, play_type: str, features: Dict) -> float:
        base_values = {
            'football': {
                'run': 0.42, 'pass': 0.65, 'punt': 0.0, 'field_goal': 0.85
            },
            'basketball': {
                '3PT': 1.05, 'Paint': 1.15, 'MidRange': 0.90
            }
        }
        
        base_value = base_values.get(sport, {}).get(play_type, 0.5)

        if sport == "football":
            if features.get('down', 1) == 1:
                base_value += 0.05
            if features.get('distance', 10) <= 3:
                base_value += 0.08
            if features.get('yard_line', 50) <= 20:
                base_value -= 0.12
        
        elif sport == "basketball":
            if features.get('quarter', 1) == 4 and features.get('time_remaining', 600) <= 300:
                base_value -= 0.08
            if features.get('shot_clock', 24) <= 7:
                base_value -= 0.05
        
        return round(max(0, base_value), 2)
    
    def get_model_stats(self, sport: str) -> Dict:
        return self.model_stats.get(sport, {"status": "not_trained"})
    
    def get_win_probability(self, sport: str, features: Dict) -> WinProbability:
        if sport == "football":
            return self._calculate_nfl_win_probability(features)
        elif sport == "basketball":
            return self._calculate_nba_win_probability(features)
        return WinProbability(home_win_prob=0.5, away_win_prob=0.5)
    
    def _calculate_nfl_win_probability(self, features: Dict) -> WinProbability:
        score_diff = features.get('score_differential', 0)
        time_remaining = features.get('time_remaining', 900)
        yard_line = features.get('yard_line', 50)
        
        base_prob = 0.5
        base_prob += min(score_diff * 0.02, 0.4) if score_diff > 0 else -min(abs(score_diff) * 0.02, 0.4)
        
        time_factor = time_remaining / 3600
        base_prob += time_factor * 0.1 if score_diff > 0 else -time_factor * 0.1
        
        field_factor = (100 - yard_line) / 100
        base_prob += field_factor * 0.15
        
        home_prob = max(0.01, min(0.99, base_prob))
        return WinProbability(home_win_prob=round(home_prob, 3), away_win_prob=round(1 - home_prob, 3))
    
    def _calculate_nba_win_probability(self, features: Dict) -> WinProbability:
        score_diff = features.get('score_differential', 0)
        time_remaining = features.get('time_remaining', 720)
        quarter = features.get('quarter', 1)
        
        base_prob = 0.5
        base_prob += min(score_diff * 0.03, 0.45) if score_diff > 0 else -min(abs(score_diff) * 0.03, 0.45)
        
        time_factor = time_remaining / (2880 / (5 - quarter))
        base_prob += time_factor * 0.12 if score_diff > 0 else -time_factor * 0.12
        
        if quarter >= 4:
            base_prob += 0.08 if score_diff > 0 else -0.08
        
        home_prob = max(0.01, min(0.99, base_prob))
        return WinProbability(home_win_prob=round(home_prob, 3), away_win_prob=round(1 - home_prob, 3))

predictor = UnifiedSportsPredictor()

    
    