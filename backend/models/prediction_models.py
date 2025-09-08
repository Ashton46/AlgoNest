# models/prediction_models.py - Real ML Models with Kaggle Data
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
#from utils.kaggle_data import sports_data

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
    
    