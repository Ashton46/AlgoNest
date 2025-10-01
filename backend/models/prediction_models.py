from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
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
                logger.error(f"No data for {sport}")
                self.is_trained[sport] = False
                return {"status": "no_data"}
            
            logger.info(f"Raw data shape: {training_data.shape}")
            logger.info(f"Columns: {list(training_data.columns)}")
            
            # PREPARE FEATURES AND TARGET
            X, y, feature_names = self._prepare_features(sport, training_data)
            
            if len(X) == 0:
                logger.error(f"Could not prepare features for {sport}")
                self.is_trained[sport] = False
                return {"status": "feature_error"}
            
            logger.info(f"Final feature matrix: {X.shape}")
            logger.info(f"Target classes: {np.unique(y)}")
            
            if len(X) < 100:
                logger.error(f"Insufficient data: {len(X)} samples")
                self.is_trained[sport] = False
                return {"status": "insufficient_data"}
            
            # SETUP MODEL COMPONENTS
            self.feature_names[sport] = feature_names
            self.scalers[sport] = StandardScaler()
            self.encoders[sport] = LabelEncoder()
            
            # ENCODE TARGET VARIABLE
            y_encoded = self.encoders[sport].fit_transform(y)
            logger.info(f"🔤 Encoded classes: {list(self.encoders[sport].classes_)}")
            
            # SPLIT DATA
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            logger.info(f"📚 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # SCALE FEATURES
            X_train_scaled = self.scalers[sport].fit_transform(X_train)
            X_test_scaled = self.scalers[sport].transform(X_test)
            
            # CREATE AND TRAIN MODEL
            self.models[sport] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("Training model...")
            self.models[sport].fit(X_train_scaled, y_train)
            logger.info("Model training completed!")
            
            y_pred = self.models[sport].predict(X_test_scaled)
            train_accuracy = accuracy_score(y_train, self.models[sport].predict(X_train_scaled))
            test_accuracy = accuracy_score(y_test, y_pred)
            
            class_report = classification_report(y_test, y_pred, 
                                               target_names=self.encoders[sport].classes_,
                                               output_dict=True)
            
            self.is_trained[sport] = True
            self.model_stats[sport] = {
                "status": "success",
                "sport": sport,
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "features_used": feature_names,
                "feature_importance": dict(zip(feature_names, 
                                            self.models[sport].feature_importances_)),
                "class_distribution": dict(zip(self.encoders[sport].classes_, 
                                             np.bincount(y_encoded))),
                "classification_report": class_report
            }
            
            logger.info(f"🎯 {sport} model trained successfully!")
            logger.info(f"📊 Train Accuracy: {train_accuracy:.3f}")
            logger.info(f"📊 Test Accuracy: {test_accuracy:.3f}")
            
            return self.model_stats[sport]
            
        except Exception as e:
            logger.error(f"❌ {sport} training failed: {e}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            self.is_trained[sport] = False
            return {"status": "error", "error": str(e)}
    
    def _prepare_features(self, sport: str, data: pd.DataFrame) -> tuple:
        """Prepare features from raw Kaggle data"""
        logger.info(f"🔧 Preparing features for {sport}...")
        
        feature_data = data.copy()
        
        if sport == "football":
            return self._prepare_nfl_features(feature_data)
        elif sport == "basketball":
            return self._prepare_nba_features(feature_data)
        else:
            return np.array([]), np.array([]), []
    
    def _prepare_nfl_features(self, data: pd.DataFrame) -> tuple:
        """Extract NFL features from raw data"""
        column_mapping = {
            'down': ['down', 'Down', 'dn'],
            'distance': ['ydstogo', 'yrdstogo', 'Distance', 'dist'],
            'yardline': ['yrdline100', 'yardline_100', 'YardLine', 'yl'],
            'quarter': ['qtr', 'quarter', 'Quarter', 'period'],
            'score_diff': ['score_differential', 'ScoreDiff', 'diff'],
            'play_type': ['play_type', 'PlayType', 'type', 'desc']
        }
        
        # Find available columns
        available_features = {}
        for feature, possible_names in column_mapping.items():
            for name in possible_names:
                if name in data.columns:
                    available_features[feature] = name
                    break
        
        logger.info(f"🔍 Found NFL columns: {available_features}")
        
        # Create feature matrix
        feature_columns = []
        feature_data = data.copy()
        
        # Map and clean features
        if 'down' in available_features:
            feature_data['down'] = pd.to_numeric(data[available_features['down']], errors='coerce').fillna(1)
            feature_columns.append('down')
        
        if 'distance' in available_features:
            feature_data['distance'] = pd.to_numeric(data[available_features['distance']], errors='coerce').fillna(10)
            feature_data['distance'] = feature_data['distance'].clip(1, 30)
            feature_columns.append('distance')
        
        if 'yardline' in available_features:
            feature_data['yardline'] = pd.to_numeric(data[available_features['yardline']], errors='coerce').fillna(50)
            feature_data['yardline'] = feature_data['yardline'].clip(1, 100)
            feature_columns.append('yardline')
        
        if 'quarter' in available_features:
            feature_data['quarter'] = pd.to_numeric(data[available_features['quarter']], errors='coerce').fillna(1)
            feature_data['quarter'] = feature_data['quarter'].clip(1, 4)
            feature_columns.append('quarter')
        
        if 'score_diff' in available_features:
            feature_data['score_diff'] = pd.to_numeric(data[available_features['score_diff']], errors='coerce').fillna(0)
            feature_data['score_diff'] = feature_data['score_diff'].clip(-35, 35)
            feature_columns.append('score_diff')
        
        # Create derived features
        if 'yardline' in feature_data.columns:
            feature_data['red_zone'] = (feature_data['yardline'] <= 20).astype(int)
            feature_columns.append('red_zone')
        
        if 'distance' in feature_data.columns:
            feature_data['short_yardage'] = (feature_data['distance'] <= 3).astype(int)
            feature_columns.append('short_yardage')
        
        if 'quarter' in feature_data.columns:
            feature_data['fourth_quarter'] = (feature_data['quarter'] == 4).astype(int)
            feature_columns.append('fourth_quarter')
        
        # Extract target variable
        if 'play_type' in available_features:
            target_col = available_features['play_type']
            feature_data['play_type'] = data[target_col]
            
            # Clean play types
            feature_data['play_type'] = feature_data['play_type'].astype(str).str.lower()
            
            # Map to common play types
            play_mapping = {
                'run': ['run', 'rush', 'running'],
                'pass': ['pass', 'throw', 'passing'],
                'punt': ['punt', 'punting'],
                'field_goal': ['field goal', 'fg', 'field_goal', 'kick']
            }
            
            def map_play_type(play_desc):
                play_desc = str(play_desc).lower()
                for play_type, keywords in play_mapping.items():
                    if any(keyword in play_desc for keyword in keywords):
                        return play_type
                return 'pass'  # default
            
            feature_data['play_type'] = feature_data['play_type'].apply(map_play_type)
        
        # Filter to valid data
        valid_plays = ['run', 'pass', 'punt', 'field_goal']
        if 'play_type' in feature_data.columns:
            feature_data = feature_data[feature_data['play_type'].isin(valid_plays)]
            feature_data = feature_data.dropna(subset=['play_type'])
        
        if len(feature_data) == 0:
            logger.error("❌ No valid play types found")
            return np.array([]), np.array([]), []
        
        X = feature_data[feature_columns].fillna(0).values
        y = feature_data['play_type'].values
        
        logger.info(f"🏈 NFL features: {len(feature_columns)}, samples: {len(X)}")
        logger.info(f"🎯 Play distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y, feature_columns
    
    def _prepare_nba_features(self, data: pd.DataFrame) -> tuple:
        """Extract NBA features from raw data"""
        column_mapping = {
            'quarter': ['quarter', 'qtr', 'period', 'Period'],
            'shot_distance': ['shot_distance', 'distance', 'shot_dist', 'dist'],
            'score_diff': ['score_diff', 'score_differential', 'point_diff', 'diff'],
            'shot_type': ['shot_type', 'shot_type', 'type', 'action_type']
        }
        
        available_features = {}
        for feature, possible_names in column_mapping.items():
            for name in possible_names:
                if name in data.columns:
                    available_features[feature] = name
                    break
        
        logger.info(f"🔍 Found NBA columns: {available_features}")
        
        feature_columns = []
        feature_data = data.copy()
        
        # Basic features
        if 'quarter' in available_features:
            feature_data['quarter'] = pd.to_numeric(data[available_features['quarter']], errors='coerce').fillna(1)
            feature_data['quarter'] = feature_data['quarter'].clip(1, 4)
            feature_columns.append('quarter')
        
        if 'shot_distance' in available_features:
            feature_data['shot_distance'] = pd.to_numeric(data[available_features['shot_distance']], errors='coerce').fillna(15)
            feature_data['shot_distance'] = feature_data['shot_distance'].clip(0, 30)
            feature_columns.append('shot_distance')
        
        if 'score_diff' in available_features:
            feature_data['score_diff'] = pd.to_numeric(data[available_features['score_diff']], errors='coerce').fillna(0)
            feature_data['score_diff'] = feature_data['score_diff'].clip(-30, 30)
            feature_columns.append('score_diff')
        
        # Derived features
        if 'shot_distance' in feature_data.columns:
            feature_data['paint_shot'] = (feature_data['shot_distance'] <= 8).astype(int)
            feature_columns.append('paint_shot')
            feature_data['three_point_shot'] = (feature_data['shot_distance'] >= 23.75).astype(int)
            feature_columns.append('three_point_shot')
        
        if 'quarter' in feature_data.columns:
            feature_data['fourth_quarter'] = (feature_data['quarter'] == 4).astype(int)
            feature_columns.append('fourth_quarter')
        
        # Target variable
        if 'shot_type' in available_features:
            feature_data['shot_type'] = data[available_features['shot_type']]
        else:
            # Infer from distance
            feature_data['shot_type'] = np.where(
                feature_data.get('shot_distance', 15) >= 23.75, '3PT',
                np.where(feature_data.get('shot_distance', 15) <= 8, 'Paint', 'MidRange')
            )
        
        # Filter valid shots
        valid_shots = ['3PT', 'Paint', 'MidRange', '2PT', 'LAYUP', 'DUNK']
        feature_data['shot_type'] = feature_data['shot_type'].astype(str).str.upper()
        feature_data = feature_data[feature_data['shot_type'].isin(valid_shots)]
        
        # Standardize shot types
        shot_mapping = {
            '2PT': 'MidRange',
            'LAYUP': 'Paint', 
            'DUNK': 'Paint'
        }
        feature_data['shot_type'] = feature_data['shot_type'].replace(shot_mapping)
        
        if len(feature_data) == 0:
            logger.error("No valid shot types found")
            return np.array([]), np.array([]), []
        
        X = feature_data[feature_columns].fillna(0).values
        y = feature_data['shot_type'].values
        
        logger.info(f"NBA features: {len(feature_columns)}, samples: {len(X)}")
        logger.info(f"Shot distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y, feature_columns
    
    def predict_plays(self, sport: str, features: Dict) -> List[PlayPrediction]:
        if self.is_trained.get(sport, False) and sport in self.models:
            return self._predict_with_ml(sport, features)
        else:
            logger.info(f"Using rule-based fallback for {sport}")
            return self._predict_with_rules(sport, features)
    
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
            logger.error(f"ML prediction failed: {e}")
            return self._predict_with_rules(sport, features)


predictor = UnifiedSportsPredictor()