import pandas as pd
import numpy as np
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from decouple import config
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class SportsDataLoader:
    def __init__(self):
        self.data_dir = "data"
        self.initialized = False
        self.datasets = {
            "football": "maxhorowitz/nflplaybyplay2009to2016",
            "basketball": "wyattowalsh/basketball"
        }
        self.cache = {}
        
    def initialize_kaggle(self):
        try:
            os.environ['KAGGLE_USERNAME'] = config('KAGGLE_USERNAME', default='')
            os.environ['KAGGLE_KEY'] = config('KAGGLE_KEY', default='')
            
            self.api = KaggleApi()
            self.api.authenticate()
            self.initialized = True
            logger.info("✅ Kaggle API authenticated")
            return True
        except Exception as e:
            logger.warning(f"❌ Kaggle API failed: {e}")
            self.initialized = False
            return False
    
    def download_dataset(self, sport: str, force_download: bool = False) -> bool:
        if not self.initialized and not self.initialize_kaggle():
            return False
        
        dataset_name = self.datasets.get(sport)
        if not dataset_name:
            logger.error(f"❌ Unknown dataset: {sport}")
            return False
        
        target_dir = os.path.join(self.data_dir, sport)
        os.makedirs(target_dir, exist_ok=True)
        
        if not force_download and os.path.exists(os.path.join(target_dir, ".downloaded")):
            logger.info(f"📁 {sport} data already downloaded")
            return True
        
        try:
            logger.info(f"⬇️ Downloading {sport} dataset...")
            self.api.dataset_download_files(
                dataset_name,
                path=target_dir,
                unzip=True,
                quiet=False
            )
            
            with open(os.path.join(target_dir, ".downloaded"), 'w') as f:
                f.write("downloaded")
            
            logger.info(f"✅ {sport} dataset downloaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def get_training_data(self, sport: str, force_retrain: bool = False) -> Tuple[pd.DataFrame, Dict]:
        if sport in self.cache and not force_retrain:
            return self.cache[sport]
        
        data_info = {"sport": sport, "source": "Kaggle", "status": "loaded"}
        
        try:
            if sport == "football":
                data = self._load_nfl_data()
            elif sport == "basketball":
                data = self._load_nba_data()
            else:
                return pd.DataFrame(), {**data_info, "status": "unsupported_sport"}
            
            if data.empty:
                logger.warning(f"⚠️ No data for {sport}, using synthetic")
                data = self._create_synthetic_data(sport)
                data_info["status"] = "synthetic"
            
            self.cache[sport] = (data, data_info)
            return data, data_info
            
        except Exception as e:
            logger.error(f"❌ Error loading {sport} data: {e}")
            return self._create_synthetic_data(sport), {**data_info, "status": "error_fallback"}
    def _load_nfl_data(self) -> pd.DataFrame:
        data_path = os.path.join(self.data_dir, "football")
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("⚠️ No NFL CSV files found")
            return pd.DataFrame()
        
        main_file = next((f for f in csv_files if 'play' in f.lower() or 'pbp' in f.lower()), csv_files[0])
        
        try:
            file_path = os.path.join(data_path, main_file)
            logger.info(f"📖 Reading NFL data from: {file_path}")
            
            data = pd.read_csv(file_path, low_memory=False, nrows=20000)
            logger.info(f"📊 Loaded {len(data):,} NFL plays")
            
            if 'play_type' in data.columns:
                data = data[data['play_type'].notna()]
                logger.info(f"🔢 {len(data):,} plays after cleaning")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error reading NFL data: {e}")
            return pd.DataFrame()
    
    def _load_nba_data(self) -> pd.DataFrame:
        data_path = os.path.join(self.data_dir, "basketball")
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not csv_files:
            logger.warning("⚠️ No NBA CSV files found")
            return pd.DataFrame()
        
        shot_file = next((f for f in csv_files if 'shot' in f.lower()), None)
        game_file = next((f for f in csv_files if 'game' in f.lower()), None)
        
        try:
            if shot_file:
                file_path = os.path.join(data_path, shot_file)
                logger.info(f"📖 Reading NBA shot data from: {file_path}")
                data = pd.read_csv(file_path, low_memory=False, nrows=15000)
                logger.info(f"🏀 Loaded {len(data):,} NBA shots")
                return data
            elif game_file:
                file_path = os.path.join(data_path, game_file)
                logger.info(f"📖 Reading NBA game data from: {file_path}")
                data = pd.read_csv(file_path, low_memory=False, nrows=5000)
                logger.info(f"🏀 Loaded {len(data):,} NBA games")
                return data
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error reading NBA data: {e}")
            return pd.DataFrame()
    
    def _create_synthetic_data(self, sport: str) -> pd.DataFrame:
        logger.info(f"🎭 Creating synthetic data for {sport}")
        
        if sport == "football":
            return self._create_synthetic_nfl_data()
        elif sport == "basketball":
            return self._create_synthetic_nba_data()
        return pd.DataFrame()
    
    def _create_synthetic_nfl_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n_samples = 8000
        
        data = pd.DataFrame({
            'down': np.random.randint(1, 5, n_samples),
            'ydstogo': np.random.randint(1, 20, n_samples),
            'yardline_100': np.random.randint(1, 100, n_samples),
            'quarter': np.random.randint(1, 6, n_samples),
            'score_diff': np.random.randint(-21, 21, n_samples),
        })
        
        data['play_type'] = np.where(
            (data['down'] == 4) & (data['yardline_100'] > 70), 'punt',
            np.where(
                (data['down'] == 4) & (data['yardline_100'] <= 30), 'field_goal',
                np.where(
                    (data['down'] == 1) | ((data['down'] == 2) & (data['ydstogo'] <= 5)),
                    np.where(np.random.random(n_samples) < 0.6, 'run', 'pass'),
                    np.where(np.random.random(n_samples) < 0.7, 'pass', 'run')
                )
            )
        )
        
        return data
    
    def _create_synthetic_nba_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n_samples = 10000
        
        data = pd.DataFrame({
            'quarter': np.random.randint(1, 6, n_samples),
            'shot_distance': np.random.uniform(0, 30, n_samples),
            'score_diff': np.random.randint(-30, 30, n_samples),
        })

        data['shot_type'] = np.where(
            data['shot_distance'] >= 23.75, '3PT',
            np.where(data['shot_distance'] <= 8, 'Paint', 'MidRange')
        )
        
        return data

sports_data = SportsDataLoader()

   