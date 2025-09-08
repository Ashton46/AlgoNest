import pandas as pd
import numpy as np
import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from decouple import config
import logging
from typing import Dict, Tuple
import zipfile

logger = logging.getLogger(__name__)

class SportsDataLoader:
    def __init__(self):
        self.data_dir = "data"
        self.initialized = False
        self.datasets = {
            "football": "maxhorowitz/nflplaybyplay2009to2016",
            "basketball": "nathanlao/nba-basketball-database"
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
        """Get training data for specific sport"""
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
    
   