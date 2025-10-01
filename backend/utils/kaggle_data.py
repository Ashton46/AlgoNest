import pandas as pd
import numpy as np
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from decouple import config
import logging
from typing import Dict, Tuple
import zipfile
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class SportsDataLoader:
    def __init__(self):
        self.data_dir = Path("data")
        self.initialized = False
        self.datasets = {
            "football": "maxhorowitz/nflplaybyplay2009to2016",
            "basketball": "nathanlauga/nba-games"
        }
        self.cache = {}
        
    def initialize_kaggle(self):
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            self.initialized = True
            logger.info("Kaggle API authenticated")
            return True
        except Exception as e:
            logger.warning(f"Kaggle API failed: {e}")
            return False
    
    def download_dataset(self, sport: str, force_download: bool = False) -> bool:
        """Simply download the dataset - no data processing"""
        if not self.initialized and not self.initialize_kaggle():
            return False
        
        dataset_name = self.datasets.get(sport)
        if not dataset_name:
            logger.error(f"Unknown dataset: {sport}")
            return False
        
        target_dir = self.data_dir / sport
        target_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(target_dir.glob("*.csv"))
        if not force_download and csv_files:
            logger.info(f"{sport} data already exists")
            return True
        
        try:
            logger.info(f"⬇Downloading {sport} dataset...")
            self.api.dataset_download_files(
                dataset_name,
                path=str(target_dir),
                unzip=False,
                quiet=False
            )

            time.sleep(3)

            zip_files = list(target_dir.glob("*.zip"))
            if not zip_files:
                logger.error("No zip file downloaded")
                return False
            
            zip_path = zip_files[0]

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                logger.info("Successfully extracted zip file")

                zip_path.unlink()
                
            except zipfile.BadZipFile:
                logger.error("Corrupted zip file")
                return self._handle_manual_extraction(zip_path, target_dir, sport)
            
            csv_files = list(target_dir.glob("*.csv"))
            if csv_files:
                logger.info(f"Downloaded {len(csv_files)} CSV files")
                return True
            else:
                logger.error("No CSV files after extraction")
                all_files = list(target_dir.glob("*"))
                logger.info(f"Extracted files: {[f.name for f in all_files]}")
                return False
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _handle_manual_extraction(self, zip_path: Path, target_dir: Path, sport: str) -> bool:
        """Handle problematic zip files"""
        try:
            import tarfile
            try:
                with tarfile.open(zip_path, 'r') as tar_ref:
                    tar_ref.extractall(target_dir)
                logger.info("Extracted with tarfile")
            except:
                import subprocess
                result = subprocess.run(['unzip', '-o', str(zip_path), '-d', str(target_dir)], 
                                    capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("Extracted with system unzip")
                else:
                    logger.error(f"All extraction methods failed: {result.stderr}")
                    return False
            
            zip_path.unlink()
            return True
        
        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")
            return False

    def get_training_data(self, sport: str, force_redownload: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Simply return the raw data - let the model handle processing"""
        if sport in self.cache and not force_redownload:
            return self.cache[sport]
        
        if force_redownload or not list((self.data_dir / sport).glob("*.csv")):
            self.download_dataset(sport, force_download=force_redownload)
        
        data_path = self.data_dir / sport
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            logger.error(f"No CSV files found for {sport}")
            return pd.DataFrame(), {"sport": sport, "status": "no_files"}
        
        try:
            main_file = max(csv_files, key=lambda x: x.stat().st_size)
            logger.info(f" Loading {sport} data from: {main_file.name}")
            
            data = pd.read_csv(main_file, low_memory=False)
            
            data_info = {
                "sport": sport,
                "source": "kaggle", 
                "status": "loaded",
                "records_loaded": len(data),
                "file": main_file.name
            }
            
            self.cache[sport] = (data, data_info)
            return data, data_info
            
        except Exception as e:
            logger.error(f"Error loading {sport} data: {e}")
            return pd.DataFrame(), {"sport": sport, "status": "load_error"}

sports_data = SportsDataLoader()
