"""
Data validator and cleaner.
"""
import pandas as pd
import os
import logging
import hashlib

logger = logging.getLogger(__name__)

class Validator:
    def __init__(self, processed_dir: str = "data/processed/"):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def clean_and_save(self, symbol: str, df: pd.DataFrame) -> str:
        if df.empty:
            return ""
            
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Filter zero/negative prices
        df = df[df["Close"] > 0]
        
        # Save to processed
        out_path = os.path.join(self.processed_dir, f"{symbol}.parquet")
        df.to_parquet(out_path)
        
        # Return hash
        with open(out_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            
        return file_hash
