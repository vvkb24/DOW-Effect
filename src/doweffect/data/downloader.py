"""
Module for downloading financial data with fallback mechanisms, retries, and caching.
"""
import os
import time
import hashlib
import logging
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self, cache_dir: str = "data/cache/", audit_dir: str = "data/audit/"):
        self.cache_dir = cache_dir
        self.audit_dir = audit_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.audit_dir, exist_ok=True)
        self.failures_file = os.path.join(self.audit_dir, "download_failures.csv")

    def _get_cache_path(self, symbol: str, start: str, end: Optional[str]) -> str:
        string_to_hash = f"{symbol}_{start}_{end}"
        hash_val = hashlib.sha256(string_to_hash.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_val}.parquet")

    def _log_failure(self, symbol: str, source: str, error_msg: str):
        file_exists = os.path.exists(self.failures_file)
        with open(self.failures_file, "a") as f:
            if not file_exists:
                f.write("timestamp,symbol,source,error_msg\n")
            # Replace commas and newlines in error_msg to prevent csv breaking
            error_msg = error_msg.replace(",", ";").replace("\n", " ")
            f.write(f"{pd.Timestamp.now()},{symbol},{source},{error_msg}\n")

    def _download_yfinance(self, symbol: str, start: str, end: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ticker = yf.Ticker(symbol)
        
        # Download adjusted
        df_adj = ticker.history(start=start, end=end, auto_adjust=True)
        # Download unadjusted (for corporate actions like splits/dividends)
        df_raw = ticker.history(start=start, end=end, auto_adjust=False)
        
        if df_adj.empty or df_raw.empty:
            raise ValueError(f"No data fetched from yfinance for {symbol}")
            
        return df_adj, df_raw

    def download_ticker(self, symbol: str, start: str, end: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download ticker data with caching and fallback.
        Returns Tuple of (adjusted_dataframe, unadjusted_dataframe).
        """
        cache_path_adj = self._get_cache_path(f"{symbol}_adj", start, end)
        cache_path_raw = self._get_cache_path(f"{symbol}_raw", start, end)
        
        if os.path.exists(cache_path_adj) and os.path.exists(cache_path_raw):
            logger.info(f"Loading {symbol} from cache")
            return pd.read_parquet(cache_path_adj), pd.read_parquet(cache_path_raw)
            
        # Try yfinance with retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                df_adj, df_raw = self._download_yfinance(symbol, start, end)
                
                # Make index timezone naive if it is timezone aware
                for df in [df_adj, df_raw]:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

                df_adj.to_parquet(cache_path_adj)
                df_raw.to_parquet(cache_path_raw)
                return df_adj, df_raw
            except Exception as e:
                logger.warning(f"yfinance failed for {symbol} on attempt {attempt+1}: {e}")
                if attempt == max_attempts - 1:
                    self._log_failure(symbol, "yfinance", str(e))
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
            
        # Fallback to other sources (stubbed out for now)
        # TODO: Implement stooq/alpha_vantage fallback
        
        # If all fail, return empty dataframes so pipeline doesn't terminate
        return pd.DataFrame(), pd.DataFrame()
