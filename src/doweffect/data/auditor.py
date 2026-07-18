"""
Data auditor for checking missing values, duplicates, extreme returns, and holidays.
"""
import pandas as pd
import numpy as np
import logging
import os
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)

class Auditor:
    def __init__(self, audit_dir: str = "data/audit/"):
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)
        self.audit_report_path = os.path.join(self.audit_dir, "audit_report.csv")
        self.bad_obs_path = os.path.join(self.audit_dir, "bad_observations.csv")
        
        # Load NSE calendar
        self.nse = mcal.get_calendar("NSE")
        
    def audit_ticker(self, symbol: str, df_adj: pd.DataFrame, df_raw: pd.DataFrame) -> dict:
        if df_adj.empty:
            return {"symbol": symbol, "status": "NO_DATA", "total_days": 0}
            
        # Ensure index is datetime and sorted
        df_adj = df_adj.sort_index()
        
        # Make index timezone naive if it is timezone aware
        if df_adj.index.tz is not None:
            df_adj.index = df_adj.index.tz_localize(None)
        
        # 1. Duplicates
        duplicates = df_adj.index.duplicated().sum()
        
        # 2. Missing days (vs NSE calendar)
        start_date = df_adj.index.min()
        end_date = df_adj.index.max()
        
        # NSE calendar schedule
        schedule = self.nse.schedule(start_date=start_date, end_date=end_date)
        expected_days = mcal.date_range(schedule, frequency='1D')
        
        # Make sure both are tz-naive
        if expected_days.tz is not None:
            expected_days = expected_days.tz_convert(None)
            
        actual_days = df_adj.index.normalize()
        expected_days = expected_days.normalize()
        
        missing_days = len(set(expected_days) - set(actual_days))
        
        # 3. Extreme returns
        returns = df_adj["Close"].pct_change()
        extreme_returns = (returns.abs() > 0.25).sum()
        
        # 4. Zero/negative prices
        zero_prices = (df_adj["Close"] <= 0).sum()
        
        # 5. Stale prices
        stale_prices = (df_adj["Close"] == df_adj["Close"].shift(1)).rolling(3).sum()
        stale_instances = (stale_prices >= 3).sum()
        
        # Missing Mondays specifically
        actual_mondays = actual_days[actual_days.dayofweek == 0]
        expected_mondays = expected_days[expected_days.dayofweek == 0]
        missing_mondays = len(set(expected_mondays) - set(actual_mondays))
        
        return {
            "symbol": symbol,
            "status": "OK",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "total_days": len(df_adj),
            "duplicates": duplicates,
            "missing_days": missing_days,
            "missing_mondays": missing_mondays,
            "extreme_returns": extreme_returns,
            "zero_prices": zero_prices,
            "stale_instances": stale_instances
        }
