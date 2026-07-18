import os
import pandas as pd
from doweffect.data.downloader import Downloader

def test_downloader_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    audit_dir = tmp_path / "audit"
    downloader = Downloader(cache_dir=str(cache_dir), audit_dir=str(audit_dir))
    
    # Create fake cached data
    symbol = "TEST.NS"
    start = "2020-01-01"
    end = "2020-01-10"
    
    df_fake = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    cache_path_adj = downloader._get_cache_path(f"{symbol}_adj", start, end)
    cache_path_raw = downloader._get_cache_path(f"{symbol}_raw", start, end)
    
    os.makedirs(cache_dir, exist_ok=True)
    df_fake.to_parquet(cache_path_adj)
    df_fake.to_parquet(cache_path_raw)
    
    df_adj, df_raw = downloader.download_ticker(symbol, start, end)
    
    assert not df_adj.empty
    assert len(df_adj) == 2
    assert df_adj.iloc[0]["Close"] == 100

def test_downloader_failure_logging(tmp_path):
    cache_dir = tmp_path / "cache"
    audit_dir = tmp_path / "audit"
    downloader = Downloader(cache_dir=str(cache_dir), audit_dir=str(audit_dir))
    
    # Use a symbol that will definitely fail
    symbol = "INVALID_SYMBOL_THAT_DOES_NOT_EXIST"
    df_adj, df_raw = downloader.download_ticker(symbol, "2020-01-01", "2020-01-10")
    
    assert df_adj.empty
    assert df_raw.empty
    
    failures_file = os.path.join(audit_dir, "download_failures.csv")
    assert os.path.exists(failures_file)
    
    df_failures = pd.read_csv(failures_file)
    assert len(df_failures) > 0
    assert df_failures.iloc[0]["symbol"] == symbol
    assert df_failures.iloc[0]["source"] == "yfinance"
