import pandas as pd
import numpy as np
from doweffect.data.auditor import Auditor

def test_auditor(tmp_path):
    audit_dir = tmp_path / "audit"
    auditor = Auditor(audit_dir=str(audit_dir))
    
    # Create fake data
    dates = pd.date_range(start="2020-01-01", periods=10, freq="B")
    df_adj = pd.DataFrame({"Close": [100]*10}, index=dates)
    
    # Inject bad data
    df_adj.iloc[3, 0] = -50  # Negative price
    df_adj.iloc[5, 0] = 200  # Extreme return (100% up)
    
    # Add duplicate index
    df_adj = pd.concat([df_adj, df_adj.iloc[[0]]])
    df_adj = df_adj.sort_index()
    
    res = auditor.audit_ticker("TEST", df_adj, df_adj)
    
    assert res["symbol"] == "TEST"
    assert res["zero_prices"] >= 1
    assert res["extreme_returns"] >= 1
    assert res["duplicates"] == 1
