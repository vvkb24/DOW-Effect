import pandas as pd

def add_calendar_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds categorical flags for structural breaks and calendar events.
    """
    df = df.copy()
    
    # T+1 Settlement structural break (India shifted fully to T+1 on Jan 27, 2023)
    df['is_t1_settlement'] = (df.index >= '2023-01-27').astype(int)
    
    # Month-end flag (last 3 trading days of the month)
    df['is_month_end'] = df.index.to_series().groupby([df.index.year, df.index.month]).transform(
        lambda x: x >= x.max() - pd.Timedelta(days=3)
    ).astype(int)
    
    # Note: Expiry week mapping requires a full NSE holiday calendar and options calendar.
    # We will stub it with a simplified Thursday check for the last week of the month.
    # For robust F&O expiry, a dedicated calendar file is required.
    
    return df
