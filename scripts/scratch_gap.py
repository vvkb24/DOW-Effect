import pandas as pd
import numpy as np

def analyze_weekend_gap(parquet_path):
    df = pd.read_parquet(parquet_path)
    
    # Ensure datetime index and sort
    df = df.sort_index()
    
    # Add Day of week
    df['DayOfWeek'] = df.index.dayofweek
    
    # We want to compare Monday's Open to the preceding Friday's Close.
    # A robust way is to just find all Mondays, and then find the last available trading day before that Monday.
    # Usually, that's Friday. If Friday is a holiday, it might be Thursday. 
    # The user specifically asked "ended on friday evening", so let's strictly look at Friday Close.
    
    fridays = df[df['DayOfWeek'] == 4].copy()
    mondays = df[df['DayOfWeek'] == 0].copy()
    
    # Shift Fridays to match the next Monday date 
    # (Friday + 3 days = Monday)
    fridays['Next_Monday'] = fridays.index + pd.Timedelta(days=3)
    
    # Set index to Next_Monday to align with Mondays
    fridays_shifted = fridays.reset_index().set_index('Next_Monday')
    
    # Join with Mondays on the index
    # Inner join ensures we only count weeks where BOTH Friday and Monday were trading days.
    merged = mondays.join(fridays_shifted, lsuffix='_Mon', rsuffix='_Fri', how='inner')
    
    total_valid_weekends = len(merged)
    
    if total_valid_weekends == 0:
        print("No valid Friday-Monday pairs found.")
        return
        
    # Monday Open > Friday Close
    gapped_up = merged['Open_Mon'] > merged['Close_Fri']
    
    count_gap_up = gapped_up.sum()
    probability = count_gap_up / total_valid_weekends
    
    # Also calculate the average size of the gap
    gap_pct = (merged['Open_Mon'] - merged['Close_Fri']) / merged['Close_Fri']
    
    print(f"Total Valid Friday-Monday Pairs: {total_valid_weekends}")
    print(f"Number of times Monday Open > Friday Close: {count_gap_up}")
    print(f"Historical Probability (Percentage): {probability * 100:.2f}%")
    print(f"Average Gap Size: {gap_pct.mean() * 100:.4f}%")
    print(f"Median Gap Size: {gap_pct.median() * 100:.4f}%")

if __name__ == "__main__":
    analyze_weekend_gap("data/processed/NTPC.NS.parquet")
