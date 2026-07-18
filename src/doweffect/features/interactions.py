import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_interaction_matrix(df: pd.DataFrame, min_support: int = 30) -> pd.DataFrame:
    """
    Generates explicitly multiplicative interaction terms between Weekday and Regimes.
    Filters out any interaction state with fewer than `min_support` observations to
    prevent rare-event overfitting.
    """
    df = df.copy()
    
    # Base categorical
    weekdays = pd.get_dummies(df['DayOfWeek'], prefix='day')
    df = pd.concat([df, weekdays], axis=1)
    
    # Define our binary regime columns (must end in int 0/1)
    regime_cols = [c for c in df.columns if c.startswith('regime_') or c.startswith('is_')]
    
    interaction_cols = []
    
    for day_col in weekdays.columns:
        for reg_col in regime_cols:
            interaction_name = f"{day_col}_x_{reg_col}"
            df[interaction_name] = df[day_col] * df[reg_col]
            
            # Minimum support check
            support = df[interaction_name].sum()
            if support < min_support:
                logger.debug(f"Dropping interaction {interaction_name} due to low support (N={support}).")
                df.drop(columns=[interaction_name], inplace=True)
            else:
                interaction_cols.append(interaction_name)
                
    logger.info(f"Generated {len(interaction_cols)} valid interaction terms with N >= {min_support}.")
    return df, interaction_cols
