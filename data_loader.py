"""
Data loading and validation module.

Handles CSV reading, type conversion, sorting, and basic validation.
Ensures strict chronological ordering without data leakage.
"""

import warnings
import pandas as pd
import numpy as np
from config import REQUIRED_COLUMNS, CSV_FILE


def load_and_validate_data(csv_path: str = CSV_FILE) -> pd.DataFrame:
    """
    Load match data from CSV and validate structure.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with validated columns and chronological ordering
        
    Raises:
        ValueError: If required columns missing or data malformed
    """
    warnings.filterwarnings("ignore")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Failed to parse Date column. Check date format.")
    
    # Convert goals
    df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
    df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")
    
    # Remove rows with invalid goals
    before_len = len(df)
    df = df.dropna(subset=["HomeGoals", "AwayGoals"])
    if len(df) < before_len:
        print(f"Warning: Dropped {before_len - len(df)} rows with invalid goals.")
    
    # Clip negative goals to 0 and convert to int
    df["HomeGoals"] = df["HomeGoals"].clip(lower=0).astype(int)
    df["AwayGoals"] = df["AwayGoals"].clip(lower=0).astype(int)
    
    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(
        subset=["HomeTeam", "AwayTeam", "Date"],
        keep="first"
    ).reset_index(drop=True)
    
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variables from match results.
    
    Args:
        df: Match data
        
    Returns:
        DataFrame with added target columns:
        - Result: '1' (home win), 'X' (draw), '2' (away win)
        - BTTS: 1 if both teams scored, 0 otherwise
    """
    df = df.copy()
    
    # 1X2 result
    def get_result(home_goals, away_goals):
        if home_goals > away_goals:
            return "1"
        elif home_goals < away_goals:
            return "2"
        else:
            return "X"
    
    df["Result"] = df.apply(
        lambda row: get_result(row["HomeGoals"], row["AwayGoals"]),
        axis=1
    )
    
    # Both teams to score
    df["BTTS"] = ((df["HomeGoals"] > 0) & (df["AwayGoals"] > 0)).astype(int)
    
    return df


def validate_chronological_order(df: pd.DataFrame) -> bool:
    """
    Verify that data is in strict chronological order (no future data).
    
    Args:
        df: Match data
        
    Returns:
        True if dates are non-decreasing
    """
    dates = df["Date"].values
    is_sorted = np.all(dates[:-1] <= dates[1:])
    if not is_sorted:
        raise ValueError("Data is not in chronological order. Data leakage risk.")
    return True
