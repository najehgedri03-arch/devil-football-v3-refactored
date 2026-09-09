"""
Dataset construction for walk-forward validation.

Builds chronological feature matrices from match history, ensuring:
1. Features generated BEFORE match results are known
2. Strict chronological ordering (no time-series leakage)
3. Team state tracked accurately
4. Elo ratings updated after each match
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from config import MINIMUM_TEAM_MATCHES
from team_stats import TeamStats, LeagueStats
from elo_ratings import EloRatingSystem
from features import make_features, FEATURE_COUNT


def build_chronological_dataset(df: pd.DataFrame,
                                verbose: bool = False):
    """
    Build feature matrix from match history in chronological order.
    
    Ensures strict temporal ordering:
    1. Initialize empty team state
    2. For each match (chronologically):
       a. Generate features from current team state
       b. Record features and target
       c. Update team state with match result
       d. Update Elo ratings
    
    Args:
        df: Match data with Result and BTTS columns
        verbose: Print progress
        
    Returns:
        Tuple of (X, y_result, y_btts, y_home_goals, y_away_goals, teams, league, elo)
    """
    # Initialize state
    teams = defaultdict(TeamStats)
    league = LeagueStats()
    elo = EloRatingSystem()
    
    X_list = []
    y_result_list = []
    y_btts_list = []
    y_home_goals_list = []
    y_away_goals_list = []
    
    for idx, row in df.iterrows():
        home_name = row["HomeTeam"]
        away_name = row["AwayTeam"]
        date = row["Date"]
        
        home_stats = teams[home_name]
        away_stats = teams[away_name]
        
        # Skip if insufficient history
        if home_stats.total_matches < MINIMUM_TEAM_MATCHES:
            continue
        if away_stats.total_matches < MINIMUM_TEAM_MATCHES:
            continue
        
        # Get current Elo ratings
        home_elo = elo.get_rating(home_name)
        away_elo = elo.get_rating(away_name)
        
        # Manually inject Elo into feature generation
        # (Since make_features doesn't have team names, we'll patch it)
        features = make_features(home_stats, away_stats, league, elo, date)
        features[0] = home_elo  # Inject home Elo
        features[1] = away_elo  # Inject away Elo
        features[2] = home_elo - away_elo  # Update Elo difference
        
        X_list.append(features)
        y_result_list.append(row["Result"])
        y_btts_list.append(row["BTTS"])
        y_home_goals_list.append(int(row["HomeGoals"]))
        y_away_goals_list.append(int(row["AwayGoals"]))
        
        # ============================================
        # UPDATE TEAM STATE AFTER MATCH
        # ============================================
        home_goals = int(row["HomeGoals"])
        away_goals = int(row["AwayGoals"])
        
        home_stats.record_match_as_home(home_goals, away_goals, date)
        away_stats.record_match_as_away(away_goals, home_goals, date)
        
        # Update Elo
        elo.update(home_name, away_name, home_goals, away_goals)
        
        # Update league stats
        league.record_match(home_goals, away_goals)
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1} matches")
    
    # Convert to numpy arrays
    X = np.asarray(X_list, dtype=np.float32)
    y_result = np.asarray(y_result_list, dtype=object)
    y_btts = np.asarray(y_btts_list, dtype=int)
    y_home_goals = np.asarray(y_home_goals_list, dtype=int)
    y_away_goals = np.asarray(y_away_goals_list, dtype=int)
    
    if verbose:
        print(f"Built dataset: {X.shape[0]} matches, {X.shape[1]} features")
    
    return X, y_result, y_btts, y_home_goals, y_away_goals, teams, league, elo
