"""
Walk-forward validation framework.

Implements strict time-series cross-validation:
1. Split data into train/test blocks chronologically
2. Train models only on historical data
3. Test on future matches
4. Accumulate out-of-fold predictions
5. Optimize ensemble weights from validation performance

This prevents all forms of data leakage:
- No future information used for training
- No random shuffling
- Test dates always after training dates
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss

from config import (
    MINIMUM_HISTORY_FOR_TRAINING,
    WALK_FORWARD_TEST_BLOCK_SIZE,
    MINIMUM_TEAM_MATCHES
)
from dataset_builder import build_chronological_dataset
from elo_ratings import EloRatingSystem
from team_stats import TeamStats, LeagueStats
from features import make_features
from ml_models import (
    create_xgb_result_model,
    create_rf_result_model,
    create_xgb_btts_model,
    create_rf_btts_model
)


def walk_forward_validation(df: pd.DataFrame,
                            min_train_size: int = MINIMUM_HISTORY_FOR_TRAINING,
                            test_block_size: int = WALK_FORWARD_TEST_BLOCK_SIZE,
                            verbose: bool = True) -> dict:
    """
    Perform walk-forward validation on match data.
    
    Splits data into sequential train/test blocks, trains models on train,
    evaluates on test, and accumulates out-of-fold predictions.
    
    Args:
        df: Match data with Date, Result, BTTS columns
        min_train_size: Minimum matches to train on initially
        test_block_size: Matches to test on per fold
        verbose: Print progress
        
    Returns:
        Dictionary with validation metrics and out-of-fold predictions
    """
    if len(df) <= min_train_size:
        raise ValueError(
            f"Insufficient data: {len(df)} matches, need at least {min_train_size}"
        )
    
    if verbose:
        print("\n" + "="*70)
        print(" WALK-FORWARD VALIDATION")
        print("="*70)
    
    # Track out-of-fold predictions
    oof_predictions = {
        "xgb_result": [],
        "rf_result": [],
        "xgb_btts": [],
        "rf_btts": [],
        "actuals_result": [],
        "actuals_btts": []
    }
    
    # Walk forward
    fold_num = 0
    for train_end_idx in range(min_train_size, len(df), test_block_size):
        fold_num += 1
        
        # Test set
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + test_block_size, len(df))
        
        train_data = df.iloc[:train_end_idx].copy()
        test_data = df.iloc[test_start_idx:test_end_idx].copy()
        
        if verbose:
            print(f"\nFold {fold_num}: Train [{0}:{train_end_idx}], Test [{test_start_idx}:{test_end_idx}]")
            print(f"  Train: {len(train_data)} matches (until {train_data['Date'].max().date()})")
            print(f"  Test:  {len(test_data)} matches (from {test_data['Date'].min().date()})")
        
        # Build training dataset
        X_train, y_train_result, y_train_btts, _, _, _, _, _ = \
            build_chronological_dataset(train_data, verbose=False)
        
        if len(X_train) == 0:
            if verbose:
                print(f"  Skipping fold: insufficient training samples")
            continue
        
        # Train models
        xgb_result = create_xgb_result_model()
        rf_result = create_rf_result_model()
        xgb_btts = create_xgb_btts_model()
        rf_btts = create_rf_btts_model()
        
        if xgb_result is not None:
            xgb_result.fit(X_train, y_train_result)
        rf_result.fit(X_train, y_train_result)
        
        if xgb_btts is not None:
            xgb_btts.fit(X_train, y_train_btts)
        rf_btts.fit(X_train, y_train_btts)
        
        # Generate features for test set
        teams = defaultdict(TeamStats)
        league = LeagueStats()
        elo = EloRatingSystem()
        
        # Rebuild training state
        for _, row in train_data.iterrows():
            home_name = row["HomeTeam"]
            away_name = row["AwayTeam"]
            date = row["Date"]
            
            home_stats = teams[home_name]
            away_stats = teams[away_name]
            
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home_name, away_name, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
        
        # Test predictions
        X_test_list = []
        
        for _, row in test_data.iterrows():
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
            
            home_elo = elo.get_rating(home_name)
            away_elo = elo.get_rating(away_name)
            
            features = make_features(home_stats, away_stats, league, elo, date)
            features[0] = home_elo
            features[1] = away_elo
            features[2] = home_elo - away_elo
            
            X_test_list.append(features)
            
            # Update state for next test match
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home_name, away_name, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
            
            # Collect predictions
            X_test_single = np.atleast_2d(features)
            
            if xgb_result is not None:
                xgb_result_pred = xgb_result.predict_proba(X_test_single)[0]
                oof_predictions["xgb_result"].append(xgb_result_pred)
            
            rf_result_pred = rf_result.predict_proba(X_test_single)[0]
            oof_predictions["rf_result"].append(rf_result_pred)
            
            if xgb_btts is not None:
                xgb_btts_pred = xgb_btts.predict_proba(X_test_single)[0]
                oof_predictions["xgb_btts"].append(xgb_btts_pred)
            
            rf_btts_pred = rf_btts.predict_proba(X_test_single)[0]
            oof_predictions["rf_btts"].append(rf_btts_pred)
            
            oof_predictions["actuals_result"].append(row["Result"])
            oof_predictions["actuals_btts"].append(row["BTTS"])
        
        if verbose and len(X_test_list) > 0:
            print(f"  Generated {len(X_test_list)} test samples")
    
    # Convert to arrays
    for key in list(oof_predictions.keys()):
        if oof_predictions[key]:
            if key.startswith("actuals"):
                oof_predictions[key] = np.asarray(oof_predictions[key])
            else:
                oof_predictions[key] = np.asarray(oof_predictions[key])
    
    if verbose:
        print(f"\nTotal out-of-fold samples: {len(oof_predictions['actuals_result'])}")
    
    return oof_predictions
