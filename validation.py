"""
Corrected walk-forward validation with strict leakage prevention.

Key fixes:
1. Features generated BEFORE any match result is known
2. Predictions made BEFORE updating team state
3. Team state/Elo updated ONLY AFTER prediction
4. Poisson/Dixon-Coles included in OOF predictions
5. Final untouched test set for validation
6. Temperature and ensemble weights learned from inner validation only

"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

from config import (
    MINIMUM_HISTORY_FOR_TRAINING,
    WALK_FORWARD_TEST_BLOCK_SIZE,
    MINIMUM_TEAM_MATCHES
)
from team_stats import TeamStats, LeagueStats
from elo_ratings import EloRatingSystem
from features import make_features
from poisson_dc import poisson_markets
from ml_models import (
    create_xgb_result_model,
    create_rf_result_model,
    create_xgb_btts_model,
    create_rf_btts_model,
    create_poisson_goal_model
)


def walk_forward_validation(df: pd.DataFrame,
                            min_train_size: int = MINIMUM_HISTORY_FOR_TRAINING,
                            test_block_size: int = WALK_FORWARD_TEST_BLOCK_SIZE,
                            final_test_ratio: float = 0.2,
                            verbose: bool = True) -> dict:
    """
    Strict walk-forward validation with separate final test set.
    
    Process:
    1. Reserve final_test_ratio% of matches as untouched final test set
    2. On remaining data, perform walk-forward validation:
       - Train on historical data only
       - Generate features (NO future/target info)
       - Make predictions
       - Update state and models ONLY after recording predictions
    3. Learn temperature and ensemble weights from inner validation
    4. Evaluate on final untouched test set
    
    Args:
        df: Match data, chronologically sorted
        min_train_size: Minimum training matches before making predictions
        test_block_size: Size of each validation fold
        final_test_ratio: Fraction of data to reserve for final test (0.1-0.3)
        verbose: Print progress
        
    Returns:
        Dictionary with OOF predictions and final test results
    """
    if len(df) <= min_train_size:
        raise ValueError(f"Insufficient data: {len(df)}, need {min_train_size}")
    
    # Split: inner validation and final test
    n_final_test = max(int(len(df) * final_test_ratio), test_block_size)
    split_idx = len(df) - n_final_test
    
    df_inner = df.iloc[:split_idx].copy()
    df_final_test = df.iloc[split_idx:].copy()
    
    if verbose:
        print("\n" + "="*80)
        print(" WALK-FORWARD VALIDATION WITH FINAL TEST SET")
        print("="*80)
        print(f"Inner validation: {len(df_inner)} matches")
        print(f"Final test set: {len(df_final_test)} matches (untouched)")
        print(f"Final test date range: {df_final_test['Date'].min().date()} to "
              f"{df_final_test['Date'].max().date()}")
    
    # ========================================================================
    # PHASE 1: INNER WALK-FORWARD VALIDATION (for learning weights/temp)
    # ========================================================================
    
    oof_predictions = {
        "xgb_result": [],
        "rf_result": [],
        "poisson_result": [],
        "xgb_btts": [],
        "rf_btts": [],
        "poisson_btts": [],
        "actuals_result": [],
        "actuals_btts": [],
        "dates": [],
        "home_teams": [],
        "away_teams": []
    }
    
    fold_num = 0
    for train_end_idx in range(min_train_size, len(df_inner), test_block_size):
        fold_num += 1
        
        test_start_idx = train_end_idx
        test_end_idx = min(train_end_idx + test_block_size, len(df_inner))
        
        train_data = df_inner.iloc[:train_end_idx].copy()
        test_data = df_inner.iloc[test_start_idx:test_end_idx].copy()
        
        if verbose:
            print(f"\nFold {fold_num}: Train [{0}:{train_end_idx}], "
                  f"Test [{test_start_idx}:{test_end_idx}]")
            print(f"  Train: {len(train_data)} matches (until {train_data['Date'].max().date()})")
            print(f"  Test:  {len(test_data)} matches")
        
        # Train models on training data
        from dataset_builder import build_chronological_dataset
        X_train, y_train_result, y_train_btts, y_train_hg, y_train_ag, _, _, _ = \
            build_chronological_dataset(train_data, verbose=False)
        
        if len(X_train) == 0:
            if verbose:
                print("  Skipping fold: insufficient training samples")
            continue
        
        # Fit models
        xgb_result = create_xgb_result_model()
        rf_result = create_rf_result_model()
        xgb_btts = create_xgb_btts_model()
        rf_btts = create_rf_btts_model()
        home_goal_model = create_poisson_goal_model()
        away_goal_model = create_poisson_goal_model()
        
        if xgb_result is not None:
            xgb_result.fit(X_train, y_train_result)
        rf_result.fit(X_train, y_train_result)
        
        if xgb_btts is not None:
            xgb_btts.fit(X_train, y_train_btts)
        rf_btts.fit(X_train, y_train_btts)
        
        home_goal_model.fit(X_train, y_train_hg)
        away_goal_model.fit(X_train, y_train_ag)
        
        # Rebuild state for test predictions
        teams = defaultdict(TeamStats)
        league = LeagueStats()
        elo = EloRatingSystem()
        
        for _, row in train_data.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]
            
            home_stats = teams[home]
            away_stats = teams[away]
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            # Update state (this is AFTER training, so no leakage)
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home, away, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
        
        # Generate test predictions: BEFORE updating with test match results
        for _, row in test_data.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]
            
            home_stats = teams[home]
            away_stats = teams[away]
            
            # Skip if insufficient history
            if home_stats.total_matches < MINIMUM_TEAM_MATCHES:
                continue
            if away_stats.total_matches < MINIMUM_TEAM_MATCHES:
                continue
            
            # ================================================================
            # GENERATE FEATURES (NO TARGET INFO YET)
            # ================================================================
            home_elo = elo.get_rating(home)
            away_elo = elo.get_rating(away)
            
            features = make_features(home_stats, away_stats, league, elo, date)
            features[0] = home_elo
            features[1] = away_elo
            features[2] = home_elo - away_elo
            
            X_test = np.atleast_2d(features)
            
            # ================================================================
            # MAKE PREDICTIONS (BEFORE SEEING RESULT)
            # ================================================================
            
            # 1X2 predictions
            if xgb_result is not None:
                xgb_pred = xgb_result.predict_proba(X_test)[0]
                oof_predictions["xgb_result"].append(xgb_pred)
            
            rf_pred = rf_result.predict_proba(X_test)[0]
            oof_predictions["rf_result"].append(rf_pred)
            
            # Poisson predictions
            home_xg = float(np.clip(home_goal_model.predict(X_test)[0], 0.1, 4.5))
            away_xg = float(np.clip(away_goal_model.predict(X_test)[0], 0.1, 4.5))
            poisson_1x2 = poisson_markets(home_xg, away_xg)
            poisson_result_vec = np.array([
                poisson_1x2["1"],
                poisson_1x2["X"],
                poisson_1x2["2"]
            ])
            oof_predictions["poisson_result"].append(poisson_result_vec)
            
            # BTTS predictions
            if xgb_btts is not None:
                xgb_btts_pred = xgb_btts.predict_proba(X_test)[0]
                oof_predictions["xgb_btts"].append(xgb_btts_pred)
            
            rf_btts_pred = rf_btts.predict_proba(X_test)[0]
            oof_predictions["rf_btts"].append(rf_btts_pred)
            
            # Poisson BTTS (convert to binary: NG, GG)
            poisson_btts_vec = np.array([
                poisson_1x2["NG"],
                poisson_1x2["GG"]
            ])
            oof_predictions["poisson_btts"].append(poisson_btts_vec)
            
            # Record actual result (no leakage: happens AFTER prediction)
            oof_predictions["actuals_result"].append(row["Result"])
            oof_predictions["actuals_btts"].append(row["BTTS"])
            oof_predictions["dates"].append(row["Date"])
            oof_predictions["home_teams"].append(home)
            oof_predictions["away_teams"].append(away)
            
            # ================================================================
            # UPDATE STATE (AFTER PREDICTION AND RECORDING)
            # ================================================================
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home, away, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
        
        if verbose:
            print(f"  Generated {len(oof_predictions['actuals_result'])} OOF predictions")
    
    # Convert to arrays
    for key in ["xgb_result", "rf_result", "poisson_result", 
                "xgb_btts", "rf_btts", "poisson_btts"]:
        if oof_predictions[key]:
            oof_predictions[key] = np.asarray(oof_predictions[key])
    
    oof_predictions["actuals_result"] = np.asarray(oof_predictions["actuals_result"])
    oof_predictions["actuals_btts"] = np.asarray(oof_predictions["actuals_btts"])
    
    if verbose:
        print(f"\nInner validation complete: {len(oof_predictions['actuals_result'])} OOF samples")
    
    # ========================================================================
    # PHASE 2: FINAL TEST SET (completely untouched)
    # ========================================================================
    
    final_test_preds = {
        "xgb_result": [],
        "rf_result": [],
        "poisson_result": [],
        "xgb_btts": [],
        "rf_btts": [],
        "poisson_btts": [],
        "actuals_result": [],
        "actuals_btts": [],
        "dates": [],
        "home_teams": [],
        "away_teams": []
    }
    
    # Train final models on ALL inner validation data
    from dataset_builder import build_chronological_dataset
    X_final_train, y_final_train_result, y_final_train_btts, \
        y_final_train_hg, y_final_train_ag, _, _, _ = \
        build_chronological_dataset(df_inner, verbose=False)
    
    if len(X_final_train) > 0:
        xgb_result_final = create_xgb_result_model()
        rf_result_final = create_rf_result_model()
        xgb_btts_final = create_xgb_btts_model()
        rf_btts_final = create_rf_btts_model()
        home_goal_model_final = create_poisson_goal_model()
        away_goal_model_final = create_poisson_goal_model()
        
        if xgb_result_final is not None:
            xgb_result_final.fit(X_final_train, y_final_train_result)
        rf_result_final.fit(X_final_train, y_final_train_result)
        
        if xgb_btts_final is not None:
            xgb_btts_final.fit(X_final_train, y_final_train_btts)
        rf_btts_final.fit(X_final_train, y_final_train_btts)
        
        home_goal_model_final.fit(X_final_train, y_final_train_hg)
        away_goal_model_final.fit(X_final_train, y_final_train_ag)
        
        # Rebuild state up to final test
        teams = defaultdict(TeamStats)
        league = LeagueStats()
        elo = EloRatingSystem()
        
        for _, row in df_inner.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]
            
            home_stats = teams[home]
            away_stats = teams[away]
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home, away, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
        
        # Test on final set
        for _, row in df_final_test.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            date = row["Date"]
            
            home_stats = teams[home]
            away_stats = teams[away]
            
            if home_stats.total_matches < MINIMUM_TEAM_MATCHES:
                continue
            if away_stats.total_matches < MINIMUM_TEAM_MATCHES:
                continue
            
            # Generate features (no target info)
            home_elo = elo.get_rating(home)
            away_elo = elo.get_rating(away)
            
            features = make_features(home_stats, away_stats, league, elo, date)
            features[0] = home_elo
            features[1] = away_elo
            features[2] = home_elo - away_elo
            
            X_test = np.atleast_2d(features)
            
            # Make predictions (BEFORE seeing result)
            if xgb_result_final is not None:
                xgb_pred = xgb_result_final.predict_proba(X_test)[0]
                final_test_preds["xgb_result"].append(xgb_pred)
            
            rf_pred = rf_result_final.predict_proba(X_test)[0]
            final_test_preds["rf_result"].append(rf_pred)
            
            home_xg = float(np.clip(home_goal_model_final.predict(X_test)[0], 0.1, 4.5))
            away_xg = float(np.clip(away_goal_model_final.predict(X_test)[0], 0.1, 4.5))
            poisson_1x2 = poisson_markets(home_xg, away_xg)
            poisson_result_vec = np.array([poisson_1x2["1"], poisson_1x2["X"], poisson_1x2["2"]])
            final_test_preds["poisson_result"].append(poisson_result_vec)
            
            if xgb_btts_final is not None:
                xgb_btts_pred = xgb_btts_final.predict_proba(X_test)[0]
                final_test_preds["xgb_btts"].append(xgb_btts_pred)
            
            rf_btts_pred = rf_btts_final.predict_proba(X_test)[0]
            final_test_preds["rf_btts"].append(rf_btts_pred)
            
            poisson_btts_vec = np.array([poisson_1x2["NG"], poisson_1x2["GG"]])
            final_test_preds["poisson_btts"].append(poisson_btts_vec)
            
            # Record actual
            final_test_preds["actuals_result"].append(row["Result"])
            final_test_preds["actuals_btts"].append(row["BTTS"])
            final_test_preds["dates"].append(row["Date"])
            final_test_preds["home_teams"].append(home)
            final_test_preds["away_teams"].append(away)
            
            # Update state (after prediction)
            home_goals = int(row["HomeGoals"])
            away_goals = int(row["AwayGoals"])
            
            home_stats.record_match_as_home(home_goals, away_goals, date)
            away_stats.record_match_as_away(away_goals, home_goals, date)
            elo.update(home, away, home_goals, away_goals)
            league.record_match(home_goals, away_goals)
    
    # Convert final test to arrays
    for key in ["xgb_result", "rf_result", "poisson_result", 
                "xgb_btts", "rf_btts", "poisson_btts"]:
        if final_test_preds[key]:
            final_test_preds[key] = np.asarray(final_test_preds[key])
    
    final_test_preds["actuals_result"] = np.asarray(final_test_preds["actuals_result"])
    final_test_preds["actuals_btts"] = np.asarray(final_test_preds["actuals_btts"])
    
    if verbose:
        print(f"\nFinal test complete: {len(final_test_preds['actuals_result'])} test samples")
    
    return {
        "oof": oof_predictions,
        "final_test": final_test_preds,
        "split_index": split_idx
    }
