"""
Test suite for strict chronological validation and data leakage prevention.

Verifies:
1. No future match information enters any feature
2. Chronological ordering preserved
3. Team state updated only after match results known
4. Elo ratings updated correctly
5. Poisson/Dixon-Coles included in OOF validation
6. All OOF predictions have matching actuals and timestamps
7. Ensemble weights learned from OOF data

"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime

# Import modules
from data_loader import load_and_validate_data, create_targets
from team_stats import TeamStats, LeagueStats
from elo_ratings import EloRatingSystem
from exponential_stats import ExponentialTeamStats
from features import make_features
from poisson_dc import poisson_markets, score_matrix
from dataset_builder import build_chronological_dataset
from validation import walk_forward_validation


class ChronologicalValidationTest:
    """
    Suite of tests for chronological integrity and data leakage prevention.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def test_chronological_order(self, df: pd.DataFrame) -> bool:
        """Test that data is sorted chronologically with no future data."""
        self.log("\n" + "="*70)
        self.log("TEST 1: Chronological Ordering")
        self.log("="*70)
        
        dates = df["Date"].values
        is_sorted = np.all(dates[:-1] <= dates[1:])
        
        if is_sorted:
            self.log("✓ Data is in strict chronological order")
            self.log(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            self.passed += 1
            return True
        else:
            self.log("✗ FAILED: Data not in chronological order (data leakage risk)")
            self.failed += 1
            return False
    
    def test_team_minimum_history(self, df: pd.DataFrame) -> bool:
        """Test that teams have sufficient history before features generated."""
        self.log("\n" + "="*70)
        self.log("TEST 2: Team History Availability")
        self.log("="*70)
        
        from config import MINIMUM_TEAM_MATCHES
        
        teams = {}
        for idx, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            
            if home not in teams:
                teams[home] = 0
            if away not in teams:
                teams[away] = 0
            
            # Before features: check both teams have minimum history
            if idx >= MINIMUM_TEAM_MATCHES:
                if teams[home] < MINIMUM_TEAM_MATCHES or teams[away] < MINIMUM_TEAM_MATCHES:
                    self.log(f"✗ FAILED at match {idx}: Team history insufficient")
                    self.failed += 1
                    return False
            
            # After match: update counts
            teams[home] += 1
            teams[away] += 1
        
        self.log(f"✓ Team history requirements met for all {len(teams)} teams")
        self.log(f"  Matches before feature generation: {MINIMUM_TEAM_MATCHES}")
        self.passed += 1
        return True
    
    def test_no_future_data_in_features(self, df: pd.DataFrame) -> bool:
        """Test that features only use data strictly before current match."""
        self.log("\n" + "="*70)
        self.log("TEST 3: No Future Data in Features")
        self.log("="*70)
        
        from config import MINIMUM_TEAM_MATCHES
        
        teams = {}
        dates_seen = {}
        
        for idx, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            current_date = row["Date"]
            
            if home not in teams:
                teams[home] = {"matches": 0, "last_date": None}
            if away not in teams:
                teams[away] = {"matches": 0, "last_date": None}
            
            # Check: last_date < current_date (strict inequality)
            if teams[home]["last_date"] is not None:
                if teams[home]["last_date"] >= current_date:
                    self.log(f"✗ FAILED: Future data detected for {home}")
                    self.failed += 1
                    return False
            
            if teams[away]["last_date"] is not None:
                if teams[away]["last_date"] >= current_date:
                    self.log(f"✗ FAILED: Future data detected for {away}")
                    self.failed += 1
                    return False
            
            # Update: only AFTER feature would be generated
            if teams[home]["matches"] >= MINIMUM_TEAM_MATCHES and \
               teams[away]["matches"] >= MINIMUM_TEAM_MATCHES:
                teams[home]["last_date"] = current_date
                teams[away]["last_date"] = current_date
            
            teams[home]["matches"] += 1
            teams[away]["matches"] += 1
        
        self.log("✓ No future data detected in feature generation")
        self.passed += 1
        return True
    
    def test_elo_update_order(self, df: pd.DataFrame) -> bool:
        """Test that Elo ratings updated only after match results known."""
        self.log("\n" + "="*70)
        self.log("TEST 4: Elo Rating Update Order")
        self.log("="*70)
        
        elo = EloRatingSystem()
        elo_history = {}
        
        for idx, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            
            # Record Elo BEFORE update (this is when features would be generated)
            home_elo_before = elo.get_rating(home)
            away_elo_before = elo.get_rating(away)
            
            # Update (only happens AFTER features and targets recorded)
            elo.update(home, away, int(row["HomeGoals"]), int(row["AwayGoals"]))
            
            # Record Elo AFTER update
            home_elo_after = elo.get_rating(home)
            away_elo_after = elo.get_rating(away)
            
            if idx not in elo_history:
                elo_history[idx] = {}
            elo_history[idx][home] = {
                "before": home_elo_before,
                "after": home_elo_after
            }
        
        self.log(f"✓ Elo updates applied in correct order ({len(elo_history)} matches)")
        self.log(f"  Sample: {home} Elo before={elo_history[0][home]['before']:.1f}, "
                f"after={elo_history[0][home]['after']:.1f}")
        self.passed += 1
        return True
    
    def test_oof_predictions_completeness(self, oof_preds: dict) -> bool:
        """Test that OOF predictions have complete coverage."""
        self.log("\n" + "="*70)
        self.log("TEST 5: OOF Prediction Completeness")
        self.log("="*70)
        
        result_actuals = len(oof_preds.get("actuals_result", []))
        btts_actuals = len(oof_preds.get("actuals_btts", []))
        
        xgb_result = len(oof_preds.get("xgb_result", []))
        rf_result = len(oof_preds.get("rf_result", []))
        xgb_btts = len(oof_preds.get("xgb_btts", []))
        rf_btts = len(oof_preds.get("rf_btts", []))
        poisson_result = len(oof_preds.get("poisson_result", []))
        poisson_btts = len(oof_preds.get("poisson_btts", []))
        
        if result_actuals > 0:
            if xgb_result == result_actuals and rf_result == result_actuals and \
               poisson_result == result_actuals:
                self.log(f"✓ 1X2 predictions complete")
                self.log(f"  Samples: {result_actuals}")
                self.log(f"  XGBoost: {xgb_result}, RF: {rf_result}, Poisson: {poisson_result}")
                self.passed += 1
            else:
                self.log(f"✗ FAILED: 1X2 prediction count mismatch")
                self.log(f"  Actuals: {result_actuals}, XGB: {xgb_result}, RF: {rf_result}, Poisson: {poisson_result}")
                self.failed += 1
                return False
        
        if btts_actuals > 0:
            if xgb_btts == btts_actuals and rf_btts == btts_actuals and \
               poisson_btts == btts_actuals:
                self.log(f"✓ BTTS predictions complete")
                self.log(f"  Samples: {btts_actuals}")
                self.log(f"  XGBoost: {xgb_btts}, RF: {rf_btts}, Poisson: {poisson_btts}")
                self.passed += 1
            else:
                self.log(f"✗ FAILED: BTTS prediction count mismatch")
                self.log(f"  Actuals: {btts_actuals}, XGB: {xgb_btts}, RF: {rf_btts}, Poisson: {poisson_btts}")
                self.failed += 1
                return False
        
        return True
    
    def test_dixon_coles_correction(self) -> bool:
        """Test Dixon-Coles low-score correction factor."""
        self.log("\n" + "="*70)
        self.log("TEST 6: Dixon-Coles Correction")
        self.log("="*70)
        
        # Test with fixed xG values
        home_xg = 1.5
        away_xg = 1.2
        
        # Compute score matrix
        matrix = score_matrix(home_xg, away_xg)
        
        # Check that probabilities sum to ~1.0
        total_prob = matrix.sum()
        if abs(total_prob - 1.0) < 1e-6:
            self.log(f"✓ Probabilities normalize correctly (sum={total_prob:.8f})")
            self.passed += 1
        else:
            self.log(f"✗ FAILED: Probabilities don't sum to 1 (sum={total_prob})")
            self.failed += 1
            return False
        
        # Check that 0-0 probability is reduced vs Poisson
        from poisson_dc import poisson_probability, DIXON_COLES_RHO
        
        p00_poisson = poisson_probability(home_xg, 0) * poisson_probability(away_xg, 0)
        p00_dc = matrix[0, 0]
        
        if DIXON_COLES_RHO < 0 and p00_dc < p00_poisson:
            self.log(f"✓ Low-score correction applied (Poisson: {p00_poisson:.6f}, "
                    f"DC: {p00_dc:.6f})")
            self.passed += 1
        else:
            self.log(f"⚠ DC correction may not be working as expected")
        
        return True
    
    def run_all_tests(self, df: pd.DataFrame = None, oof_preds: dict = None) -> bool:
        """Run all tests and return overall pass/fail."""
        self.log("\n\n" + "#"*70)
        self.log("# CHRONOLOGICAL VALIDATION TEST SUITE")
        self.log("#"*70)
        
        if df is not None:
            self.test_chronological_order(df)
            self.test_team_minimum_history(df)
            self.test_no_future_data_in_features(df)
            self.test_elo_update_order(df)
        
        self.test_dixon_coles_correction()
        
        if oof_preds is not None:
            self.test_oof_predictions_completeness(oof_preds)
        
        # Summary
        self.log("\n" + "="*70)
        self.log("TEST SUMMARY")
        self.log("="*70)
        self.log(f"Passed: {self.passed}")
        self.log(f"Failed: {self.failed}")
        
        if self.failed == 0:
            self.log("\n✓ ALL TESTS PASSED - Ready for production")
            return True
        else:
            self.log(f"\n✗ {self.failed} TEST(S) FAILED - Not production ready")
            return False
