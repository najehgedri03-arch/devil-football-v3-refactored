"""
Comprehensive validation and comparison testing.

Generates detailed reports for:
1. Chronological integrity (no data leakage)
2. Dixon-Coles correction validation
3. Out-of-fold prediction completeness
4. 1X2 metrics: Accuracy, Log Loss, Brier Score, Calibration Error
5. BTTS metrics: Log Loss, Brier Score, AUC, Calibration Error
6. Final test set evaluation

Comparison:
- Old model (if available) vs New model (refactored)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from scipy.stats import entropy

from config import MINIMUM_TEAM_MATCHES, DIXON_COLES_RHO
from poisson_dc import poisson_probability, dixon_coles_phi
from calibration import calibration_error


class ComprehensiveValidationReport:
    """
    Generates complete validation report for the prediction engine.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.results = {}
    
    def log(self, msg: str = ""):
        if self.verbose:
            print(msg)
    
    def section(self, title: str):
        """Print section header."""
        self.log("\n" + "="*80)
        self.log(f" {title}")
        self.log("="*80)
    
    def test_dixon_coles_four_cells(self) -> bool:
        """
        Validate Dixon-Coles correction on four low-score cells numerically.
        
        Formula check:
        - (0,0): φ = 1 - ρ·λ_h·λ_a
        - (1,0): φ = 1 - ρ·λ_h·λ_a
        - (0,1): φ = 1 - ρ·λ_h·λ_a
        - (1,1): φ = 1 + ρ·λ_h·λ_a
        """
        self.section("TEST: Dixon-Coles Four-Cell Formula")
        
        home_xg = 1.5
        away_xg = 1.2
        rho = DIXON_COLES_RHO  # Should be negative
        
        self.log(f"Testing with home_xg={home_xg}, away_xg={away_xg}, rho={rho}")
        
        # Expected values
        phi_00_expected = 1.0 - rho * home_xg * away_xg
        phi_10_expected = 1.0 - rho * home_xg * away_xg
        phi_01_expected = 1.0 - rho * home_xg * away_xg
        phi_11_expected = 1.0 + rho * home_xg * away_xg
        
        # Computed values
        phi_00 = dixon_coles_phi(0, 0, home_xg, away_xg, rho)
        phi_10 = dixon_coles_phi(1, 0, home_xg, away_xg, rho)
        phi_01 = dixon_coles_phi(0, 1, home_xg, away_xg, rho)
        phi_11 = dixon_coles_phi(1, 1, home_xg, away_xg, rho)
        
        all_correct = True
        tol = 1e-6
        
        checks = [
            ("0-0", phi_00, phi_00_expected),
            ("1-0", phi_10, phi_10_expected),
            ("0-1", phi_01, phi_01_expected),
            ("1-1", phi_11, phi_11_expected)
        ]
        
        for scoreline, computed, expected in checks:
            match = abs(computed - expected) < tol
            status = "✓" if match else "✗"
            self.log(f"  {status} {scoreline}: computed={computed:.6f}, expected={expected:.6f}")
            if not match:
                all_correct = False
                self.failed += 1
        
        if rho < 0:
            # Check that low scores are reduced vs pure Poisson
            p00_poisson = poisson_probability(home_xg, 0) * poisson_probability(away_xg, 0)
            p00_dc = p00_poisson * phi_00
            
            if p00_dc < p00_poisson:
                self.log(f"  ✓ Low-score reduction: Poisson={p00_poisson:.6f}, DC={p00_dc:.6f}")
                self.passed += 1
            else:
                self.log(f"  ✗ Low-score reduction FAILED")
                self.failed += 1
                all_correct = False
        
        if all_correct:
            self.log("\n✓ Dixon-Coles formula PASSED")
            self.passed += 1
        else:
            self.log("\n✗ Dixon-Coles formula FAILED")
        
        return all_correct
    
    def test_chronological_integrity(self, df: pd.DataFrame) -> bool:
        """
        Verify strict chronological ordering.
        """
        self.section("TEST: Chronological Integrity")
        
        dates = df["Date"].values
        is_sorted = np.all(dates[:-1] <= dates[1:])
        
        if is_sorted:
            self.log(f"✓ Data sorted chronologically")
            self.log(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            self.log(f"  Total matches: {len(df)}")
            self.passed += 1
            return True
        else:
            self.log("✗ FAILED: Data not chronologically sorted")
            self.failed += 1
            return False
    
    def test_oof_completeness(self, oof_preds: dict) -> bool:
        """
        Verify OOF predictions have matching actuals.
        """
        self.section("TEST: OOF Prediction Completeness")
        
        n_result = len(oof_preds.get("actuals_result", []))
        n_btts = len(oof_preds.get("actuals_btts", []))
        
        all_correct = True
        
        if n_result > 0:
            n_xgb_result = len(oof_preds.get("xgb_result", []))
            n_rf_result = len(oof_preds.get("rf_result", []))
            n_poisson_result = len(oof_preds.get("poisson_result", []))
            
            if n_xgb_result == n_result and n_rf_result == n_result and n_poisson_result == n_result:
                self.log(f"✓ 1X2 predictions complete: {n_result} samples")
                self.log(f"  XGBoost: {n_xgb_result}, RF: {n_rf_result}, Poisson: {n_poisson_result}")
                self.passed += 1
            else:
                self.log(f"✗ 1X2 prediction count mismatch")
                self.log(f"  Actuals: {n_result}, XGB: {n_xgb_result}, RF: {n_rf_result}, Poisson: {n_poisson_result}")
                self.failed += 1
                all_correct = False
        
        if n_btts > 0:
            n_xgb_btts = len(oof_preds.get("xgb_btts", []))
            n_rf_btts = len(oof_preds.get("rf_btts", []))
            n_poisson_btts = len(oof_preds.get("poisson_btts", []))
            
            if n_xgb_btts == n_btts and n_rf_btts == n_btts and n_poisson_btts == n_btts:
                self.log(f"✓ BTTS predictions complete: {n_btts} samples")
                self.log(f"  XGBoost: {n_xgb_btts}, RF: {n_rf_btts}, Poisson: {n_poisson_btts}")
                self.passed += 1
            else:
                self.log(f"✗ BTTS prediction count mismatch")
                self.log(f"  Actuals: {n_btts}, XGB: {n_xgb_btts}, RF: {n_rf_btts}, Poisson: {n_poisson_btts}")
                self.failed += 1
                all_correct = False
        
        return all_correct
    
    def evaluate_1x2(self, predictions: dict, actuals: np.ndarray, 
                     model_name: str = "Ensemble") -> dict:
        """
        Evaluate 1X2 predictions.
        
        Args:
            predictions: Probability array, shape (N, 3)
            actuals: Class labels, shape (N,)
            model_name: Name for reporting
            
        Returns:
            Dictionary with metrics
        """
        # Convert labels to indices
        label_map = {"1": 0, "X": 1, "2": 2}
        y_true = np.array([label_map[label] for label in actuals])
        
        # Clip for stability
        preds = np.clip(predictions, 1e-8, 1 - 1e-8)
        preds = preds / preds.sum(axis=1, keepdims=True)
        
        # Get predicted class
        y_pred = np.argmax(preds, axis=1)
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        loss = log_loss(y_true, preds)
        brier = brier_score_loss(y_true, preds.max(axis=1))
        ece, _ = calibration_error(preds, y_true)
        
        metrics = {
            "Model": model_name,
            "Samples": len(y_true),
            "Accuracy": acc,
            "Log Loss": loss,
            "Brier Score": brier,
            "ECE": ece
        }
        
        return metrics
    
    def evaluate_btts(self, predictions: dict, actuals: np.ndarray,
                      model_name: str = "Ensemble") -> dict:
        """
        Evaluate BTTS (GG/NG) predictions.
        
        Args:
            predictions: Probability array, shape (N, 2) for [NG, GG]
            actuals: Binary labels (0=NG, 1=GG), shape (N,)
            model_name: Name for reporting
            
        Returns:
            Dictionary with metrics
        """
        y_true = np.asarray(actuals, dtype=int)
        
        # Clip and normalize
        preds = np.clip(predictions, 1e-8, 1 - 1e-8)
        if preds.ndim == 2 and preds.shape[1] == 2:
            preds = preds / preds.sum(axis=1, keepdims=True)
            preds_gg = preds[:, 1]  # Probability of GG
        else:
            preds_gg = preds
        
        # Metrics
        loss = log_loss(y_true, np.column_stack([1 - preds_gg, preds_gg]))
        brier = brier_score_loss(y_true, preds_gg)
        auc = roc_auc_score(y_true, preds_gg)
        
        # For ECE, create one-hot
        y_true_onehot = np.column_stack([1 - y_true, y_true])
        preds_matrix = np.column_stack([1 - preds_gg, preds_gg])
        ece, _ = calibration_error(preds_matrix, y_true_onehot)
        
        metrics = {
            "Model": model_name,
            "Samples": len(y_true),
            "Log Loss": loss,
            "Brier Score": brier,
            "AUC": auc,
            "ECE": ece
        }
        
        return metrics
    
    def report_summary(self, results_dict: dict):
        """Print final summary."""
        self.section("SUMMARY")
        
        self.log(f"Passed: {self.passed}")
        self.log(f"Failed: {self.failed}")
        
        if self.failed == 0:
            self.log("\n" + "#"*80)
            self.log("# ✓ ALL TESTS PASSED - SYSTEM IS PRODUCTION-READY")
            self.log("#"*80)
            return True
        else:
            self.log("\n" + "#"*80)
            self.log(f"# ✗ {self.failed} TEST(S) FAILED - NOT YET PRODUCTION-READY")
            self.log("#"*80)
            return False
