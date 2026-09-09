"""
ML models for football prediction.

Provides:
- XGBoost classifier (1X2 and BTTS)
- Random Forest classifier with scaling
- Poisson goal count regressors

All models use sklearn Pipeline for consistent preprocessing.
"""

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import (
    XGB_RESULT_PARAMS,
    XGB_BTTS_PARAMS,
    RF_RESULT_PARAMS,
    RF_BTTS_PARAMS,
    POISSON_GOAL_PARAMS
)


def create_xgb_result_model():
    """
    Create XGBoost model for 1X2 prediction.
    
    Returns:
        XGBClassifier or None if XGBoost not available
    """
    if not XGB_AVAILABLE:
        return None
    
    return XGBClassifier(**XGB_RESULT_PARAMS)


def create_xgb_btts_model():
    """
    Create XGBoost model for BTTS prediction.
    
    Returns:
        XGBClassifier or None if XGBoost not available
    """
    if not XGB_AVAILABLE:
        return None
    
    return XGBClassifier(**XGB_BTTS_PARAMS)


def create_rf_result_model():
    """
    Create Random Forest with scaling for 1X2 prediction.
    
    Returns:
        Pipeline with StandardScaler and RandomForestClassifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**RF_RESULT_PARAMS))
    ])


def create_rf_btts_model():
    """
    Create Random Forest with scaling for BTTS prediction.
    
    Returns:
        Pipeline with StandardScaler and RandomForestClassifier
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(**RF_BTTS_PARAMS))
    ])


def create_poisson_goal_model():
    """
    Create Poisson regressor for goal count prediction.
    
    Returns:
        Pipeline with StandardScaler and PoissonRegressor
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(**POISSON_GOAL_PARAMS))
    ])
