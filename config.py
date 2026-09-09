"""
Configuration module for DEVIL Football Prediction Engine v3.0 Refactored.

All constants centralized here for easy tuning and reproducibility.
"""

# ============================================================
# DATA CONFIGURATION
# ============================================================

CSV_FILE = "matches.csv"

# Required columns in CSV
REQUIRED_COLUMNS = [
    "HomeTeam",
    "AwayTeam",
    "HomeGoals",
    "AwayGoals",
    "Date"
]

# Optional odds columns
OPTIONAL_ODDS_COLUMNS = [
    "HomeOdds",
    "DrawOdds",
    "AwayOdds",
    "GGOdds",
    "NGOdds"
]

# ============================================================
# CHRONOLOGICAL VALIDATION
# ============================================================

MINIMUM_HISTORY_FOR_TRAINING = 250  # Must have 250+ matches before predictions
WALK_FORWARD_TEST_BLOCK_SIZE = 25   # Test on 25 matches at a time
MINIMUM_TEAM_MATCHES = 5            # Team must have 5+ matches before feature generation

# ============================================================
# ELO CONFIGURATION
# ============================================================

INITIAL_ELO = 1500.0
ELO_K_FACTOR = 20.0                 # Base K-factor for rating adjustment
HOME_ADVANTAGE_ELO = 55.0           # Home advantage in Elo ratings

# Margin-of-victory adjustment
ELO_MARGIN_FACTORS = {
    0: 1.0,    # Draw
    1: 1.0,    # 1-goal margin
    2: 1.10,   # 2-goal margin
    3: 1.15,   # 3-goal margin
}  # Default: 1.20 for 4+ goal margins

# ============================================================
# FEATURE WINDOWS
# ============================================================

RECENT_FORM_WINDOW = 5              # Recent form: last 5 matches
LONG_TERM_WINDOW = 10               # Long-term stats: last 10 matches
CLEAN_SHEET_WINDOW = 10             # Clean sheets: last 10 matches

# ============================================================
# EXPONENTIAL WEIGHTING
# ============================================================

# For exponentially weighted form (more recent = higher weight)
EXPONENTIAL_DECAY = 0.5             # Decay factor per match (older matches count less)

# ============================================================
# GOAL MODELS (POISSON + DIXON-COLES)
# ============================================================

MAX_GOALS_FOR_MATRIX = 10           # Maximum goals for Poisson matrix (0-10)
DIXON_COLES_ADJUSTMENT = True       # Apply Dixon-Coles low-score correction

# Dixon-Coles rho parameter (low-score correlation)
# Typical: -0.05 to -0.10 (negative correlation for 0-0, 1-0, 0-1)
DIXON_COLES_RHO = -0.075

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

# XGBoost (1X2 result prediction)
XGB_RESULT_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "gamma": 0.05,
    "reg_alpha": 0.10,
    "reg_lambda": 1.50,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1
}

# XGBoost (BTTS prediction)
XGB_BTTS_PARAMS = {
    "n_estimators": 600,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "gamma": 0.05,
    "reg_alpha": 0.10,
    "reg_lambda": 1.50,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1
}

# Random Forest (1X2 result prediction)
RF_RESULT_PARAMS = {
    "n_estimators": 800,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1
}

# Random Forest (BTTS prediction)
RF_BTTS_PARAMS = {
    "n_estimators": 800,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1
}

# Poisson goal models
POISSON_GOAL_PARAMS = {
    "alpha": 0.20,
    "max_iter": 1500
}

# ============================================================
# CALIBRATION
# ============================================================

# Temperature scaling for probability calibration
CALIBRATION_TEMPERATURE = 1.08
CALIBRATION_MIN_PROB = 1e-8
CALIBRATION_MAX_PROB = 1.0 - 1e-8

# ============================================================
# ENSEMBLE WEIGHTING
# ============================================================

# Initial weights (will be optimized via walk-forward validation)
# These sum to 1.0 per market
INITIAL_1X2_WEIGHTS = {
    "xgb": 0.40,
    "rf": 0.20,
    "poisson": 0.40
}

INITIAL_BTTS_WEIGHTS = {
    "xgb": 0.30,
    "rf": 0.25,
    "poisson": 0.45
}

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

# Confidence classification based on prediction probability and margin
CONFIDENCE_THRESHOLDS = {
    "VERY_HIGH": {"min_prob": 0.65, "min_margin": 0.20},
    "HIGH": {"min_prob": 0.55, "min_margin": 0.12},
    "MEDIUM": {"min_prob": 0.45, "min_margin": 0.07},
    "LOW": {"min_prob": 0.0, "min_margin": 0.0}
}

# ============================================================
# PREDICTION CLIPPING
# ============================================================

xG_MIN = 0.10
xG_MAX = 4.50

# ============================================================
# LOGGING
# ============================================================

VERBOSE = True  # Set to True for detailed logging
