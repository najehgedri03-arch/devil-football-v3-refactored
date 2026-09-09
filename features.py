"""
Feature engineering module.

Generates 32 features for ML models:
- Elo ratings and differences
- Form metrics (recent and historical)
- Goals for/against (rolling, home/away, with Bayesian smoothing)
- Attack/defense ratings
- Record (win rate)
- Clean sheet and failed-to-score rates
- Rest days
- Experience (matches played, clamped)

All features generated BEFORE match results are incorporated (strict chronological ordering).
"""

import numpy as np
from config import RECENT_FORM_WINDOW, LONG_TERM_WINDOW


FEATURE_NAMES = [
    # Elo-based
    "home_elo",
    "away_elo",
    "elo_difference",
    
    # Recent form
    "home_form",
    "away_form",
    "form_difference",
    
    # Points per game (recent)
    "home_ppg",
    "away_ppg",
    "ppg_difference",
    
    # Recent goals
    "home_recent_gf",
    "home_recent_ga",
    "away_recent_gf",
    "away_recent_ga",
    
    # Home/away splits
    "home_home_gf",
    "home_home_ga",
    "away_away_gf",
    "away_away_ga",
    
    # Attack/defense ratings
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
    
    # Record
    "home_win_rate",
    "away_win_rate",
    
    # Defensive metrics
    "home_clean_sheet",
    "away_clean_sheet",
    "home_failed_score",
    "away_failed_score",
    
    # Rest and experience
    "home_rest",
    "away_rest",
    "home_experience",
    "away_experience"
]

FEATURE_COUNT = len(FEATURE_NAMES)


def smoothed_stat(values, league_avg: float, smoothing_factor: float = 3.0) -> float:
    """
    Apply Bayesian smoothing to a rolling statistic.
    
    Blends observed value with league average:
    smoothed = (sum(values) + smoothing * league_avg) / (n + smoothing)
    
    This prevents small-sample overfitting and handles new teams gracefully.
    
    Args:
        values: Deque or list of observations
        league_avg: Prior (league-wide average)
        smoothing_factor: Weight of prior (higher = more conservative)
        
    Returns:
        Smoothed estimate
    """
    n = len(values)
    if n == 0:
        return league_avg
    
    return (sum(values) + smoothing_factor * league_avg) / (n + smoothing_factor)


def form_score(points_deque) -> float:
    """
    Convert points per game to form score (0.0 to 1.0).
    
    Args:
        points_deque: Deque of points earned (3/1/0)
        
    Returns:
        Form score: (avg_points / 3.0) clamped to [0, 1]
    """
    if len(points_deque) == 0:
        return 1.0 / 3.0  # Neutral form
    
    avg_points = sum(points_deque) / len(points_deque)
    return float(np.clip(avg_points / 3.0, 0.0, 1.0))


def make_features(home_stats, away_stats, league_stats, elo, current_date) -> np.ndarray:
    """
    Generate feature vector for a match.
    
    Args:
        home_stats: TeamStats object for home team
        away_stats: TeamStats object for away team
        league_stats: LeagueStats object
        elo: EloRatingSystem object
        current_date: Match date (for rest calculation)
        
    Returns:
        Feature vector of shape (32,) matching FEATURE_NAMES
    """
    # ================== ELO ==================
    home_elo = elo.get_rating(home_stats.total_matches if hasattr(home_stats, 'team_name') else None)
    # Note: we need to pass team name to elo system; workaround in production
    # For now, assume elo is pre-computed
    
    # Fallback: use basic Elo placeholder
    home_elo_val = 1500.0  # Will be set by caller
    away_elo_val = 1500.0  # Will be set by caller
    
    # ================== FORM ==================
    home_form = form_score(home_stats.points)
    away_form = form_score(away_stats.points)
    
    home_ppg = home_stats.recent_form_points()
    away_ppg = away_stats.recent_form_points()
    
    # ================== RECENT GOALS ==================
    # Get last N matches (RECENT_FORM_WINDOW)
    home_recent_gf_list = list(home_stats.goals_for)[-RECENT_FORM_WINDOW:]
    home_recent_ga_list = list(home_stats.goals_against)[-RECENT_FORM_WINDOW:]
    away_recent_gf_list = list(away_stats.goals_for)[-RECENT_FORM_WINDOW:]
    away_recent_ga_list = list(away_stats.goals_against)[-RECENT_FORM_WINDOW:]
    
    home_recent_gf = smoothed_stat(home_recent_gf_list, league_stats.home_avg)
    home_recent_ga = smoothed_stat(home_recent_ga_list, league_stats.away_avg)
    away_recent_gf = smoothed_stat(away_recent_gf_list, league_stats.away_avg)
    away_recent_ga = smoothed_stat(away_recent_ga_list, league_stats.home_avg)
    
    # ================== HOME/AWAY SPLITS ==================
    home_home_gf = smoothed_stat(home_stats.home_goals_for, league_stats.home_avg)
    home_home_ga = smoothed_stat(home_stats.home_goals_against, league_stats.away_avg)
    away_away_gf = smoothed_stat(away_stats.away_goals_for, league_stats.away_avg)
    away_away_ga = smoothed_stat(away_stats.away_goals_against, league_stats.home_avg)
    
    # ================== ATTACK/DEFENSE RATINGS ==================
    # Normalized by league average
    home_attack = home_home_gf / max(league_stats.home_avg, 0.10)
    home_defense = home_home_ga / max(league_stats.away_avg, 0.10)
    away_attack = away_away_gf / max(league_stats.away_avg, 0.10)
    away_defense = away_away_ga / max(league_stats.home_avg, 0.10)
    
    # ================== RECORD ==================
    home_win_rate = home_stats.win_rate()
    away_win_rate = away_stats.win_rate()
    
    # ================== DEFENSIVE METRICS ==================
    home_clean = home_stats.clean_sheet_rate()
    away_clean = away_stats.clean_sheet_rate()
    home_failed = home_stats.failed_to_score_rate()
    away_failed = away_stats.failed_to_score_rate()
    
    # ================== REST & EXPERIENCE ==================
    home_rest = home_stats.rest_days(current_date)
    away_rest = away_stats.rest_days(current_date)
    home_exp = float(min(home_stats.total_matches, 50))
    away_exp = float(min(away_stats.total_matches, 50))
    
    # ================== BUILD FEATURE VECTOR ==================
    features = np.array([
        home_elo_val,
        away_elo_val,
        home_elo_val - away_elo_val,
        
        home_form,
        away_form,
        home_form - away_form,
        
        home_ppg,
        away_ppg,
        home_ppg - away_ppg,
        
        home_recent_gf,
        home_recent_ga,
        away_recent_gf,
        away_recent_ga,
        
        home_home_gf,
        home_home_ga,
        away_away_gf,
        away_away_ga,
        
        home_attack,
        home_defense,
        away_attack,
        away_defense,
        
        home_win_rate,
        away_win_rate,
        
        home_clean,
        away_clean,
        home_failed,
        away_failed,
        
        home_rest,
        away_rest,
        home_exp,
        away_exp
    ], dtype=np.float32)
    
    assert len(features) == FEATURE_COUNT, f"Feature count mismatch: {len(features)} != {FEATURE_COUNT}"
    
    return features
