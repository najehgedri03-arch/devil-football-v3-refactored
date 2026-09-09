"""
Exponentially weighted team statistics.

Replaces simple rolling averages with exponential weighting,
giving more importance to recent matches.

For each observation x_i, weight w_i = decay^(n-i) where n is total count.
Recent matches have higher weight.
"""

import numpy as np
from collections import deque
from config import EXPONENTIAL_DECAY, LONG_TERM_WINDOW


class ExponentiallyWeightedStats:
    """
    Tracks statistics with exponential weighting toward recent events.
    """
    
    def __init__(self, decay: float = EXPONENTIAL_DECAY, maxlen: int = LONG_TERM_WINDOW):
        """
        Args:
            decay: Decay factor per match (0 < decay < 1)
                   Higher decay = more weight on recent matches
                   decay=0.5 means each match back has 50% the weight
            maxlen: Maximum history to keep
        """
        if not (0 < decay < 1):
            raise ValueError(f"Decay must be in (0, 1), got {decay}")
        
        self.decay = decay
        self.values = deque(maxlen=maxlen)
    
    def add(self, value: float):
        """Add a new observation."""
        self.values.append(float(value))
    
    def weighted_mean(self, default: float = 0.0) -> float:
        """
        Compute exponentially weighted mean.
        
        Most recent observation has weight 1.0.
        Previous observation has weight decay.
        Earlier observations have progressively lower weights.
        
        Returns:
            Weighted mean
        """
        if len(self.values) == 0:
            return default
        
        values_list = list(self.values)
        n = len(values_list)
        
        # Weights: most recent = 1.0, previous = decay, etc.
        # weights[i] = decay^(n-1-i) for oldest to newest
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()  # Normalize
        
        return float(np.dot(weights, values_list))
    
    def __len__(self) -> int:
        return len(self.values)
    
    def __repr__(self) -> str:
        return f"ExponentialWeightedStats(decay={self.decay}, values={list(self.values)})"


class ExponentialTeamStats:
    """
    Team statistics using exponential weighting throughout.
    """
    
    def __init__(self, decay: float = EXPONENTIAL_DECAY):
        self.decay = decay
        
        # Goals
        self.goals_for = ExponentiallyWeightedStats(decay)
        self.goals_against = ExponentiallyWeightedStats(decay)
        self.points = ExponentiallyWeightedStats(decay)
        
        # Home/away splits
        self.home_goals_for = ExponentiallyWeightedStats(decay)
        self.home_goals_against = ExponentiallyWeightedStats(decay)
        self.away_goals_for = ExponentiallyWeightedStats(decay)
        self.away_goals_against = ExponentiallyWeightedStats(decay)
        
        # Defensive
        self.clean_sheets = ExponentiallyWeightedStats(decay)
        self.failed_to_score = ExponentiallyWeightedStats(decay)
        
        # All-time record
        self.wins = 0
        self.draws = 0
        self.losses = 0
        
        # Tracking
        self.total_matches = 0
        self.last_match_date = None
    
    def record_match_as_home(self, goals_for: int, goals_against: int, date):
        """Record home match."""
        self.goals_for.add(goals_for)
        self.goals_against.add(goals_against)
        self.home_goals_for.add(goals_for)
        self.home_goals_against.add(goals_against)
        
        if goals_for > goals_against:
            points = 3
            self.wins += 1
        elif goals_for == goals_against:
            points = 1
            self.draws += 1
        else:
            points = 0
            self.losses += 1
        
        self.points.add(points)
        self.clean_sheets.add(1 if goals_against == 0 else 0)
        self.failed_to_score.add(1 if goals_for == 0 else 0)
        
        self.total_matches += 1
        self.last_match_date = date
    
    def record_match_as_away(self, goals_for: int, goals_against: int, date):
        """Record away match."""
        self.goals_for.add(goals_for)
        self.goals_against.add(goals_against)
        self.away_goals_for.add(goals_for)
        self.away_goals_against.add(goals_against)
        
        if goals_for > goals_against:
            points = 3
            self.wins += 1
        elif goals_for == goals_against:
            points = 1
            self.draws += 1
        else:
            points = 0
            self.losses += 1
        
        self.points.add(points)
        self.clean_sheets.add(1 if goals_against == 0 else 0)
        self.failed_to_score.add(1 if goals_for == 0 else 0)
        
        self.total_matches += 1
        self.last_match_date = date
    
    def recent_form_points(self) -> float:
        """Exponentially weighted average points."""
        return self.points.weighted_mean(1.0)
    
    def win_rate(self) -> float:
        """All-time win rate."""
        total = self.wins + self.draws + self.losses
        if total == 0:
            return 1.0 / 3.0
        return float(self.wins) / total
    
    def clean_sheet_rate(self) -> float:
        """Exponentially weighted clean sheet rate."""
        return self.clean_sheets.weighted_mean(0.33)
    
    def failed_to_score_rate(self) -> float:
        """Exponentially weighted failed-to-score rate."""
        return self.failed_to_score.weighted_mean(0.33)
    
    def rest_days(self, current_date) -> float:
        """Days since last match (0-30)."""
        if self.last_match_date is None:
            return 7.0
        days = (current_date - self.last_match_date).days
        return float(min(max(days, 0), 30))
