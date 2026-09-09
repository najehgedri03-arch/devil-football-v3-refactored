"""
Elo rating system with margin-of-victory adjustment.

Implements a football-specific Elo system that accounts for:
- Goal margin (larger wins = more rating change)
- Draw handling
- Home advantage

Ref: FiveThirtyEight NFL Elo, adapted for football.
"""

import numpy as np
from config import (
    INITIAL_ELO,
    ELO_K_FACTOR,
    HOME_ADVANTAGE_ELO,
    ELO_MARGIN_FACTORS
)


class EloRatingSystem:
    """
    Elo rating system for football teams.
    """
    
    def __init__(self):
        self.ratings = {}
    
    def get_rating(self, team: str) -> float:
        """
        Get current Elo rating for a team.
        
        Args:
            team: Team name
            
        Returns:
            Elo rating (default: INITIAL_ELO if unseen)
        """
        if team not in self.ratings:
            self.ratings[team] = INITIAL_ELO
        return self.ratings[team]
    
    def expected_win_probability(self, home_elo: float, away_elo: float) -> float:
        """
        Calculate expected win probability for home team.
        
        Formula: P = 1 / (1 + 10^((away_elo - home_elo_adj) / 400))
        where home_elo_adj = home_elo + HOME_ADVANTAGE_ELO
        
        Args:
            home_elo: Home team rating
            away_elo: Away team rating
            
        Returns:
            Probability home team wins (0.0 to 1.0)
        """
        home_elo_adj = home_elo + HOME_ADVANTAGE_ELO
        return 1.0 / (1.0 + 10 ** ((away_elo - home_elo_adj) / 400.0))
    
    def margin_factor(self, goal_margin: int) -> float:
        """
        Get margin-of-victory adjustment factor.
        
        Args:
            goal_margin: Absolute difference in goals
            
        Returns:
            Multiplicative factor (1.0 to 1.20)
        """
        if goal_margin in ELO_MARGIN_FACTORS:
            return ELO_MARGIN_FACTORS[goal_margin]
        else:
            return 1.20  # 4+ goal margin
    
    def update(self, home_team: str, away_team: str,
               home_goals: int, away_goals: int) -> dict:
        """
        Update Elo ratings after a match.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            
        Returns:
            Dictionary with update details
        """
        home_old = self.get_rating(home_team)
        away_old = self.get_rating(away_team)
        
        # Calculate expected probability
        expected_prob = self.expected_win_probability(home_old, away_old)
        
        # Determine actual result
        if home_goals > away_goals:
            actual = 1.0  # Home win
        elif home_goals < away_goals:
            actual = 0.0  # Away win
        else:
            actual = 0.5  # Draw
        
        # Margin of victory adjustment
        margin = abs(home_goals - away_goals)
        margin_mult = self.margin_factor(margin)
        
        # Rating change
        delta = ELO_K_FACTOR * margin_mult * (actual - expected_prob)
        
        # Update ratings
        self.ratings[home_team] = home_old + delta
        self.ratings[away_team] = away_old - delta
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_old_elo": home_old,
            "away_old_elo": away_old,
            "home_new_elo": self.ratings[home_team],
            "away_new_elo": self.ratings[away_team],
            "home_delta": delta,
            "expected_prob": expected_prob,
            "actual": actual
        }
