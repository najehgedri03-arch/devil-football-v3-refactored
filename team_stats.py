"""
Team statistics tracking module.

Maintains rolling windows of match statistics for each team:
- Goals for/against (rolling)
- Points earned (rolling)
- Home/away splits
- Clean sheets
- Failed-to-score matches
- Win/draw/loss counts
- Rest days between matches

All statistics use fixed-size deques to prevent data leakage.
"""

from collections import defaultdict, deque
from config import LONG_TERM_WINDOW, CLEAN_SHEET_WINDOW


class TeamStats:
    """
    Maintains rolling statistics for a single team.
    """
    
    def __init__(self):
        # Rolling windows
        self.goals_for = deque(maxlen=LONG_TERM_WINDOW)  # Goals scored
        self.goals_against = deque(maxlen=LONG_TERM_WINDOW)  # Goals conceded
        self.points = deque(maxlen=LONG_TERM_WINDOW)  # Points per match (3/1/0)
        
        # Home/away splits
        self.home_goals_for = deque(maxlen=LONG_TERM_WINDOW)
        self.home_goals_against = deque(maxlen=LONG_TERM_WINDOW)
        self.away_goals_for = deque(maxlen=LONG_TERM_WINDOW)
        self.away_goals_against = deque(maxlen=LONG_TERM_WINDOW)
        
        # Defensive metrics
        self.clean_sheets = deque(maxlen=CLEAN_SHEET_WINDOW)
        self.failed_to_score = deque(maxlen=CLEAN_SHEET_WINDOW)
        
        # Win/Draw/Loss record
        self.wins = 0
        self.draws = 0
        self.losses = 0
        
        # Match tracking
        self.total_matches = 0
        self.last_match_date = None
    
    def record_match_as_home(self, goals_for: int, goals_against: int, date):
        """
        Record a home match result.
        
        Args:
            goals_for: Goals scored
            goals_against: Goals conceded
            date: Match date
        """
        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.home_goals_for.append(goals_for)
        self.home_goals_against.append(goals_against)
        
        # Points: 3 for win, 1 for draw, 0 for loss
        if goals_for > goals_against:
            points = 3
            self.wins += 1
        elif goals_for == goals_against:
            points = 1
            self.draws += 1
        else:
            points = 0
            self.losses += 1
        
        self.points.append(points)
        self.clean_sheets.append(1 if goals_against == 0 else 0)
        self.failed_to_score.append(1 if goals_for == 0 else 0)
        
        self.total_matches += 1
        self.last_match_date = date
    
    def record_match_as_away(self, goals_for: int, goals_against: int, date):
        """
        Record an away match result.
        
        Args:
            goals_for: Goals scored
            goals_against: Goals conceded
            date: Match date
        """
        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.away_goals_for.append(goals_for)
        self.away_goals_against.append(goals_against)
        
        # Points
        if goals_for > goals_against:
            points = 3
            self.wins += 1
        elif goals_for == goals_against:
            points = 1
            self.draws += 1
        else:
            points = 0
            self.losses += 1
        
        self.points.append(points)
        self.clean_sheets.append(1 if goals_against == 0 else 0)
        self.failed_to_score.append(1 if goals_for == 0 else 0)
        
        self.total_matches += 1
        self.last_match_date = date
    
    def get_mean(self, values: deque, default: float = 0.0) -> float:
        """
        Calculate mean of rolling window, with default for empty.
        
        Args:
            values: Deque of values
            default: Default value if empty
            
        Returns:
            Mean value
        """
        if len(values) == 0:
            return float(default)
        return float(sum(values) / len(values))
    
    def recent_form_points(self) -> float:
        """
        Average points per match (recent form).
        Range: 0.0 (all losses) to 3.0 (all wins)
        """
        return self.get_mean(self.points, 1.0)
    
    def win_rate(self) -> float:
        """
        Proportion of matches won.
        Range: 0.0 to 1.0
        """
        total = self.wins + self.draws + self.losses
        if total == 0:
            return 1.0 / 3.0  # Uninformed prior
        return float(self.wins) / total
    
    def clean_sheet_rate(self) -> float:
        """
        Proportion of matches with clean sheet (no goals conceded).
        Range: 0.0 to 1.0
        """
        return self.get_mean(self.clean_sheets, 0.33)
    
    def failed_to_score_rate(self) -> float:
        """
        Proportion of matches where team failed to score.
        Range: 0.0 to 1.0
        """
        return self.get_mean(self.failed_to_score, 0.33)
    
    def rest_days(self, current_date) -> float:
        """
        Days since last match (clamped 0-30).
        
        Args:
            current_date: Current match date
            
        Returns:
            Days of rest (0-30)
        """
        if self.last_match_date is None:
            return 7.0  # Default for unseen team
        
        days = (current_date - self.last_match_date).days
        return float(min(max(days, 0), 30))


class LeagueStats:
    """
    Maintains league-wide aggregate statistics.
    
    Used for:
    - Smoothing team stats (Bayesian prior)
    - Normalizing attack/defense ratings
    """
    
    def __init__(self):
        self.home_goals = deque(maxlen=300)
        self.away_goals = deque(maxlen=300)
    
    def record_match(self, home_goals: int, away_goals: int):
        """
        Record match goals for league statistics.
        
        Args:
            home_goals: Home team goals
            away_goals: Away team goals
        """
        self.home_goals.append(home_goals)
        self.away_goals.append(away_goals)
    
    @property
    def home_avg(self) -> float:
        """Average goals scored by home team."""
        if len(self.home_goals) == 0:
            return 1.35
        return float(sum(self.home_goals) / len(self.home_goals))
    
    @property
    def away_avg(self) -> float:
        """Average goals scored by away team."""
        if len(self.away_goals) == 0:
            return 1.10
        return float(sum(self.away_goals) / len(self.away_goals))
    
    @property
    def total_avg(self) -> float:
        """Average total goals per match."""
        if len(self.home_goals) == 0:
            return 2.45
        all_goals = list(self.home_goals) + list(self.away_goals)
        return float(sum(all_goals) / len(all_goals))
