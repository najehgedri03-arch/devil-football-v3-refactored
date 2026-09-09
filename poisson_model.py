"""
Poisson goal modeling with Dixon-Coles low-score correction.

Generates probability distributions for match scorelines using:
1. Poisson distribution for goal counts
2. Dixon-Coles rho adjustment for low scores (0-0, 1-0, 0-1, 1-1)

Ref: Dixon & Coles (1997) "Modelling association football scores and inefficiencies in betting"
"""

import numpy as np
from math import exp, factorial
from config import MAX_GOALS_FOR_MATRIX, DIXON_COLES_RHO, DIXON_COLES_ADJUSTMENT


def poisson_probability(lam: float, k: int) -> float:
    """
    Poisson probability mass function.
    
    P(X=k) = (e^-λ * λ^k) / k!
    
    Args:
        lam: Rate parameter (expected value, e.g., expected goals)
        k: Number of goals
        
    Returns:
        Probability of exactly k goals
    """
    if k < 0 or lam < 0:
        return 0.0
    
    try:
        # Use log-space to prevent overflow
        log_prob = -lam + k * np.log(lam) - np.log(factorial(k))
        return float(np.exp(log_prob))
    except (OverflowError, ValueError):
        return 0.0


def dixon_coles_adjustment(home_goals: int, away_goals: int, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring matches.
    
    Adjusts probabilities of 0-0, 1-0, 0-1, 1-1 to match real-world data.
    
    Args:
        home_goals: Home team goals
        away_goals: Away team goals
        rho: Correlation parameter (typically -0.05 to -0.10)
        
    Returns:
        Multiplicative correction factor (typically 0.8 to 1.0)
    """
    if not DIXON_COLES_ADJUSTMENT:
        return 1.0
    
    # Only apply to low-scoring matches
    if home_goals <= 1 and away_goals <= 1:
        # tau adjustment
        tau = 1 + rho * (home_goals == 1) * (away_goals == 1) * 1.0
        return (1 - rho) + rho * tau
    
    return 1.0


def score_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """
    Generate probability matrix for all possible scorelines.
    
    Args:
        home_xg: Home team expected goals (from model)
        away_xg: Away team expected goals (from model)
        
    Returns:
        Matrix of shape (MAX_GOALS+1, MAX_GOALS+1) with probabilities
    """
    matrix = np.zeros((MAX_GOALS_FOR_MATRIX + 1, MAX_GOALS_FOR_MATRIX + 1))
    
    for h in range(MAX_GOALS_FOR_MATRIX + 1):
        for a in range(MAX_GOALS_FOR_MATRIX + 1):
            # Base Poisson probability
            p_h = poisson_probability(home_xg, h)
            p_a = poisson_probability(away_xg, a)
            p = p_h * p_a
            
            # Apply Dixon-Coles adjustment
            p *= dixon_coles_adjustment(h, a, DIXON_COLES_RHO)
            
            matrix[h, a] = p
    
    # Normalize to ensure probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total
    
    return matrix


def poisson_markets(home_xg: float, away_xg: float) -> dict:
    """
    Extract 1X2 and BTTS probabilities from Poisson matrix.
    
    Args:
        home_xg: Home expected goals
        away_xg: Away expected goals
        
    Returns:
        Dictionary with market probabilities:
        - '1': Home win
        - 'X': Draw
        - '2': Away win
        - 'GG': Both teams to score
        - 'NG': No both teams to score
    """
    matrix = score_matrix(home_xg, away_xg)
    
    p1 = 0.0   # Home wins (h > a)
    px = 0.0   # Draws (h == a)
    p2 = 0.0   # Away wins (a > h)
    pgg = 0.0  # Both scored (h > 0 and a > 0)
    
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            p = matrix[h, a]
            
            if h > a:
                p1 += p
            elif h == a:
                px += p
            else:
                p2 += p
            
            if h > 0 and a > 0:
                pgg += p
    
    return {
        "1": float(p1),
        "X": float(px),
        "2": float(p2),
        "GG": float(pgg),
        "NG": float(1.0 - pgg)
    }


def best_scorelines(home_xg: float, away_xg: float, n: int = 5) -> list:
    """
    Get the N most likely exact scorelines.
    
    Args:
        home_xg: Home expected goals
        away_xg: Away expected goals
        n: Number of top scorelines to return
        
    Returns:
        List of tuples: [(probability, "h-a"), ...]
    """
    matrix = score_matrix(home_xg, away_xg)
    
    scores = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((
                matrix[h, a],
                f"{h}-{a}"
            ))
    
    # Sort by probability, descending
    scores.sort(reverse=True, key=lambda x: x[0])
    
    return scores[:n]
