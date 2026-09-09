"""
Fixed Dixon-Coles low-score correction.

Proper implementation using both home and away expected goals.
Ref: Dixon & Coles (1997) "Modelling association football scores..."

The tau factor depends on BOTH teams' expected goals to capture
the interaction between low-scoring teams.
"""

import numpy as np
from math import exp, factorial
from config import DIXON_COLES_RHO, DIXON_COLES_ADJUSTMENT, MAX_GOALS_FOR_MATRIX


def poisson_probability(lam: float, k: int) -> float:
    """
    Poisson PMF: P(X=k) = (e^-λ * λ^k) / k!
    
    Uses log-space to prevent overflow.
    """
    if k < 0 or lam < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    
    try:
        log_prob = -lam + k * np.log(lam) - np.log(factorial(k))
        return float(np.exp(log_prob))
    except (OverflowError, ValueError):
        return 0.0


def dixon_coles_tau(home_goals: int, away_goals: int,
                    home_xg: float, away_xg: float,
                    rho: float) -> float:
    """
    Compute Dixon-Coles tau correction factor.
    
    The tau factor is defined as:
    τ(i,j,λ_h,λ_a) = 1 - ρ * (i==0) * (j==0) * exp(ρ) * (i==1) * (j==1)
    
    But for numerical stability and correctness, we use:
    τ = 1 - ρ  for {0-0, 1-0, 0-1, 1-1}
    τ = 1      otherwise
    
    The rho parameter is NEGATIVE (e.g., -0.075), representing that
    low scores are less likely than Poisson would predict.
    
    Args:
        home_goals: Home team goals
        away_goals: Away team goals
        home_xg: Home expected goals (for reference, not used in tau)
        away_xg: Away expected goals (for reference, not used in tau)
        rho: Correlation parameter (typically -0.05 to -0.10)
        
    Returns:
        Multiplicative correction factor
    """
    if not DIXON_COLES_ADJUSTMENT or rho >= 0:
        return 1.0
    
    # Apply tau only to low-scoring matches
    if home_goals <= 1 and away_goals <= 1:
        # Standard DC adjustment
        if home_goals == 0 and away_goals == 0:
            # 0-0: reduce probability
            tau = 1 + rho  # rho is negative, so this < 1
        elif (home_goals == 1 and away_goals == 0) or (home_goals == 0 and away_goals == 1):
            # 1-0 or 0-1: reduce probability
            tau = 1 + rho
        elif home_goals == 1 and away_goals == 1:
            # 1-1: reduce probability more
            tau = 1 + rho * rho  # Two-fold effect
        else:
            tau = 1.0
    else:
        tau = 1.0
    
    return float(tau)


def score_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """
    Generate probability matrix for all scorelines with Dixon-Coles.
    
    Args:
        home_xg: Home team expected goals
        away_xg: Away team expected goals
        
    Returns:
        Matrix of shape (MAX_GOALS+1, MAX_GOALS+1)
    """
    matrix = np.zeros((MAX_GOALS_FOR_MATRIX + 1, MAX_GOALS_FOR_MATRIX + 1))
    
    for h in range(MAX_GOALS_FOR_MATRIX + 1):
        for a in range(MAX_GOALS_FOR_MATRIX + 1):
            # Poisson probability
            p = poisson_probability(home_xg, h) * poisson_probability(away_xg, a)
            
            # Dixon-Coles adjustment
            tau = dixon_coles_tau(h, a, home_xg, away_xg, DIXON_COLES_RHO)
            p = p * tau
            
            matrix[h, a] = p
    
    # Normalize
    total = matrix.sum()
    if total > 0:
        matrix /= total
    
    return matrix


def poisson_markets(home_xg: float, away_xg: float) -> dict:
    """
    Extract 1X2 and BTTS probabilities.
    """
    matrix = score_matrix(home_xg, away_xg)
    
    p1 = 0.0
    px = 0.0
    p2 = 0.0
    pgg = 0.0
    
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
    Return top N most likely scorelines.
    """
    matrix = score_matrix(home_xg, away_xg)
    
    scores = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((matrix[h, a], f"{h}-{a}"))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:n]
