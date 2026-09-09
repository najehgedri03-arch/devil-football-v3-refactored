"""
CORRECTED Dixon-Coles low-score correction.

STANDARD FORMULA (Dixon & Coles 1997):

tau(0,0) = 1 - lambda_home * lambda_away * rho
tau(1,0) = 1 + lambda_away * rho
tau(0,1) = 1 + lambda_home * rho
tau(1,1) = 1 - rho
tau(i,j) = 1 for all other (i,j)

Where rho is NEGATIVE (e.g., -0.075), representing correlation in low scores.

The correction is applied as:
P(i,j) = Poisson(i; lambda_h) * Poisson(j; lambda_a) * tau(i,j,lambda_h,lambda_a,rho)
"""

import numpy as np
from math import factorial
from config import DIXON_COLES_RHO, DIXON_COLES_ADJUSTMENT, MAX_GOALS_FOR_MATRIX


def poisson_probability(lam: float, k: int) -> float:
    """
    Poisson PMF: P(X=k) = (e^-lambda * lambda^k) / k!
    
    Uses log-space to prevent numerical issues.
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
    STANDARD Dixon-Coles tau correction factor.
    
    Exact formulas:
    - tau(0,0) = 1 - lambda_home * lambda_away * rho
    - tau(1,0) = 1 + lambda_away * rho
    - tau(0,1) = 1 + lambda_home * rho
    - tau(1,1) = 1 - rho
    - tau(i,j) = 1 otherwise
    
    Args:
        home_goals: Home team goals
        away_goals: Away team goals
        home_xg: Home expected goals (lambda_home in formula)
        away_xg: Away expected goals (lambda_away in formula)
        rho: Correlation parameter (NEGATIVE, typically -0.05 to -0.10)
        
    Returns:
        Correction factor tau
    """
    if not DIXON_COLES_ADJUSTMENT or rho >= 0:
        return 1.0
    
    # Apply standard four-cell formula
    if home_goals == 0 and away_goals == 0:
        # tau(0,0) = 1 - lambda_home * lambda_away * rho
        tau = 1.0 - home_xg * away_xg * rho
    elif home_goals == 1 and away_goals == 0:
        # tau(1,0) = 1 + lambda_away * rho
        tau = 1.0 + away_xg * rho
    elif home_goals == 0 and away_goals == 1:
        # tau(0,1) = 1 + lambda_home * rho
        tau = 1.0 + home_xg * rho
    elif home_goals == 1 and away_goals == 1:
        # tau(1,1) = 1 - rho
        tau = 1.0 - rho
    else:
        # All other scorelines
        tau = 1.0
    
    return float(tau)


def score_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """
    Generate probability matrix for all scorelines with Dixon-Coles.
    
    P(i,j) = Poisson(i; home_xg) * Poisson(j; away_xg) * tau(i,j,...)
    
    Args:
        home_xg: Home expected goals
        away_xg: Away expected goals
        
    Returns:
        Probability matrix, shape (MAX_GOALS+1, MAX_GOALS+1)
    """
    matrix = np.zeros((MAX_GOALS_FOR_MATRIX + 1, MAX_GOALS_FOR_MATRIX + 1))
    
    for h in range(MAX_GOALS_FOR_MATRIX + 1):
        for a in range(MAX_GOALS_FOR_MATRIX + 1):
            # Poisson probabilities
            p_h = poisson_probability(home_xg, h)
            p_a = poisson_probability(away_xg, a)
            p = p_h * p_a
            
            # Dixon-Coles correction
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
    
    p1 = 0.0   # Home win
    px = 0.0   # Draw
    p2 = 0.0   # Away win
    pgg = 0.0  # Both scored
    
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
    
    pgg = float(np.clip(pgg, 0.0, 1.0))
    png = float(np.clip(1.0 - pgg, 0.0, 1.0))
    
    return {
        "1": float(p1),
        "X": float(px),
        "2": float(p2),
        "GG": pgg,
        "NG": png
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
