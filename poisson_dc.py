"""
Fixed Dixon-Coles low-score correction using correct four-cell formula.

Reference: Dixon & Coles (1997)
The correction factor for each scoreline (i,j) is:

  φ(i,j,λ_h,λ_a,ρ) = 1 - ρ·λ_h·λ_a   if (i,j) ∈ {(0,0), (1,0), (0,1), (1,1)}
  φ(i,j,λ_h,λ_a,ρ) = 1                  otherwise

Where ρ is typically -0.05 to -0.10 (negative, representing low-score deficit).
"""

import numpy as np
from math import factorial
from config import DIXON_COLES_RHO, DIXON_COLES_ADJUSTMENT, MAX_GOALS_FOR_MATRIX


def poisson_probability(lam: float, k: int) -> float:
    """
    Poisson PMF: P(X=k) = (e^-λ * λ^k) / k!
    
    Uses log-space to prevent overflow/underflow.
    
    Args:
        lam: Rate parameter (expected goals)
        k: Number of goals (non-negative integer)
        
    Returns:
        Probability mass at k
    """
    if k < 0 or lam < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    
    try:
        # Log-space computation for numerical stability
        log_prob = -lam + k * np.log(lam) - np.log(factorial(k))
        return float(np.exp(log_prob))
    except (OverflowError, ValueError):
        return 0.0


def dixon_coles_phi(home_goals: int, away_goals: int,
                    home_xg: float, away_xg: float,
                    rho: float) -> float:
    """
    Dixon-Coles correction factor φ(i,j,λ_h,λ_a,ρ).
    
    Standard four-cell formula:
    - (0,0): φ = 1 - ρ·λ_h·λ_a
    - (1,0): φ = 1 - ρ·λ_h·λ_a
    - (0,1): φ = 1 - ρ·λ_h·λ_a
    - (1,1): φ = 1 + ρ·λ_h·λ_a
    - All others: φ = 1
    
    Args:
        home_goals: Home team goals scored
        away_goals: Away team goals scored
        home_xg: Home team expected goals (from Poisson model)
        away_xg: Away team expected goals (from Poisson model)
        rho: Correlation parameter (typically -0.05 to -0.10)
        
    Returns:
        Multiplicative correction factor
    """
    if not DIXON_COLES_ADJUSTMENT or rho >= 0:
        return 1.0
    
    # Only apply to low-scoring matches
    if (home_goals, away_goals) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        if (home_goals, away_goals) == (1, 1):
            # 1-1: φ = 1 + ρ·λ_h·λ_a (note: + not -)
            phi = 1.0 + rho * home_xg * away_xg
        else:
            # 0-0, 1-0, 0-1: φ = 1 - ρ·λ_h·λ_a
            phi = 1.0 - rho * home_xg * away_xg
    else:
        phi = 1.0
    
    return float(phi)


def score_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    """
    Generate probability matrix for all scorelines with Dixon-Coles correction.
    
    P(i,j) = Poisson(i; λ_h) * Poisson(j; λ_a) * φ(i,j,λ_h,λ_a,ρ)
    
    Args:
        home_xg: Home team expected goals
        away_xg: Away team expected goals
        
    Returns:
        Probability matrix, shape (MAX_GOALS+1, MAX_GOALS+1)
    """
    matrix = np.zeros((MAX_GOALS_FOR_MATRIX + 1, MAX_GOALS_FOR_MATRIX + 1))
    
    for h in range(MAX_GOALS_FOR_MATRIX + 1):
        for a in range(MAX_GOALS_FOR_MATRIX + 1):
            # Base Poisson probability
            p_h = poisson_probability(home_xg, h)
            p_a = poisson_probability(away_xg, a)
            p = p_h * p_a
            
            # Apply Dixon-Coles correction
            phi = dixon_coles_phi(h, a, home_xg, away_xg, DIXON_COLES_RHO)
            p = p * phi
            
            matrix[h, a] = p
    
    # Normalize to ensure probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix /= total
    
    return matrix


def poisson_markets(home_xg: float, away_xg: float) -> dict:
    """
    Extract 1X2 and BTTS market probabilities from Poisson matrix.
    
    Args:
        home_xg: Home team expected goals
        away_xg: Away team expected goals
        
    Returns:
        Dictionary with keys '1', 'X', '2', 'GG', 'NG'
    """
    matrix = score_matrix(home_xg, away_xg)
    
    p1 = 0.0   # Home wins (h > a)
    px = 0.0   # Draws (h == a)
    p2 = 0.0   # Away wins (a > h)
    pgg = 0.0  # Both teams to score (h > 0 and a > 0)
    
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
    Return top N most likely exact scorelines.
    
    Args:
        home_xg: Home team expected goals
        away_xg: Away team expected goals
        n: Number of top scorelines to return
        
    Returns:
        List of (probability, "h-a") tuples, sorted descending
    """
    matrix = score_matrix(home_xg, away_xg)
    
    scores = []
    for h in range(matrix.shape[0]):
        for a in range(matrix.shape[1]):
            scores.append((matrix[h, a], f"{h}-{a}"))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:n]
