"""
Temperature scaling optimization from out-of-fold validation data.

Learns the optimal temperature parameter by minimizing log loss
on validation predictions, rather than using a hardcoded value.

"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss


def optimize_temperature(logits: np.ndarray,
                         actuals: np.ndarray,
                         class_indices: dict = None,
                         verbose: bool = True) -> dict:
    """
    Optimize temperature scaling on validation data.
    
    Finds the temperature T that minimizes log loss:
    L(T) = -sum(y_i * log(softmax(logits_i / T)))
    
    Args:
        logits: Raw model scores, shape (N, C)
        actuals: True labels, either indices (N,) or one-hot (N, C)
        class_indices: Dict mapping class names to indices (if actuals are strings)
        verbose: Print results
        
    Returns:
        Dict with optimal temperature and validation log loss
    """
    logits = np.asarray(logits, dtype=np.float64)
    
    # Convert actuals to indices if needed
    if isinstance(actuals[0], str):
        if class_indices is None:
            raise ValueError("class_indices required for string labels")
        y_true = np.array([class_indices[label] for label in actuals])
    else:
        y_true = np.asarray(actuals, dtype=int)
    
    def log_loss_with_temp(T: float) -> float:
        """Compute log loss for given temperature."""
        if T <= 0:
            return 1e10
        
        # Temperature scale logits
        scaled_logits = logits / T
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        
        # Softmax
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Clip for stability
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        
        # Compute log loss (averaged over samples)
        return log_loss(y_true, probs)
    
    # Search for optimal temperature
    result = minimize_scalar(
        log_loss_with_temp,
        bounds=(0.5, 2.0),
        method="bounded",
        options={"xatol": 1e-6}
    )
    
    optimal_temp = float(result.x)
    optimal_loss = float(result.fun)
    
    if verbose:
        print(f"\nTemperature Scaling Optimization")
        print(f"  Optimal temperature: {optimal_temp:.6f}")
        print(f"  Validation log loss: {optimal_loss:.6f}")
    
    return {
        "temperature": optimal_temp,
        "log_loss": optimal_loss
    }


def apply_temperature_scaling(predictions: np.ndarray,
                              temperature: float = 1.0) -> np.ndarray:
    """
    Apply learned temperature scaling to predictions.
    
    Args:
        predictions: Probabilities, shape (N, C)
        temperature: Optimal temperature from validation
        
    Returns:
        Scaled probabilities
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    
    # Convert to log space
    log_probs = np.log(np.clip(predictions, 1e-8, 1))
    
    # Scale
    log_probs = log_probs / temperature
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
    
    # Convert back
    scaled = np.exp(log_probs)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    
    return scaled
