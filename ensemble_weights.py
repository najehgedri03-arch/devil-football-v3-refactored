"""
Ensemble weighting optimization.

Optimizes ensemble weights using out-of-fold predictions from walk-forward validation.

Instead of using fixed arbitrary weights, learns optimal weights that minimize
log loss and calibration error on validation data.
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from sklearn.metrics import log_loss


def optimize_ensemble_weights(predictions: dict,
                              actuals: np.ndarray,
                              market: str = "1x2",
                              verbose: bool = True) -> dict:
    """
    Optimize ensemble weights to minimize log loss.
    
    Constraints:
    - Weights sum to 1.0
    - Each weight in [0.0, 1.0]
    
    Args:
        predictions: Dict with keys ['xgb', 'rf', 'poisson']
                    Values are arrays of shape (N, 3) for 1x2 or (N,) for BTTS
        actuals: True labels, shape (N,) with class indices
        market: '1x2' or 'btts' (affects label encoding)
        verbose: Print optimization results
        
    Returns:
        Dict with optimal weights and final log loss
    """
    if market == "1x2":
        n_classes = 3
        label_map = {"1": 0, "X": 1, "2": 2}
    elif market == "btts":
        n_classes = 2
        label_map = {"NG": 0, "GG": 1}
    else:
        raise ValueError(f"Unknown market: {market}")
    
    # Convert actuals to indices if needed
    if isinstance(actuals[0], str):
        y_true = np.array([label_map[label] for label in actuals])
    else:
        y_true = np.asarray(actuals, dtype=int)
    
    # Prepare prediction arrays
    pred_arrays = {}
    for model_name, preds in predictions.items():
        if preds is None:
            continue
        preds = np.asarray(preds, dtype=np.float64)
        if preds.ndim == 1:
            # BTTS: convert to (N, 2) one-hot-like
            preds_2d = np.column_stack([1 - preds, preds])
            pred_arrays[model_name] = preds_2d
        else:
            pred_arrays[model_name] = preds
    
    available_models = list(pred_arrays.keys())
    n_models = len(available_models)
    
    if n_models == 0:
        raise ValueError("No predictions available")
    
    if n_models == 1:
        # Only one model: weight = 1.0
        return {
            "weights": {available_models[0]: 1.0},
            "log_loss": log_loss(y_true, pred_arrays[available_models[0]])
        }
    
    # Objective: minimize log loss
    def objective(w):
        # Compute weighted ensemble
        ensemble = np.zeros_like(pred_arrays[available_models[0]])
        for i, model_name in enumerate(available_models):
            ensemble += w[i] * pred_arrays[model_name]
        
        # Normalize
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
        
        # Clip for numerical stability
        ensemble = np.clip(ensemble, 1e-8, 1 - 1e-8)
        
        return log_loss(y_true, ensemble)
    
    # Constraint: weights sum to 1
    constraints = LinearConstraint(
        np.ones((1, n_models)),
        [1.0],
        [1.0]
    )
    
    # Bounds: each weight in [0, 1]
    bounds = Bounds(
        lb=np.zeros(n_models),
        ub=np.ones(n_models)
    )
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )
    
    if not result.success and verbose:
        print(f"Warning: Optimization did not fully converge: {result.message}")
    
    # Final weights
    final_weights = dict(zip(available_models, result.x))
    final_loss = objective(result.x)
    
    if verbose:
        print(f"\nOptimal {market} ensemble weights:")
        for model, w in final_weights.items():
            print(f"  {model}: {w:.4f}")
        print(f"Validation log loss: {final_loss:.6f}")
    
    return {
        "weights": final_weights,
        "log_loss": float(final_loss),
        "success": result.success
    }


def apply_ensemble_weights(predictions: dict,
                           weights: dict,
                           normalize: bool = True) -> np.ndarray:
    """
    Apply learned ensemble weights to predictions.
    
    Args:
        predictions: Dict with model predictions
        weights: Dict with model weights (should sum to ~1.0)
        normalize: Whether to normalize final ensemble
        
    Returns:
        Weighted ensemble predictions
    """
    ensemble = None
    total_weight = 0.0
    
    for model_name, weight in weights.items():
        if model_name not in predictions or predictions[model_name] is None:
            continue
        
        preds = np.asarray(predictions[model_name], dtype=np.float64)
        
        if ensemble is None:
            ensemble = weight * preds
        else:
            ensemble = ensemble + weight * preds
        
        total_weight += weight
    
    if ensemble is None:
        raise ValueError("No predictions to ensemble")
    
    if normalize and total_weight > 0:
        ensemble = ensemble / total_weight
    
    # Normalize if 2D (probability distributions)
    if ensemble.ndim == 2:
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
    
    return ensemble
