"""
OOF-based temperature and ensemble weight optimization.

Key principle: Learn ONLY from inner OOF validation data.
Never touch or fit on final test set.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import log_loss


def optimize_temperature_1x2(oof_predictions: dict,
                              verbose: bool = True) -> dict:
    """
    Optimize temperature scaling for 1X2 predictions using OOF data.
    
    Learns T by minimizing log loss:
    L(T) = -(1/N) * sum_i sum_j y_ij * log(softmax(logits_i / T)_j)
    
    Args:
        oof_predictions: Dict with 'xgb_result', 'rf_result', 'poisson_result', 'actuals_result'
        verbose: Print results
        
    Returns:
        Dict with optimal temperature and validation log loss
    """
    xgb_probs = oof_predictions["xgb_result"]
    rf_probs = oof_predictions["rf_result"]
    poisson_probs = oof_predictions["poisson_result"]
    actuals = oof_predictions["actuals_result"]
    
    # Convert actuals to one-hot
    label_map = {"1": 0, "X": 1, "2": 2}
    y_true = np.array([label_map[label] for label in actuals])
    y_onehot = np.eye(3)[y_true]
    
    # Average ensemble (equal weights for now)
    n_models = sum([len(xgb_probs) > 0, len(rf_probs) > 0, len(poisson_probs) > 0])
    ensemble = np.zeros_like(y_onehot, dtype=np.float64)
    
    if len(xgb_probs) > 0:
        ensemble += xgb_probs
    if len(rf_probs) > 0:
        ensemble += rf_probs
    if len(poisson_probs) > 0:
        ensemble += poisson_probs
    
    ensemble /= n_models
    
    # Clip for stability
    ensemble = np.clip(ensemble, 1e-8, 1 - 1e-8)
    ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
    
    def log_loss_with_temp(T: float) -> float:
        if T <= 0:
            return 1e10
        
        # Temperature scale
        logits = np.log(ensemble)
        scaled_logits = logits / T
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        
        # Softmax
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        
        # Log loss
        return log_loss(y_true, probs)
    
    # Optimize
    result = minimize_scalar(
        log_loss_with_temp,
        bounds=(0.5, 2.0),
        method="bounded",
        options={"xatol": 1e-6}
    )
    
    optimal_temp = float(result.x)
    optimal_loss = float(result.fun)
    
    if verbose:
        print(f"\n1X2 Temperature Optimization")
        print(f"  Optimal temperature: {optimal_temp:.6f}")
        print(f"  OOF log loss: {optimal_loss:.6f}")
    
    return {
        "temperature": optimal_temp,
        "log_loss": optimal_loss
    }


def optimize_temperature_btts(oof_predictions: dict,
                              verbose: bool = True) -> dict:
    """
    Optimize temperature for BTTS predictions using OOF data.
    """
    xgb_probs = oof_predictions["xgb_btts"]
    rf_probs = oof_predictions["rf_btts"]
    poisson_probs = oof_predictions["poisson_btts"]
    actuals = oof_predictions["actuals_btts"]
    
    y_true = np.asarray(actuals, dtype=int)
    
    # Average ensemble
    n_models = sum([len(xgb_probs) > 0, len(rf_probs) > 0, len(poisson_probs) > 0])
    ensemble = np.zeros_like(xgb_probs, dtype=np.float64)
    
    if len(xgb_probs) > 0:
        ensemble += xgb_probs
    if len(rf_probs) > 0:
        ensemble += rf_probs
    if len(poisson_probs) > 0:
        ensemble += poisson_probs
    
    ensemble /= n_models
    ensemble = np.clip(ensemble, 1e-8, 1 - 1e-8)
    ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
    
    def log_loss_with_temp(T: float) -> float:
        if T <= 0:
            return 1e10
        
        logits = np.log(ensemble)
        scaled_logits = logits / T
        scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        
        exp_logits = np.exp(scaled_logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        
        return log_loss(y_true, probs)
    
    result = minimize_scalar(
        log_loss_with_temp,
        bounds=(0.5, 2.0),
        method="bounded",
        options={"xatol": 1e-6}
    )
    
    optimal_temp = float(result.x)
    optimal_loss = float(result.fun)
    
    if verbose:
        print(f"\nBTTS Temperature Optimization")
        print(f"  Optimal temperature: {optimal_temp:.6f}")
        print(f"  OOF log loss: {optimal_loss:.6f}")
    
    return {
        "temperature": optimal_temp,
        "log_loss": optimal_loss
    }


def optimize_ensemble_weights_1x2(oof_predictions: dict,
                                  verbose: bool = True) -> dict:
    """
    Optimize 1X2 ensemble weights using OOF data.
    
    Minimizes log loss subject to: sum(w) = 1, 0 <= w_i <= 1
    """
    xgb_probs = oof_predictions["xgb_result"]
    rf_probs = oof_predictions["rf_result"]
    poisson_probs = oof_predictions["poisson_result"]
    actuals = oof_predictions["actuals_result"]
    
    label_map = {"1": 0, "X": 1, "2": 2}
    y_true = np.array([label_map[label] for label in actuals])
    
    # Models available
    models = {}
    if len(xgb_probs) > 0:
        models["xgb"] = xgb_probs
    if len(rf_probs) > 0:
        models["rf"] = rf_probs
    if len(poisson_probs) > 0:
        models["poisson"] = poisson_probs
    
    n_models = len(models)
    if n_models == 0:
        raise ValueError("No predictions available")
    if n_models == 1:
        model_name = list(models.keys())[0]
        return {
            "weights": {model_name: 1.0},
            "log_loss": float(log_loss(y_true, models[model_name]))
        }
    
    model_names = list(models.keys())
    model_arrays = [models[name] for name in model_names]
    
    def objective(w):
        # Weighted ensemble
        ensemble = np.zeros_like(model_arrays[0], dtype=np.float64)
        for i, arr in enumerate(model_arrays):
            ensemble += w[i] * arr
        
        # Normalize
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
        ensemble = np.clip(ensemble, 1e-8, 1 - 1e-8)
        
        return log_loss(y_true, ensemble)
    
    # Constraints: sum to 1
    from scipy.optimize import LinearConstraint, Bounds
    constraints = LinearConstraint(np.ones((1, n_models)), [1.0], [1.0])
    bounds = Bounds(lb=np.zeros(n_models), ub=np.ones(n_models))
    
    # Initial: equal weights
    x0 = np.ones(n_models) / n_models
    
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )
    
    weights = {model_names[i]: result.x[i] for i in range(n_models)}
    final_loss = objective(result.x)
    
    if verbose:
        print(f"\n1X2 Ensemble Weights Optimization")
        for model, w in weights.items():
            print(f"  {model}: {w:.4f}")
        print(f"  OOF log loss: {final_loss:.6f}")
    
    return {
        "weights": weights,
        "log_loss": float(final_loss)
    }


def optimize_ensemble_weights_btts(oof_predictions: dict,
                                   verbose: bool = True) -> dict:
    """
    Optimize BTTS ensemble weights using OOF data.
    """
    xgb_probs = oof_predictions["xgb_btts"]
    rf_probs = oof_predictions["rf_btts"]
    poisson_probs = oof_predictions["poisson_btts"]
    actuals = oof_predictions["actuals_btts"]
    
    y_true = np.asarray(actuals, dtype=int)
    
    # Models available
    models = {}
    if len(xgb_probs) > 0:
        models["xgb"] = xgb_probs
    if len(rf_probs) > 0:
        models["rf"] = rf_probs
    if len(poisson_probs) > 0:
        models["poisson"] = poisson_probs
    
    n_models = len(models)
    if n_models == 0:
        raise ValueError("No BTTS predictions available")
    if n_models == 1:
        model_name = list(models.keys())[0]
        return {
            "weights": {model_name: 1.0},
            "log_loss": float(log_loss(y_true, models[model_name]))
        }
    
    model_names = list(models.keys())
    model_arrays = [models[name] for name in model_names]
    
    def objective(w):
        ensemble = np.zeros_like(model_arrays[0], dtype=np.float64)
        for i, arr in enumerate(model_arrays):
            ensemble += w[i] * arr
        
        ensemble = ensemble / ensemble.sum(axis=1, keepdims=True)
        ensemble = np.clip(ensemble, 1e-8, 1 - 1e-8)
        
        return log_loss(y_true, ensemble)
    
    from scipy.optimize import LinearConstraint, Bounds
    constraints = LinearConstraint(np.ones((1, n_models)), [1.0], [1.0])
    bounds = Bounds(lb=np.zeros(n_models), ub=np.ones(n_models))
    
    x0 = np.ones(n_models) / n_models
    
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9}
    )
    
    weights = {model_names[i]: result.x[i] for i in range(n_models)}
    final_loss = objective(result.x)
    
    if verbose:
        print(f"\nBTTS Ensemble Weights Optimization")
        for model, w in weights.items():
            print(f"  {model}: {w:.4f}")
        print(f"  OOF log loss: {final_loss:.6f}")
    
    return {
        "weights": weights,
        "log_loss": float(final_loss)
    }
