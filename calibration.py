"""
Probability calibration module.

Calibrates model predictions to match observed frequencies using:
1. Temperature scaling
2. Isotonic regression (optional)
3. Empirical calibration curves

Ref: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

import numpy as np
from scipy.optimize import curve_fit
from config import (
    CALIBRATION_TEMPERATURE,
    CALIBRATION_MIN_PROB,
    CALIBRATION_MAX_PROB
)


def temperature_scale(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to logits.
    
    Higher temperature -> softer probabilities (more uncertain).
    Lower temperature -> sharper probabilities (more confident).
    
    Args:
        logits: Raw model scores (unbounded)
        temperature: Scaling parameter (> 0)
        
    Returns:
        Calibrated probabilities
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits / temperature
    logits = logits - logits.max()  # Numerical stability
    
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    return probs


def calibrate_probabilities(probs: np.ndarray,
                            temperature: float = CALIBRATION_TEMPERATURE,
                            clip: bool = True) -> np.ndarray:
    """
    Calibrate probability vector using temperature scaling.
    
    Args:
        probs: Probability vector or matrix
        temperature: Temperature for scaling
        clip: Whether to clip to valid probability range
        
    Returns:
        Calibrated probabilities
    """
    probs = np.asarray(probs, dtype=np.float64)
    
    # Clip before temperature scaling
    if clip:
        probs = np.clip(probs, CALIBRATION_MIN_PROB, CALIBRATION_MAX_PROB)
    
    # Handle single probability (scalar case)
    if probs.ndim == 0:
        probs = np.atleast_1d(probs)
        scalar_input = True
    else:
        scalar_input = False
    
    # For vectors, apply temp scaling
    if probs.ndim == 1:
        log_probs = np.log(probs)
        log_probs = log_probs / temperature
        log_probs = log_probs - log_probs.max()
        
        calibrated = np.exp(log_probs)
        calibrated = calibrated / calibrated.sum()
    else:
        # For matrices, apply row-wise
        calibrated = temperature_scale(np.log(probs), temperature=1.0)
    
    if clip:
        calibrated = np.clip(calibrated, CALIBRATION_MIN_PROB, CALIBRATION_MAX_PROB)
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    
    return calibrated[0] if scalar_input else calibrated


def brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """
    Calculate Brier Score (mean squared error of probabilities).
    
    Brier = (1/N) * sum((pred_i - actual_i)^2)
    
    Lower is better. Range: [0, 1]
    - 0.0: Perfect calibration
    - 0.25: Random guessing (binary classification)
    - 1.0: Worst possible
    
    Args:
        predictions: Predicted probabilities, shape (N, C) or (N,)
        actuals: True one-hot labels, shape (N, C) or (N,)
        
    Returns:
        Brier score
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    if predictions.shape != actuals.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {actuals.shape}")
    
    return float(np.mean((predictions - actuals) ** 2))


def calibration_error(predictions: np.ndarray, actuals: np.ndarray,
                      n_bins: int = 10) -> tuple:
    """
    Calculate expected calibration error (ECE).
    
    Divides predictions into N bins by confidence, computes calibration error per bin.
    
    Args:
        predictions: Predicted probabilities (one class per column)
        actuals: True labels (one-hot or class indices)
        n_bins: Number of confidence bins
        
    Returns:
        (ece, bin_results) where bin_results is list of dicts with bin stats
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    # Get maximum probability per sample
    if predictions.ndim == 2:
        max_probs = predictions.max(axis=1)
        pred_classes = predictions.argmax(axis=1)
    else:
        max_probs = predictions
        pred_classes = np.round(predictions).astype(int)
    
    # Convert actuals to class indices if one-hot
    if actuals.ndim == 2:
        actual_classes = actuals.argmax(axis=1)
    else:
        actual_classes = actuals.astype(int)
    
    bin_results = []
    total_error = 0.0
    total_samples = 0
    
    # Create bins by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        # Find samples in this confidence bin
        in_bin = (max_probs >= lower) & (max_probs < upper)
        if i == n_bins - 1:
            in_bin = (max_probs >= lower) & (max_probs <= upper)  # Include upper bound
        
        if in_bin.sum() == 0:
            continue
        
        bin_preds = max_probs[in_bin]
        bin_actuals = (pred_classes[in_bin] == actual_classes[in_bin]).astype(float)
        
        avg_confidence = bin_preds.mean()
        accuracy = bin_actuals.mean()
        bin_error = abs(avg_confidence - accuracy)
        bin_samples = in_bin.sum()
        
        bin_results.append({
            'bin': i,
            'lower': lower,
            'upper': upper,
            'avg_confidence': avg_confidence,
            'accuracy': accuracy,
            'error': bin_error,
            'samples': int(bin_samples)
        })
        
        total_error += bin_error * bin_samples
        total_samples += bin_samples
    
    ece = total_error / max(total_samples, 1)
    
    return float(ece), bin_results
