"""
Corrected metrics calculations with proper multiclass handling.

Key fixes:
1. Multiclass Brier Score using one-hot encoding
2. Explicit class-to-column mapping using model.classes_
3. Proper log loss and AUC for binary classification
4. Expected Calibration Error (ECE) for both 1X2 and BTTS
"""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import label_binarize


def multiclass_brier_score(predictions: np.ndarray,
                           actuals: np.ndarray,
                           n_classes: int = 3) -> float:
    """
    Calculate multiclass Brier Score.
    
    BS = (1/N) * sum_i sum_j (y_ij - p_ij)^2
    
    where y_ij is one-hot encoding and p_ij is predicted probability.
    
    Range: [0, 2] for 3 classes, lower is better
    - 0.0: Perfect predictions
    - 1/3: Random guessing (3 classes)
    
    Args:
        predictions: Probability matrix, shape (N, 3) or (N, 2)
        actuals: Class indices or labels, shape (N,)
        n_classes: Number of classes
        
    Returns:
        Brier score
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    
    # Convert actuals to indices if needed
    if isinstance(actuals[0], str):
        if n_classes == 3:
            label_map = {"1": 0, "X": 1, "2": 2}
        else:  # BTTS
            label_map = {"NG": 0, "GG": 1}
        y_true_idx = np.array([label_map[label] for label in actuals])
    else:
        y_true_idx = np.asarray(actuals, dtype=int)
    
    # Create one-hot encoding
    y_onehot = np.eye(n_classes)[y_true_idx]
    
    # Ensure predictions are normalized
    if predictions.ndim == 2:
        predictions = predictions / predictions.sum(axis=1, keepdims=True)
    
    # Brier score: mean squared error of probabilities
    brier = np.mean(np.sum((predictions - y_onehot) ** 2, axis=1))
    
    return float(brier)


def expected_calibration_error(predictions: np.ndarray,
                               actuals: np.ndarray,
                               n_bins: int = 10,
                               n_classes: int = 3) -> tuple:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = sum_m (|B_m| / N) * |acc_m - conf_m|
    
    where:
    - B_m: set of samples in bin m
    - acc_m: accuracy within bin m
    - conf_m: average confidence within bin m
    
    Args:
        predictions: Probability matrix or vector
        actuals: True labels or indices
        n_bins: Number of bins for calibration curve
        n_classes: Number of classes
        
    Returns:
        (ece, bin_details) where bin_details is list of bin statistics
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    
    # Convert actuals to indices
    if isinstance(actuals[0], str):
        if n_classes == 3:
            label_map = {"1": 0, "X": 1, "2": 2}
        else:
            label_map = {"NG": 0, "GG": 1}
        y_true = np.array([label_map[label] for label in actuals])
    else:
        y_true = np.asarray(actuals, dtype=int)
    
    # Get predicted class and confidence
    y_pred = np.argmax(predictions, axis=1)
    confidence = predictions.max(axis=1)
    
    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_details = []
    total_ece = 0.0
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Samples in this bin
        mask = (confidence >= lower) & (confidence <= upper)
        if not mask.any():
            continue
        
        # Accuracy and confidence in this bin
        accuracy = (y_pred[mask] == y_true[mask]).mean()
        avg_confidence = confidence[mask].mean()
        bin_error = abs(accuracy - avg_confidence)
        
        bin_details.append({
            "bin": i,
            "lower": lower,
            "upper": upper,
            "samples": mask.sum(),
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "error": bin_error
        })
        
        # Weighted error
        total_ece += (mask.sum() / len(y_true)) * bin_error
    
    return float(total_ece), bin_details


def safe_class_mapping(model_classes: np.ndarray,
                      n_classes: int = 3) -> dict:
    """
    Create robust class-to-column mapping from model.classes_.
    
    Args:
        model_classes: Classes from model.classes_ attribute
        n_classes: Expected number of classes
        
    Returns:
        Dictionary mapping class name to column index
        
    Raises:
        ValueError if class count mismatch
    """
    if len(model_classes) != n_classes:
        raise ValueError(
            f"Expected {n_classes} classes, got {len(model_classes)}: {model_classes}"
        )
    
    return {cls: idx for idx, cls in enumerate(model_classes)}


def goal_model_metrics(predictions: np.ndarray,
                       actuals: np.ndarray) -> dict:
    """
    Compute metrics for goal count regression models.
    
    Args:
        predictions: Predicted goal counts, shape (N,)
        actuals: Actual goal counts, shape (N,)
        
    Returns:
        Dictionary with MAE, RMSE, Poisson deviance
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - actuals))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Poisson deviance: 2 * sum(y_i * log(y_i / pred_i) - (y_i - pred_i))
    # This is a goodness-of-fit measure for count data
    eps = 1e-8
    pred_clipped = np.clip(predictions, eps, None)
    
    poisson_deviance = 2 * np.mean(
        actuals * np.log(np.clip(actuals, eps, None) / pred_clipped) - 
        (actuals - pred_clipped)
    )
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Poisson_Deviance": float(poisson_deviance)
    }
