import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
from src.config import SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS

def compute_subtask1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Macro F1 and class-wise metrics for Subtask 1 (Polarization Detection)."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    binary_f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    return {
        "subtask1_macro_f1": float(macro_f1),
        "subtask1_f1_pos": float(f[1]) if len(f) > 1 else 0.0,
        "subtask1_f1_neg": float(f[0]) if len(f) > 0 else 0.0,
        "subtask1_binary_f1": float(binary_f1)
    }

def compute_subtask2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute Macro F1 and per-label F1 for Subtask 2 (Polarization Type)."""
    # Average Macro F1 across the 5 target classes
    per_class_f1 = []
    res = {}
    for idx, col in enumerate(SUBTASK_2_COLS):
        # Macro F1 for binary classification on each label
        col_macro = f1_score(y_true[:, idx], y_pred[:, idx], average="macro", zero_division=0)
        col_pos_f1 = f1_score(y_true[:, idx], y_pred[:, idx], average="binary", zero_division=0)
        per_class_f1.append(col_macro)
        res[f"subtask2_{col}_macro_f1"] = float(col_macro)
        res[f"subtask2_{col}_f1"] = float(col_pos_f1)
        
    res["subtask2_macro_f1"] = float(np.mean(per_class_f1))
    return res

def compute_subtask3_metrics(y_true: np.ndarray, y_pred: np.ndarray, lang_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute Macro F1 and per-label F1 for Subtask 3 (Manifestation), filtering excluded languages."""
    if lang_mask is not None:
        y_true = y_true[lang_mask]
        y_pred = y_pred[lang_mask]
        
    if len(y_true) == 0:
        return {"subtask3_macro_f1": 0.0}
        
    per_class_f1 = []
    res = {}
    for idx, col in enumerate(SUBTASK_3_COLS):
        col_macro = f1_score(y_true[:, idx], y_pred[:, idx], average="macro", zero_division=0)
        col_pos_f1 = f1_score(y_true[:, idx], y_pred[:, idx], average="binary", zero_division=0)
        per_class_f1.append(col_macro)
        res[f"subtask3_{col}_macro_f1"] = float(col_macro)
        res[f"subtask3_{col}_f1"] = float(col_pos_f1)
        
    res["subtask3_macro_f1"] = float(np.mean(per_class_f1))
    return res

def compute_all_metrics(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    languages: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute overall evaluation metrics across all 3 subtasks."""
    m1 = compute_subtask1_metrics(y_true_dict["subtask1"], y_pred_dict["subtask1"])
    m2 = compute_subtask2_metrics(y_true_dict["subtask2"], y_pred_dict["subtask2"])
    
    # Filter out Italian and Russian for Subtask 3
    if languages is not None:
        s3_valid_mask = np.array([lang not in SUBTASK_3_EXCLUDED_LANGS for lang in languages])
    else:
        s3_valid_mask = None
        
    m3 = compute_subtask3_metrics(y_true_dict["subtask3"], y_pred_dict["subtask3"], s3_valid_mask)
    
    overall = {
        **m1,
        **m2,
        **m3,
        "overall_mean_macro_f1": float(np.mean([
            m1["subtask1_macro_f1"],
            m2["subtask2_macro_f1"],
            m3["subtask3_macro_f1"]
        ]))
    }
    return overall

def find_optimal_thresholds(y_true: np.ndarray, y_probs: np.ndarray, grid=np.linspace(0.1, 0.9, 17)) -> np.ndarray:
    """Find threshold that maximizes Macro F1 for binary/multi-label probabilities."""
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_probs = y_probs[:, None]
        
    num_cols = y_true.shape[1]
    best_thresholds = np.full(num_cols, 0.5)
    
    for c in range(num_cols):
        best_f1 = -1.0
        for th in grid:
            preds = (y_probs[:, c] >= th).astype(int)
            f1 = f1_score(y_true[:, c], preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = th
                
    return best_thresholds
