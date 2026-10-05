import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from src.config import (
    DEV_DIR, LANGUAGES, LANGUAGE_NAMES, OUTPUT_DIR,
    SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS
)
from src.dataset import load_polar_split
from src.utils import (
    compute_subtask1_metrics, compute_subtask2_metrics, 
    compute_subtask3_metrics, compute_all_metrics
)

def evaluate_predictions_df(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    save_prefix: str = "eval"
) -> Dict:
    """
    Evaluate predicted DataFrame against ground truth DataFrame across all 3 subtasks
    and per-language breakdown.
    """
    # Ensure aligned IDs
    merged = pd.merge(gold_df, pred_df, on=["id", "lang"], suffixes=("_gold", "_pred"))
    
    y_true_s1 = merged[f"{SUBTASK_1_COLS[0]}_gold"].values
    y_pred_s1 = merged[f"{SUBTASK_1_COLS[0]}_pred"].values
    
    y_true_s2 = merged[[f"{c}_gold" for c in SUBTASK_2_COLS]].values
    y_pred_s2 = merged[[f"{c}_pred" for c in SUBTASK_2_COLS]].values
    
    y_true_s3 = merged[[f"{c}_gold" for c in SUBTASK_3_COLS]].values
    y_pred_s3 = merged[[f"{c}_pred" for c in SUBTASK_3_COLS]].values
    
    languages = merged["lang"].tolist()
    
    y_true_dict = {
        "subtask1": y_true_s1,
        "subtask2": y_true_s2,
        "subtask3": y_true_s3
    }
    y_pred_dict = {
        "subtask1": y_pred_s1,
        "subtask2": y_pred_s2,
        "subtask3": y_pred_s3
    }
    
    overall_metrics = compute_all_metrics(y_true_dict, y_pred_dict, languages=languages)
    
    # Per language breakdown
    lang_metrics = {}
    lang_table_rows = []
    
    for lang in merged["lang"].unique():
        sub = merged[merged["lang"] == lang]
        l_true_s1 = sub[f"{SUBTASK_1_COLS[0]}_gold"].values
        l_pred_s1 = sub[f"{SUBTASK_1_COLS[0]}_pred"].values
        
        l_true_s2 = sub[[f"{c}_gold" for c in SUBTASK_2_COLS]].values
        l_pred_s2 = sub[[f"{c}_pred" for c in SUBTASK_2_COLS]].values
        
        m1 = compute_subtask1_metrics(l_true_s1, l_pred_s1)["subtask1_macro_f1"]
        m2 = compute_subtask2_metrics(l_true_s2, l_pred_s2)["subtask2_macro_f1"]
        
        if lang not in SUBTASK_3_EXCLUDED_LANGS:
            l_true_s3 = sub[[f"{c}_gold" for c in SUBTASK_3_COLS]].values
            l_pred_s3 = sub[[f"{c}_pred" for c in SUBTASK_3_COLS]].values
            m3 = compute_subtask3_metrics(l_true_s3, l_pred_s3)["subtask3_macro_f1"]
            avg_m = np.mean([m1, m2, m3])
        else:
            m3 = np.nan
            avg_m = np.mean([m1, m2])
            
        lang_metrics[lang] = {
            "subtask1_macro_f1": float(m1),
            "subtask2_macro_f1": float(m2),
            "subtask3_macro_f1": float(m3) if not np.isnan(m3) else None,
            "mean_macro_f1": float(avg_m)
        }
        
        lang_table_rows.append({
            "Lang Code": lang,
            "Language": LANGUAGE_NAMES.get(lang, lang),
            "Subtask 1 (Binary)": m1,
            "Subtask 2 (Type)": m2,
            "Subtask 3 (Manifestation)": m3 if not np.isnan(m3) else None,
            "Average F1": avg_m
        })
        
    breakdown_df = pd.DataFrame(lang_table_rows).sort_values(by="Average F1", ascending=False)
    
    # Save results
    results = {
        "overall": overall_metrics,
        "per_language": lang_metrics
    }
    
    out_json = OUTPUT_DIR / f"{save_prefix}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    out_csv = OUTPUT_DIR / f"{save_prefix}_per_language.csv"
    breakdown_df.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 70)
    print(f"📊 EVALUATION SUMMARY ({save_prefix.upper()})")
    print("=" * 70)
    print(f"⭐ Overall Mean Macro F1:              {overall_metrics['overall_mean_macro_f1']:.4f}")
    print(f"├─ Subtask 1 (Polarization Detection): {overall_metrics['subtask1_macro_f1']:.4f}")
    print(f"├─ Subtask 2 (Polarization Type):      {overall_metrics['subtask2_macro_f1']:.4f}")
    print(f"└─ Subtask 3 (Manifestation):          {overall_metrics['subtask3_macro_f1']:.4f}")
    print("=" * 70)
    print("\nPer-Language Performance (Top 5 & Lowest 5):")
    print(breakdown_df.to_string(index=False))
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to predicted CSV")
    parser.add_argument("--split", type=str, default="dev", help="Split to evaluate against (default: dev)")
    parser.add_argument("--save_prefix", type=str, default="eval", help="Prefix for output report files")
    args = parser.parse_args()
    
    gold_df = load_polar_split(args.split)
    pred_df = pd.read_csv(args.pred_csv)
    evaluate_predictions_df(gold_df, pred_df, save_prefix=args.save_prefix)
