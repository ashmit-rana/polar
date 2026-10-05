import json
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion
from pathlib import Path
from src.config import (
    TRAIN_DIR, DEV_DIR, TEST_DIR, LANGUAGES, OUTPUT_DIR, SUBMISSIONS_DIR,
    SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS
)
from src.dataset import load_polar_split
from src.utils import compute_all_metrics, compute_subtask1_metrics, compute_subtask2_metrics, compute_subtask3_metrics

def train_and_eval_baseline(languages=None, sample_per_lang=None):
    """
    Train and evaluate a strong multilingual TF-IDF + Logistic Regression baseline
    for all 3 subtasks across all specified languages.
    """
    print("=" * 60)
    print("🚀 Training Multilingual TF-IDF Baseline for All 3 Subtasks")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/4] Loading train and dev datasets...")
    train_df = load_polar_split("train", languages=languages)
    dev_df = load_polar_split("dev", languages=languages)

    if sample_per_lang is not None:
        train_df = train_df.groupby("lang", group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_per_lang), random_state=42)
        )
        print(f"Sampled train set size: {len(train_df)} rows")

    print(f"Total Train Samples: {len(train_df):,}")
    print(f"Total Dev Samples:   {len(dev_df):,}")
    print(f"Languages:           {len(train_df['lang'].unique())} languages")

    # 2. Extract Features
    print("\n[2/4] Vectorizing multilingual text (Word + Char n-grams)...")
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000, sublinear_tf=True))
    ])

    X_train = vectorizer.fit_transform(train_df["text"])
    X_dev = vectorizer.transform(dev_df["text"])

    # Prepare Labels
    y_train_s1 = train_df[SUBTASK_1_COLS[0]].values
    y_dev_s1 = dev_df[SUBTASK_1_COLS[0]].values

    y_train_s2 = train_df[SUBTASK_2_COLS].values
    y_dev_s2 = dev_df[SUBTASK_2_COLS].values

    # For Subtask 3, filter out Italian and Russian for training & eval
    s3_train_mask = ~train_df["lang"].isin(SUBTASK_3_EXCLUDED_LANGS)
    s3_dev_mask = ~dev_df["lang"].isin(SUBTASK_3_EXCLUDED_LANGS)

    X_train_s3 = X_train[s3_train_mask]
    y_train_s3 = train_df.loc[s3_train_mask, SUBTASK_3_COLS].values
    y_dev_s3 = dev_df.loc[s3_dev_mask, SUBTASK_3_COLS].values
    X_dev_s3 = X_dev[s3_dev_mask]

    # 3. Train Subtask Classifiers
    print("\n[3/4] Fitting classifiers for Subtasks 1, 2, and 3...")
    
    # Subtask 1: Polarization Detection
    print("  -> Fitting Subtask 1 (Binary Polarization)...")
    clf_s1 = LogisticRegression(C=1.0, max_iter=500, solver="liblinear", class_weight="balanced", random_state=42)
    clf_s1.fit(X_train, y_train_s1)
    preds_s1 = clf_s1.predict(X_dev)

    # Subtask 2: Polarization Type
    print("  -> Fitting Subtask 2 (Multi-Label Polarization Types: 5 classes)...")
    clf_s2 = MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=500, solver="liblinear", class_weight="balanced", random_state=42))
    clf_s2.fit(X_train, y_train_s2)
    preds_s2 = clf_s2.predict(X_dev)

    # Subtask 3: Manifestation Identification
    print("  -> Fitting Subtask 3 (Multi-Label Manifestation: 6 classes)...")
    clf_s3 = MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=500, solver="liblinear", class_weight="balanced", random_state=42))
    clf_s3.fit(X_train_s3, y_train_s3)
    preds_s3_non_excluded = clf_s3.predict(X_dev_s3)

    # Reconstruct full length preds for subtask 3
    preds_s3 = np.zeros((len(dev_df), len(SUBTASK_3_COLS)), dtype=int)
    preds_s3[s3_dev_mask] = preds_s3_non_excluded

    # 4. Evaluation
    print("\n[4/4] Calculating SemEval Macro F1 Metrics...")
    
    y_true_dict = {
        "subtask1": y_dev_s1,
        "subtask2": y_dev_s2,
        "subtask3": dev_df[SUBTASK_3_COLS].values
    }
    y_pred_dict = {
        "subtask1": preds_s1,
        "subtask2": preds_s2,
        "subtask3": preds_s3
    }

    metrics = compute_all_metrics(y_true_dict, y_pred_dict, languages=dev_df["lang"].tolist())

    print("\n" + "=" * 60)
    print("📊 BASELINE EVALUATION RESULTS (Dev Set)")
    print("=" * 60)
    print(f"⭐ Overall Mean Macro F1:              {metrics['overall_mean_macro_f1']:.4f}")
    print(f"├─ Subtask 1 (Polarization Detection): {metrics['subtask1_macro_f1']:.4f}")
    print(f"├─ Subtask 2 (Polarization Type):      {metrics['subtask2_macro_f1']:.4f}")
    print(f"└─ Subtask 3 (Manifestation):          {metrics['subtask3_macro_f1']:.4f}")
    print("=" * 60)

    # Per-Language Breakdown
    print("\n📍 Per-Language Subtask 1 Macro F1 Breakdown:")
    lang_results = {}
    for lang in dev_df["lang"].unique():
        idx = dev_df["lang"] == lang
        l_s1 = compute_subtask1_metrics(y_dev_s1[idx], preds_s1[idx])["subtask1_macro_f1"]
        l_s2 = compute_subtask2_metrics(y_dev_s2[idx], preds_s2[idx])["subtask2_macro_f1"]
        
        if lang not in SUBTASK_3_EXCLUDED_LANGS:
            l_s3 = compute_subtask3_metrics(dev_df.loc[idx, SUBTASK_3_COLS].values, preds_s3[idx])["subtask3_macro_f1"]
        else:
            l_s3 = None
            
        lang_results[lang] = {
            "subtask1_macro_f1": l_s1,
            "subtask2_macro_f1": l_s2,
            "subtask3_macro_f1": l_s3
        }
        s3_str = f"{l_s3:.4f}" if l_s3 is not None else "N/A (Excluded)"
        print(f"  {lang:<5}: Subtask 1={l_s1:.4f} | Subtask 2={l_s2:.4f} | Subtask 3={s3_str}")

    # Save to JSON
    output_path = OUTPUT_DIR / "baseline_metrics.json"
    with open(output_path, "w") as f:
        json.dump({"overall": metrics, "per_language": lang_results}, f, indent=2)
    print(f"\nSaved metrics report to {output_path}")

    # Generate Test Submissions if test split exists
    test_sub_dir = SUBMISSIONS_DIR / "baseline_by_language"
    test_sub_dir.mkdir(parents=True, exist_ok=True)
    all_test_dfs = []
    
    print("\n🔮 Generating Baseline Test Set Submissions...")
    for lang in LANGUAGES:
        test_file = TEST_DIR / f"{lang}.csv"
        if not test_file.exists():
            continue
        test_df = pd.read_csv(test_file)
        test_df["text"] = test_df["text"].fillna("").astype(str)
        test_df["lang"] = lang
        
        X_test = vectorizer.transform(test_df["text"])
        p_s1 = clf_s1.predict(X_test)
        p_s2 = clf_s2.predict(X_test)
        
        if lang not in SUBTASK_3_EXCLUDED_LANGS:
            p_s3 = clf_s3.predict(X_test)
        else:
            p_s3 = np.zeros((len(test_df), len(SUBTASK_3_COLS)), dtype=int)
            
        out_df = test_df.copy()
        out_df["polarization"] = p_s1.astype(int)
        for idx, col in enumerate(SUBTASK_2_COLS):
            out_df[col] = p_s2[:, idx].astype(int)
        for idx, col in enumerate(SUBTASK_3_COLS):
            out_df[col] = p_s3[:, idx].astype(int)
            
        # Ensure all target columns are int
        for col in ALL_TARGET_COLS:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype(int)
            
        # Drop temporary 'lang' column if not in original test file
        orig_cols = [c for c in test_df.columns if c != "lang"]
        out_df = out_df[orig_cols]
            
        out_df.to_csv(test_sub_dir / f"{lang}.csv", index=False)
        all_test_dfs.append(out_df)
        
    if all_test_dfs:
        full_sub = pd.concat(all_test_dfs, ignore_index=True)
        full_sub_path = SUBMISSIONS_DIR / "polar_baseline_test_submission.csv"
        full_sub.to_csv(full_sub_path, index=False)
        print(f"Consolidated test submission saved to: {full_sub_path}")

    return {
        "vectorizer": vectorizer,
        "clf_s1": clf_s1,
        "clf_s2": clf_s2,
        "clf_s3": clf_s3,
        "metrics": metrics
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_per_lang", type=int, default=None, help="Optional subsample per language for rapid testing")
    args = parser.parse_args()
    train_and_eval_baseline(sample_per_lang=args.sample_per_lang)
