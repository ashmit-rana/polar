import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from src.config import (
    TEST_DIR, SUBMISSIONS_DIR, MODEL_DIR, LANGUAGES, ALL_TARGET_COLS,
    SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS
)
from src.dataset import PolarMultiTaskDataset, load_polar_split
from src.models import PolarMultiTaskModel

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@torch.no_grad()
def predict_split(
    model_checkpoint_dir: Path,
    split_dir: Path = TEST_DIR,
    batch_size: int = 32,
    device: Optional[torch.device] = None
):
    """
    Run inference on test/blind data and produce competition-ready submission files.
    """
    if device is None:
        device = get_device()

    print("=" * 60)
    print("🔮 POLAR Multi-Task Inference & Submission Generator")
    print(f"Loading checkpoint from: {model_checkpoint_dir}")
    print(f"Device:                  {device}")
    print("=" * 60)

    # 1. Load Config & Thresholds
    config_file = model_checkpoint_dir / "config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = json.load(f)
        model_name = cfg.get("model_name", "xlm-roberta-base")
        max_length = cfg.get("max_length", 128)
        thresholds = cfg.get("thresholds", {})
        th_s1 = np.array(thresholds.get("subtask1", [0.5]))
        th_s2 = np.array(thresholds.get("subtask2", [0.5] * 5))
        th_s3 = np.array(thresholds.get("subtask3", [0.5] * 6))
    else:
        model_name = "xlm-roberta-base"
        max_length = 128
        th_s1 = np.array([0.5])
        th_s2 = np.full(5, 0.5)
        th_s3 = np.full(6, 0.5)

    print(f"Using Thresholds:")
    print(f"  Subtask 1: {th_s1}")
    print(f"  Subtask 2: {th_s2}")
    print(f"  Subtask 3: {th_s3}")

    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_dir)
    model = PolarMultiTaskModel(model_name=model_name)
    weights_path = model_checkpoint_dir / "model_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Predict per language
    all_language_dfs = []
    lang_out_dir = SUBMISSIONS_DIR / "by_language"
    lang_out_dir.mkdir(parents=True, exist_ok=True)

    for lang in LANGUAGES:
        test_file = split_dir / f"{lang}.csv"
        if not test_file.exists():
            continue

        df = pd.read_csv(test_file)
        df["text"] = df["text"].fillna("").astype(str)
        df["lang"] = lang

        dataset = PolarMultiTaskDataset(
            texts=df["text"].values,
            languages=df["lang"].values,
            tokenizer=tokenizer,
            max_length=max_length
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        probs_s1_list, probs_s2_list, probs_s3_list = [], [], []

        for batch in tqdm(loader, desc=f"Predicting {lang:<4}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs_s1_list.append(outputs["probs_s1"].cpu().numpy())
            probs_s2_list.append(outputs["probs_s2"].cpu().numpy())
            probs_s3_list.append(outputs["probs_s3"].cpu().numpy())

        probs_s1 = np.concatenate(probs_s1_list, axis=0)
        probs_s2 = np.concatenate(probs_s2_list, axis=0)
        probs_s3 = np.concatenate(probs_s3_list, axis=0)

        preds_s1 = (probs_s1 >= th_s1[0]).astype(int)
        preds_s2 = (probs_s2 >= th_s2).astype(int)
        preds_s3 = (probs_s3 >= th_s3).astype(int)

        # Set Subtask 3 predictions to 0 for excluded languages (ita, rus)
        if lang in SUBTASK_3_EXCLUDED_LANGS:
            preds_s3 = np.zeros_like(preds_s3)

        # Build output dataframe
        out_df = df.copy()
        out_df["polarization"] = preds_s1.astype(int)

        for idx, col in enumerate(SUBTASK_2_COLS):
            out_df[col] = preds_s2[:, idx].astype(int)

        for idx, col in enumerate(SUBTASK_3_COLS):
            out_df[col] = preds_s3[:, idx].astype(int)

        # Ensure all target columns are int
        for col in ALL_TARGET_COLS:
            if col in out_df.columns:
                out_df[col] = out_df[col].astype(int)

        # Drop temporary 'lang' column if not in original test file
        orig_cols = [c for c in df.columns if c != "lang"]
        out_df = out_df[orig_cols]

        # Save per-language submission
        out_df.to_csv(lang_out_dir / f"{lang}.csv", index=False)
        all_language_dfs.append(out_df)

    # 4. Save combined submission
    combined_submission = pd.concat(all_language_dfs, ignore_index=True)
    combined_path = SUBMISSIONS_DIR / "polar_full_test_submission.csv"
    combined_submission.to_csv(combined_path, index=False)

    print("\n" + "=" * 60)
    print("✅ PREDICTIONS GENERATED SUCCESSFULLY")
    print(f"Per-language submissions saved to: {lang_out_dir}")
    print(f"Consolidated test submission:      {combined_path}")
    print(f"Total Rows:                       {len(combined_submission):,}")
    print("=" * 60)

    return combined_submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(MODEL_DIR / "best_multitask_polar_model"))
    parser.add_argument("--split_dir", type=str, default=str(TEST_DIR))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="Device: 'mps', 'cpu', or 'cuda'")
    args = parser.parse_args()

    dev = torch.device(args.device) if args.device else None

    predict_split(
        model_checkpoint_dir=Path(args.checkpoint),
        split_dir=Path(args.split_dir),
        batch_size=args.batch_size,
        device=dev
    )
