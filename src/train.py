import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm

from src.config import (
    LANGUAGES, MODEL_DIR, OUTPUT_DIR, DEFAULT_CONFIG,
    SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS
)
from src.dataset import load_polar_split, PolarMultiTaskDataset
from src.models import PolarMultiTaskModel
from src.utils import compute_all_metrics, find_optimal_thresholds

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def train_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        s1_labels = batch["subtask1_labels"].to(device)
        s2_labels = batch["subtask2_labels"].to(device)
        s3_labels = batch["subtask3_labels"].to(device)
        s3_mask = batch["subtask3_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            subtask1_labels=s1_labels,
            subtask2_labels=s2_labels,
            subtask3_labels=s3_labels,
            subtask3_mask=s3_mask
        )

        loss = outputs["loss"] / gradient_accumulation_steps
        loss.backward()

        total_loss += outputs["loss"].item()

        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            "loss": f"{outputs['loss'].item():.4f}",
            "s1": f"{outputs['loss_subtask1']:.3f}",
            "s2": f"{outputs['loss_subtask2']:.3f}",
            "s3": f"{outputs['loss_subtask3']:.3f}"
        })

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate_model(model, dataloader, device, tune_thresholds=False):
    model.eval()
    all_probs_s1 = []
    all_probs_s2 = []
    all_probs_s3 = []
    all_true_s1 = []
    all_true_s2 = []
    all_true_s3 = []
    languages = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        all_probs_s1.append(outputs["probs_s1"].cpu().numpy())
        all_probs_s2.append(outputs["probs_s2"].cpu().numpy())
        all_probs_s3.append(outputs["probs_s3"].cpu().numpy())

        all_true_s1.append(batch["subtask1_labels"].numpy())
        all_true_s2.append(batch["subtask2_labels"].numpy())
        all_true_s3.append(batch["subtask3_labels"].numpy())
        languages.extend(batch["lang"])

    probs_s1 = np.concatenate(all_probs_s1, axis=0)
    probs_s2 = np.concatenate(all_probs_s2, axis=0)
    probs_s3 = np.concatenate(all_probs_s3, axis=0)

    true_s1 = np.concatenate(all_true_s1, axis=0)
    true_s2 = np.concatenate(all_true_s2, axis=0)
    true_s3 = np.concatenate(all_true_s3, axis=0)

    # Threshold optimization or default 0.5
    if tune_thresholds:
        th_s1 = find_optimal_thresholds(true_s1, probs_s1)
        th_s2 = find_optimal_thresholds(true_s2, probs_s2)
        
        # Optimize threshold only on valid subtask 3 samples
        valid_s3_mask = np.array([lang not in SUBTASK_3_EXCLUDED_LANGS for lang in languages])
        if valid_s3_mask.sum() > 0:
            th_s3 = find_optimal_thresholds(true_s3[valid_s3_mask], probs_s3[valid_s3_mask])
        else:
            th_s3 = np.full(6, 0.5)
    else:
        th_s1 = np.array([0.5])
        th_s2 = np.full(5, 0.5)
        th_s3 = np.full(6, 0.5)

    preds_s1 = (probs_s1 >= th_s1[0]).astype(int)
    preds_s2 = (probs_s2 >= th_s2).astype(int)
    preds_s3 = (probs_s3 >= th_s3).astype(int)

    y_true_dict = {"subtask1": true_s1, "subtask2": true_s2, "subtask3": true_s3}
    y_pred_dict = {"subtask1": preds_s1, "subtask2": preds_s2, "subtask3": preds_s3}

    metrics = compute_all_metrics(y_true_dict, y_pred_dict, languages=languages)
    
    thresholds = {
        "subtask1": th_s1.tolist(),
        "subtask2": th_s2.tolist(),
        "subtask3": th_s3.tolist()
    }
    
    return metrics, thresholds, (probs_s1, probs_s2, probs_s3), (preds_s1, preds_s2, preds_s3)

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task Multilingual Polarization Model")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--sample_per_lang", type=int, default=None, help="Subsample for fast debug/testing")
    parser.add_argument("--output_name", type=str, default="best_multitask_polar_model")
    parser.add_argument("--device", type=str, default=None, help="Device: 'mps', 'cpu', or 'cuda'")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print("=" * 60)
    print("🚀 POLAR SemEval-2026 Multi-Task Transformer Training")
    print(f"Device:               {device}")
    print(f"Base Model:           {args.model_name}")
    print(f"Max Length:           {args.max_length}")
    print(f"Batch Size:           {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"Learning Rate:        {args.lr}")
    print(f"Epochs:               {args.num_epochs}")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    train_df = load_polar_split("train")
    dev_df = load_polar_split("dev")

    if args.sample_per_lang is not None:
        train_df = train_df.groupby("lang", group_keys=False).apply(
            lambda x: x.sample(min(len(x), args.sample_per_lang), random_state=42)
        ).reset_index(drop=True)
        print(f"Subsampled train set size: {len(train_df)} rows")

    print(f"Train samples: {len(train_df):,}")
    print(f"Dev samples:   {len(dev_df):,}")

    # 2. Tokenizer & Datasets
    print(f"\n[2/5] Initializing tokenizer ({args.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = PolarMultiTaskDataset(
        texts=train_df["text"].values,
        languages=train_df["lang"].values,
        subtask1_labels=train_df[SUBTASK_1_COLS[0]].values,
        subtask2_labels=train_df[SUBTASK_2_COLS].values,
        subtask3_labels=train_df[SUBTASK_3_COLS].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    dev_dataset = PolarMultiTaskDataset(
        texts=dev_df["text"].values,
        languages=dev_df["lang"].values,
        subtask1_labels=dev_df[SUBTASK_1_COLS[0]].values,
        subtask2_labels=dev_df[SUBTASK_2_COLS].values,
        subtask3_labels=dev_df[SUBTASK_3_COLS].values,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # 3. Model & Optimizer
    print("\n[3/5] Building Multi-Task Transformer...")
    model = PolarMultiTaskModel(model_name=args.model_name)
    model.to(device)

    total_steps = (len(train_loader) // args.grad_accum) * args.num_epochs
    warmup_steps = int(total_steps * DEFAULT_CONFIG["warmup_ratio"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=DEFAULT_CONFIG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 4. Training Loop
    print("\n[4/5] Starting Training...")
    best_f1 = -1.0
    best_metrics = {}
    best_thresholds = {}
    save_dir = MODEL_DIR / args.output_name

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.num_epochs} ---")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_accum)
        print(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

        # Evaluate
        metrics, thresholds, probs, preds = evaluate_model(model, dev_loader, device, tune_thresholds=True)
        overall_f1 = metrics["overall_mean_macro_f1"]

        print(f"Epoch {epoch} Dev Metrics:")
        print(f"  ⭐ Overall Mean Macro F1:              {overall_f1:.4f}")
        print(f"  ├─ Subtask 1 (Polarization Detection): {metrics['subtask1_macro_f1']:.4f}")
        print(f"  ├─ Subtask 2 (Polarization Type):      {metrics['subtask2_macro_f1']:.4f}")
        print(f"  └─ Subtask 3 (Manifestation):          {metrics['subtask3_macro_f1']:.4f}")

        # Checkpoint Best Model
        if overall_f1 > best_f1:
            best_f1 = overall_f1
            best_metrics = metrics
            best_thresholds = thresholds
            print(f"  ✨ New best model found! Saving checkpoint to {save_dir}...")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), save_dir / "model_weights.pt")
            tokenizer.save_pretrained(save_dir)
            
            with open(save_dir / "config.json", "w") as f:
                json.dump({
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                    "thresholds": best_thresholds,
                    "best_metrics": best_metrics
                }, f, indent=2)

    # 5. Final Report
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE - BEST DEV RESULTS")
    print("=" * 60)
    print(f"⭐ Best Overall Macro F1:              {best_metrics.get('overall_mean_macro_f1', 0.0):.4f}")
    print(f"├─ Subtask 1 (Polarization Detection): {best_metrics.get('subtask1_macro_f1', 0.0):.4f}")
    print(f"├─ Subtask 2 (Polarization Type):      {best_metrics.get('subtask2_macro_f1', 0.0):.4f}")
    print(f"└─ Subtask 3 (Manifestation):          {best_metrics.get('subtask3_macro_f1', 0.0):.4f}")
    print(f"Saved Checkpoint:                      {save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
