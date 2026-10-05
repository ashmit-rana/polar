import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from src.config import (
    TRAIN_DIR, DEV_DIR, TEST_DIR, LANGUAGES,
    SUBTASK_1_COLS, SUBTASK_2_COLS, SUBTASK_3_COLS, SUBTASK_3_EXCLUDED_LANGS
)

def load_polar_split(
    split: str = "train",
    languages: Optional[List[str]] = None,
    data_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load and concatenate dataset CSV files for a given split across specified languages.
    
    Args:
        split: "train", "dev", or "test"
        languages: list of language codes to load (defaults to all 22 languages)
        data_root: optional override for data directory
    """
    if data_root is None:
        if split == "train":
            split_dir = TRAIN_DIR
        elif split == "dev":
            split_dir = DEV_DIR
        elif split == "test":
            split_dir = TEST_DIR
        else:
            raise ValueError(f"Unknown split: {split}")
    else:
        split_dir = Path(data_root) / split

    if languages is None:
        languages = LANGUAGES

    dfs = []
    for lang in languages:
        file_path = split_dir / f"{lang}.csv"
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            # Ensure text column is string and drop NaNs
            df["text"] = df["text"].fillna("").astype(str)
            df["lang"] = lang
            
            # Fill NaNs in target columns (for unannotated languages in subtask 3)
            for col in SUBTASK_1_COLS + SUBTASK_2_COLS + SUBTASK_3_COLS:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
                    
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")

    if not dfs:
        raise FileNotFoundError(f"No CSV data found in {split_dir} for languages {languages}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

class PolarMultiTaskDataset(Dataset):
    """
    PyTorch Dataset for Unified Multi-Task Polarization Classification.
    Provides tokenized text inputs and target tensors for all 3 subtasks.
    """
    def __init__(
        self,
        texts: List[str],
        languages: List[str],
        subtask1_labels: Optional[np.ndarray] = None,
        subtask2_labels: Optional[np.ndarray] = None,
        subtask3_labels: Optional[np.ndarray] = None,
        tokenizer = None,
        max_length: int = 128
    ):
        self.texts = list(texts)
        self.languages = list(languages)
        self.subtask1_labels = subtask1_labels
        self.subtask2_labels = subtask2_labels
        self.subtask3_labels = subtask3_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        lang = self.languages[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "lang": lang
        }

        # Labels (if available, e.g. for train/dev)
        if self.subtask1_labels is not None:
            item["subtask1_labels"] = torch.tensor(self.subtask1_labels[idx], dtype=torch.float)
        
        if self.subtask2_labels is not None:
            item["subtask2_labels"] = torch.tensor(self.subtask2_labels[idx], dtype=torch.float)
            
        if self.subtask3_labels is not None:
            item["subtask3_labels"] = torch.tensor(self.subtask3_labels[idx], dtype=torch.float)
            # Mask out subtask 3 loss for Italian and Russian
            item["subtask3_mask"] = torch.tensor(0.0 if lang in SUBTASK_3_EXCLUDED_LANGS else 1.0, dtype=torch.float)

        return item
