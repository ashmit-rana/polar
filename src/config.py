import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data-public"
TRAIN_DIR = DATA_DIR / "train"
DEV_DIR = DATA_DIR / "dev"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Create output dirs if not exist
for d in [OUTPUT_DIR, MODEL_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 22 Covered Languages (ISO 639-3 / 639-2 codes)
LANGUAGES = [
    "amh", "arb", "ben", "deu", "eng", "fas", "hau", "hin", 
    "ita", "khm", "mya", "nep", "ori", "pan", "pol", "rus", 
    "spa", "swa", "tel", "tur", "urd", "zho"
]

LANGUAGE_NAMES = {
    "amh": "Amharic", "arb": "Arabic", "ben": "Bengali", "deu": "German",
    "eng": "English", "fas": "Persian", "hau": "Hausa", "hin": "Hindi",
    "ita": "Italian", "khm": "Khmer", "mya": "Burmese", "nep": "Nepali",
    "ori": "Odia", "pan": "Punjabi", "pol": "Polish", "rus": "Russian",
    "spa": "Spanish", "swa": "Swahili", "tel": "Telugu", "tur": "Turkish",
    "urd": "Urdu", "zho": "Chinese"
}

# Subtask Target Column Definitions
SUBTASK_1_COLS = ["polarization"]

SUBTASK_2_COLS = [
    "political",
    "racial/ethnic",
    "religious",
    "gender/sexual",
    "other"
]

SUBTASK_3_COLS = [
    "stereotype",
    "vilification",
    "dehumanization",
    "extreme_language",
    "lack_of_empathy",
    "invalidation"
]

# Italian, Burmese, Polish, and Russian do not have Subtask 3 annotations in this dataset release
SUBTASK_3_EXCLUDED_LANGS = ["ita", "mya", "pol", "rus"]

ALL_TARGET_COLS = SUBTASK_1_COLS + SUBTASK_2_COLS + SUBTASK_3_COLS

# Training Hyperparameters
DEFAULT_CONFIG = {
    "model_name": "xlm-roberta-base",  # Alternative: "microsoft/mdeberta-v3-base"
    "max_length": 128,
    "batch_size": 16,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "num_epochs": 4,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 2,
    "subtask_loss_weights": {
        "subtask1": 1.0,
        "subtask2": 1.0,
        "subtask3": 1.0
    },
    "seed": 42
}
