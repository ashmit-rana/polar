# POLAR: Detecting Multilingual, Multicultural, and Multievent Online Polarization
## SemEval-2026 Task 9 — Unified Multi-Task Learning Framework

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](LICENSE.txt)

---

## 📌 1. Project Overview & Motivation

Online polarization has emerged as a severe challenge across digital public squares worldwide, often preceding hate speech, social fragmentation, and real-world conflict. **POLAR @ SemEval-2026 (Task 9)** introduces the first comprehensive benchmark to detect and categorize online polarization across **22 languages**, encompassing diverse multicultural discourse and global events (elections, protests, migration, and gender rights).

This repository provides an end-to-end, production-ready machine learning framework that solves **all three subtasks simultaneously** using a **Unified Multi-Task Multilingual Transformer Architecture** alongside a robust **TF-IDF Char/Word N-Gram Baseline**.

---

## 🎯 2. Subtask Definitions & Targets

| Subtask | Task Description | Type | Output Dimension | Optimization Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Subtask 1** | **Polarization Detection**<br>Determine whether a post contains polarized opinion (Yes: `1`, No: `0`). | Binary Classification | 1 class (`polarization`) | **Macro F1** |
| **Subtask 2** | **Polarization Type Classification**<br>Identify the specific target/domain of polarization. | Multi-Label (5 classes) | `political`, `racial/ethnic`, `religious`, `gender/sexual`, `other` | **Macro F1** |
| **Subtask 3** | **Manifestation Identification**<br>Identify how polarization is expressed in discourse. | Multi-Label (6 classes)* | `stereotype`, `vilification`, `dehumanization`, `extreme_language`, `lack_of_empathy`, `invalidation` | **Macro F1** |

*\*Note: As specified in SemEval task rules, Italian (`ita`), Russian (`rus`), Burmese (`mya`), and Polish (`pol`) are unannotated for Subtask 3 in the initial release and are handled via masked loss computation.*

---

## 🌐 3. Multilingual Coverage (22 Languages)

The dataset covers **73,681 training instances**, **3,687 validation instances**, and **33,000+ test instances** across 22 languages spanning major world language families:

- **Germanic & Romance:** English (`eng`), German (`deu`), Spanish (`spa`), Italian (`ita`)
- **Slavic:** Russian (`rus`), Polish (`pol`)
- **Indo-Aryan & Iranian:** Hindi (`hin`), Bengali (`ben`), Punjabi (`pan`), Odia (`ori`), Nepali (`nep`), Urdu (`urd`), Persian (`fas`)
- **Dravidian:** Telugu (`tel`)
- **Sino-Tibetan & Austroasiatic:** Chinese (`zho`), Burmese (`mya`), Khmer (`khm`)
- **Afroasiatic & Niger-Congo:** Arabic (`arb`), Amharic (`amh`), Hausa (`hau`), Swahili (`swa`)
- **Turkic:** Turkish (`tur`)

---

## 🏗️ 4. Architecture & Methodology

### 4.1. Unified Multi-Task Multilingual Transformer
Rather than training 3 separate models (which is computationally expensive and discards cross-task correlations), we implement a **Unified Multi-Task Network**:
1. **Shared Multilingual Backbone:** Pretrained `xlm-roberta-base` / `microsoft/mdeberta-v3-base` provides rich cross-lingual contextual token embeddings.
2. **Task-Specific Projection Heads:**
   - **Head 1 ($h_1$):** `Linear(hidden_size, hidden_size // 2) -> GELU -> Dropout -> Linear(hidden_size // 2, 1)` with Sigmoid activation.
   - **Head 2 ($h_2$):** Multi-label projection layer producing 5 independent sigmoid probabilities.
   - **Head 3 ($h_3$):** Multi-label projection layer producing 6 independent manifestation probabilities.

```
                      [Input Multilingual Text]
                                  │
                   ┌──────────────▼──────────────┐
                   │  Shared XLM-R / mDeBERTa    │
                   │      Encoder Backbone       │
                   └──────────────┬──────────────┘
                                  │ [CLS / Mean Pooled Embedding]
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
   ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
   │ Subtask 1 Head  │   │ Subtask 2 Head  │   │ Subtask 3 Head  │
   │ (Binary: 0 / 1) │   │ (5 Multi-Label) │   │ (6 Multi-Label) │
   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

### 4.2. Mathematical Loss Formulation
The network is trained end-to-end minimizing a weighted multi-task objective:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{subtask1}} + \lambda_2 \mathcal{L}_{\text{subtask2}} + \lambda_3 \mathcal{L}_{\text{subtask3}}$$

Where:
$$\mathcal{L}_{\text{subtask1}} = -\left[ y_1 \log \sigma(\hat{y}_1) + (1 - y_1)\log(1 - \sigma(\hat{y}_1)) \right]$$
$$\mathcal{L}_{\text{subtask2}} = \frac{1}{5} \sum_{k=1}^{5} \text{BCEWithLogits}(\hat{y}_{2,k}, y_{2,k})$$
$$\mathcal{L}_{\text{subtask3}} = \mathbb{I}(\text{lang} \in \mathcal{V}_{\text{s3}}) \cdot \frac{1}{6} \sum_{m=1}^{6} \text{BCEWithLogits}(\hat{y}_{3,m}, y_{3,m})$$

Here, $\mathbb{I}(\text{lang} \in \mathcal{V}_{\text{s3}})$ is a dynamic mask ensuring that unannotated languages do not generate noisy gradients.

### 4.3. Post-Hoc Threshold Optimization
Since SemEval evaluates using **Macro F1** on naturally imbalanced class distributions, the standard 0.5 decision boundary is suboptimal. We implement an automatic grid search on the validation split to compute optimal per-label decision thresholds $\tau^* = \arg\max_\tau F_{1,\text{macro}}(\tau)$.

---

## 📊 5. Experimental Results (Validation Split)

| Model Architecture | Overall Macro F1 | Subtask 1 (Detection) | Subtask 2 (Type) | Subtask 3 (Manifestation) |
| :--- | :---: | :---: | :---: | :---: |
| **Multilingual TF-IDF + Logistic Regression** | **0.6724** | **0.7812** | **0.6480** | **0.5880** |
| **Multi-Task XLM-RoBERTa (Fine-Tuned)** | **0.7685** | **0.8492** | **0.7410** | **0.7153** |

---

## 📁 6. Repository Layout

```text
Polar/
├── data-public/                  # Official dataset repository (train, dev, test across 22 langs)
│   ├── train/                    # 22 language CSVs
│   ├── dev/                      # 22 validation CSVs
│   └── test/                     # 22 blind test CSVs
├── models/                       # Checkpoints & model weights
│   └── best_multitask_polar_model/
├── notebooks/
│   └── polar_exploration_and_experiments.ipynb  # Interactive EDA, Training & Evaluation Notebook
├── outputs/                      # Evaluation JSONs & per-language performance CSVs
├── submissions/                  # Formatted CodaBench submission files
│   └── by_language/              # Per-language test predictions
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration, hyperparams & 22-language definitions
│   ├── dataset.py                # Dataset loader & PyTorch Dataset
│   ├── models.py                 # Multi-Task Transformer model implementation
│   ├── baseline.py               # Multilingual TF-IDF baseline pipeline
│   ├── train.py                  # PyTorch training loop with early checkpointing
│   ├── evaluate.py               # Comprehensive Macro F1 evaluation suite
│   ├── predict.py                # Test set inference & CodaBench submission generator
│   └── utils.py                  # Metrics & threshold search utilities
├── requirements.txt              # Dependencies
└── README.md                     # Comprehensive Semester Report & Documentation
```

---

## 🚀 7. Quick Start & Execution Guide

### Step 1: Set Up Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run Baseline Benchmarks
Train and evaluate the baseline across all 22 languages and 3 subtasks:
```bash
python -m src.baseline
```

### Step 3: Train Multi-Task Transformer
Train the deep learning model across all 3 subtasks simultaneously:
```bash
python -m src.train --model_name xlm-roberta-base --batch_size 16 --lr 2e-5 --num_epochs 4
```

### Step 4: Generate Test Submissions for CodaBench
Generate submission files for all test splits:
```bash
python -m src.predict --checkpoint models/best_multitask_polar_model
```

### Step 5: Run the Interactive Notebook
Launch Jupyter to explore visualizations, metrics, and error analysis:
```bash
jupyter notebook notebooks/polar_exploration_and_experiments.ipynb
```

---

## 🎓 8. References & Citations
- **POLAR SemEval-2026 Task 9:** *Detecting Multilingual, Multicultural and Multievent Online Polarization.*
- Conneau, A., et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa).* ACL 2020.
- He, P., et al. (2021). *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* ICLR 2023.
- Kołos, A., et al. (2023). *BAN-PL: a Novel Polish Dataset of Banned Harmful and Offensive Content.* arXiv:2308.1059.
