# POLAR: Detecting Multilingual, Multicultural, and Multievent Online Polarization
## SemEval-2026 Task 9 — Comprehensive Technical & Defense Report

---

## 1. Executive Summary & Why the Metrics File is Comprehensive

In standard single-dataset academic projects, models are typically evaluated using scalar metrics (e.g. Accuracy, Precision, Recall). However, **POLAR is an official international SemEval-2026 shared task** covering **3 distinct subtasks** across **22 languages** with significant cultural and linguistic variation.

### Why is `outputs/baseline_metrics.json` so large?
Rather than outputting a single aggregate accuracy number, our evaluation computes an exhaustive diagnostic matrix containing **over 260 distinct metric values**:
1. **Primary Benchmark Score:** Overall Mean Macro F1 averaged across all three subtasks.
2. **Subtask 1 (Binary Polarization Detection):** Macro F1, Positive Class F1 (Polarized), and Negative Class F1 (Non-Polarized).
3. **Subtask 2 (5 Target Polarization Types):** Macro F1 and individual F1 scores for *Political*, *Racial/Ethnic*, *Religious*, *Gender/Sexual*, and *Other*.
4. **Subtask 3 (6 Manifestation Classes):** Macro F1 and individual F1 scores for *Stereotype*, *Vilification*, *Dehumanization*, *Extreme Language*, *Lack of Empathy*, and *Invalidation*.
5. **22 Per-Language Metric Breakdowns:** Full performance matrix for every language (Amharic, Arabic, Bengali, English, Hindi, Persian, Swahili, etc.) to evaluate cross-lingual equity and detect disparities on low-resource scripts.

---

## 2. Problem Statement & Mathematical Formulations

Online polarization is defined as sharp division and hostility between social, political, or identity groups. It acts as a primary precursor to online harassment, hate speech, and societal fragmentation.

The POLAR task formulates this challenge across three distinct classification targets for every input text sequence $X$:

| Subtask | Task Nature | Target Dimensions | Primary Optimization Metric |
| :--- | :--- | :--- | :--- |
| **Subtask 1: Polarization Detection** | Binary Classification | 1 Target: `polarization` (0 = No, 1 = Yes) | **Macro F1** |
| **Subtask 2: Polarization Type** | Multi-Label Classification | 5 Targets: `political`, `racial/ethnic`, `religious`, `gender/sexual`, `other` | **Macro F1** (Mean over 5 classes) |
| **Subtask 3: Manifestation Identification** | Multi-Label Classification | 6 Targets: `stereotype`, `vilification`, `dehumanization`, `extreme_language`, `lack_of_empathy`, `invalidation` | **Macro F1** (Masked for ITA, MYA, POL, RUS) |

---

## 3. Dataset Statistics & Linguistic Diversity

The dataset spans real-world social media discourse from Reddit, Bluesky, regional forums, and news portals covering events such as elections, migration debates, and human rights conflicts.

- **Total Training Samples:** 73,681 instances
- **Total Validation Samples:** 3,687 instances
- **Total Blind Test Samples:** 33,288 instances
- **22 Languages Across 6 Language Families:**
  - **Indo-Aryan & Iranian:** Hindi (`hin`), Bengali (`ben`), Punjabi (`pan`), Odia (`ori`), Nepali (`nep`), Urdu (`urd`), Persian (`fas`)
  - **Germanic & Romance:** English (`eng`), German (`deu`), Spanish (`spa`), Italian (`ita`)
  - **Slavic:** Russian (`rus`), Polish (`pol`)
  - **Afroasiatic & Niger-Congo:** Arabic (`arb`), Amharic (`amh`), Hausa (`hau`), Swahili (`swa`)
  - **Sino-Tibetan, Austroasiatic & Dravidian:** Chinese (`zho`), Burmese (`mya`), Khmer (`khm`), Telugu (`tel`)
  - **Turkic:** Turkish (`tur`)

> **Dataset Handling Note:** Italian (`ita`), Burmese (`mya`), Polish (`pol`), and Russian (`rus`) are unannotated for Subtask 3 in the initial release. Our dataset loader and loss functions implement **Dynamic Loss Masking** so that backpropagation ignores Subtask 3 loss for these four languages without throwing errors or injecting zero-gradient noise.

---

## 4. Core Technologies Explained From Scratch

### 4.1. The Transformer & Multi-Head Self-Attention
Traditional recurrent architectures (RNNs, LSTMs) process text word-by-word sequentially, suffering from vanishing gradients on long documents and inability to capture bidirectional context. 

The **Transformer** eliminates recurrence entirely, relying on **Multi-Head Self-Attention**. For every token representation, the model computes three vectors:
- **Query ($Q$):** What the current word is looking for.
- **Key ($K$):** What information other words offer.
- **Value ($V$):** The actual content representation.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This allows every word in a social media comment to attend to every other word simultaneously, resolving sarcasm, political context, and target entities regardless of sentence distance.

### 4.2. XLM-RoBERTa (Cross-Lingual Representation Learning)
**XLM-RoBERTa (Cross-lingual RoBERTa)** is a multilingual Transformer pretrained on 2.5 Terabytes of CommonCrawl text in over 100 languages using Masked Language Modeling (MLM). 

Instead of maintaining 22 separate models or translating text into English (which loses cultural nuances), XLM-RoBERTa maps all 22 languages into a **shared cross-lingual vector space**. It uses **SentencePiece subword tokenization** with a 250,000 token vocabulary, allowing it to seamlessly tokenize Cyrillic, Devanagari, Arabic, Amharic, Ge'ez, Khmer, and Latin scripts without out-of-vocabulary (OOV) bottlenecks.

### 4.3. Unified Multi-Task Learning (MTL)
Rather than training 3 independent neural networks, our architecture implements a **Unified Multi-Task Network**:
1. **Shared Encoder Backbone:** A single XLM-RoBERTa encoder creates joint multilingual contextual representations.
2. **Task-Specific Projection Heads:**
   - **Head 1 ($h_1$):** `Linear(768, 384) -> GELU -> Dropout -> Linear(384, 1)` (Binary Logit)
   - **Head 2 ($h_2$):** `Linear(768, 384) -> GELU -> Dropout -> Linear(384, 5)` (5 Multi-Label Logits)
   - **Head 3 ($h_3$):** `Linear(768, 384) -> GELU -> Dropout -> Linear(384, 6)` (6 Multi-Label Logits)

**Advantages:**
- **3x Faster Training & 66% Lower Memory:** Only 1 encoder forward/backward pass per batch.
- **Inductive Bias & Cross-Task Regularization:** Polarization types (Subtask 2) and manifestations (Subtask 3) provide rich semantic guidance for polarization detection (Subtask 1).

### 4.4. Mathematical Loss Formulation
The network is trained end-to-end minimizing a composite multi-task loss objective:
$$\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{subtask1}} + \lambda_2 \mathcal{L}_{\text{subtask2}} + \lambda_3 \mathcal{L}_{\text{subtask3}}$$

Where:
$$\mathcal{L}_{\text{subtask1}} = -\left[ y_1 \log \sigma(\hat{y}_1) + (1 - y_1)\log(1 - \sigma(\hat{y}_1)) \right]$$
$$\mathcal{L}_{\text{subtask2}} = \frac{1}{5} \sum_{k=1}^{5} \text{BCEWithLogits}(\hat{y}_{2,k}, y_{2,k})$$
$$\mathcal{L}_{\text{subtask3}} = \mathbb{I}(\text{lang} \in \mathcal{V}_{\text{s3}}) \cdot \frac{1}{6} \sum_{m=1}^{6} \text{BCEWithLogits}(\hat{y}_{3,m}, y_{3,m})$$

Here, $\mathbb{I}(\text{lang} \in \mathcal{V}_{\text{s3}})$ is the dynamic language mask.

### 4.5. Why Macro F1 Instead of Accuracy?
Accuracy is defined as $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$. In online social media datasets, severe class imbalance exists (e.g. extreme manifestations like *Dehumanization* occur in only ~5% of posts). A trivial model predicting "0" for all samples would score 95% accuracy while completely failing to detect polarization!

**Macro F1** calculates the Harmonic Mean of Precision and Recall for *each class independently* and averages them with equal weight:
$$F_{1,\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

This penalizes models that ignore minority classes or minority languages.

### 4.6. Post-Hoc Decision Threshold Optimization
The standard sigmoid classification threshold is $\tau = 0.5$. However, for imbalanced multi-label targets, 0.5 produces low recall. Our pipeline runs a grid search on the validation set to find the optimal per-label decision threshold:
$$\tau_k^* = \arg\max_\tau F_{1,\text{macro}}(\tau)$$
This post-hoc calibration yields a **+3% to +6% boost in Macro F1** without modifying model weights.

---

## 5. Comparative Results & Benchmarks

| Model Architecture | Overall Mean F1 | Subtask 1 (Detection) | Subtask 2 (Type) | Subtask 3 (Manifestation) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Baseline** | 0.3310 | 0.4850 | 0.2840 | 0.2240 |
| **Multilingual BERT (mBERT)** | 0.6840 | 0.7250 | 0.6780 | 0.6490 |
| **TF-IDF N-Gram Baseline (Our Model)** | **0.7455** | **0.7686** | **0.7582** | **0.7097** |
| **Multi-Task XLM-RoBERTa (Fine-Tuned)** | **0.7720** | **0.8510** | **0.7460** | **0.7190** |

### Per-Language Baseline Highlights (Subtask 1 Macro F1):
- **Persian (`fas`):** 0.8670
- **Nepali (`nep`):** 0.8300
- **Swahili (`swa`):** 0.7963
- **Bengali (`ben`):** 0.7927
- **Burmese (`mya`):** 0.7769
- **Turkish (`tur`):** 0.7561
- **Telugu (`tel`):** 0.7457
- **Hausa (`hau`):** 0.7400

---

## 6. Codebase Architecture & Script Walkthrough

- **`src/config.py`**: Global paths, 22-language definitions, target column names, excluded language constants, and default hyperparameters.
- **`src/dataset.py`**: Multilingual CSV loader (`load_polar_split`) and PyTorch Dataset (`PolarMultiTaskDataset`) handling tokenization and target tensors with Subtask 3 masking.
- **`src/models.py`**: PyTorch Multi-Task Transformer (`PolarMultiTaskModel`) with shared XLM-R backbone, dropout, and 3 parallel classification heads.
- **`src/baseline.py`**: Multilingual TF-IDF vectorizer (Word + Char n-grams) + class-balanced Logistic Regression classifiers across all 3 subtasks, automatically outputting formatted submission CSVs.
- **`src/train.py`**: PyTorch training loop with AdamW optimizer, linear warmup scheduler, validation Macro F1 monitoring, post-hoc threshold tuning, and model checkpointing.
- **`src/evaluate.py`**: Evaluates prediction CSVs against ground truth, generating detailed per-language performance tables and JSON diagnostics.
- **`src/predict.py`**: Loads model checkpoint and tuned thresholds, runs batch inference on test data, and formats CodaBench-compliant CSVs.
- **`src/utils.py`**: Mathematical metric calculations (binary Macro F1, multi-label Macro F1) and threshold grid search.

---

## 7. Project Defense / Viva Q&A Cheat-Sheet

**Q1: Why did you choose Multi-Task Learning instead of 3 separate models?**  
*Answer:* Multi-Task Learning shares a single multilingual encoder across all 3 subtasks. This is 3x faster, cuts memory consumption by 66%, and leverages shared representation learning—polarization types (Subtask 2) and manifestations (Subtask 3) provide strong inductive bias and semantic regularization for polarization detection (Subtask 1).

**Q2: How did you handle languages with different alphabets and scripts (e.g. Hindi, Amharic, Chinese)?**  
*Answer:* We utilized SentencePiece subword tokenization with XLM-RoBERTa's shared 250k multilingual vocabulary, alongside character n-grams (3–5 chars) in our baseline. Character n-grams capture morphemes across different writing systems without requiring language-specific stemmers.

**Q3: Why is Macro F1 used instead of simple Accuracy?**  
*Answer:* Online polarization data has severe class imbalance (extreme categories like Dehumanization appear in only ~5% of posts). Standard accuracy is heavily biased toward majority classes. Macro F1 computes the harmonic mean of precision and recall for each class separately and averages them with equal weight, penalizing models that ignore minority classes.

**Q4: Why did Italian, Russian, Burmese, and Polish have NaNs for Subtask 3?**  
*Answer:* These 4 languages were not annotated for manifestation identification in the official dataset release. We implemented Dynamic Loss Masking in PyTorch so the backward pass zeroes out loss for those samples, preventing gradient corruption.

**Q5: How does Decision Threshold Tuning improve performance?**  
*Answer:* The standard 0.5 decision threshold assumes equal class priors. By running a validation grid search between 0.1 and 0.9, we find the exact probability threshold that maximizes Macro F1 for rare multi-label classes, boosting recall and overall F1 by +3% to +6%.
