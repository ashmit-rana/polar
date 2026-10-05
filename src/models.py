import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple

class PolarMultiTaskModel(nn.Module):
    """
    Unified Multi-Task Multilingual Transformer for POLAR SemEval-2026 Task 9.
    
    Shares a multilingual encoder (e.g. XLM-RoBERTa or mDeBERTa-v3) and branches into
    three task-specific classification heads:
      1. Subtask 1: Binary Polarization Detection (1 logit)
      2. Subtask 2: Multi-Label Polarization Type (5 logits)
      3. Subtask 3: Multi-Label Manifestation Identification (6 logits)
    """
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        dropout_rate: float = 0.2,
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception:
            self.config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=True)
        
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # Task Heads
        # Subtask 1: Polarization Detection (Binary: 1 output with BCEWithLogits)
        self.head_subtask1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

        # Subtask 2: Polarization Type (5 classes)
        self.head_subtask2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 5)
        )

        # Subtask 3: Manifestation (6 classes)
        self.head_subtask3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 6)
        )

        self.loss_weights = loss_weights or {"subtask1": 1.0, "subtask2": 1.0, "subtask3": 1.0}
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subtask1_labels: Optional[torch.Tensor] = None,
        subtask2_labels: Optional[torch.Tensor] = None,
        subtask3_labels: Optional[torch.Tensor] = None,
        subtask3_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooled representation (use first token / [CLS] representation or mean pooling)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over active tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        pooled = self.dropout(pooled)

        # Logits
        logits_s1 = self.head_subtask1(pooled).squeeze(-1)  # [batch_size]
        logits_s2 = self.head_subtask2(pooled)              # [batch_size, 5]
        logits_s3 = self.head_subtask3(pooled)              # [batch_size, 6]

        # Probabilities
        probs_s1 = torch.sigmoid(logits_s1)
        probs_s2 = torch.sigmoid(logits_s2)
        probs_s3 = torch.sigmoid(logits_s3)

        result = {
            "logits_s1": logits_s1,
            "logits_s2": logits_s2,
            "logits_s3": logits_s3,
            "probs_s1": probs_s1,
            "probs_s2": probs_s2,
            "probs_s3": probs_s3,
        }

        # Compute Losses if labels provided
        if subtask1_labels is not None and subtask2_labels is not None and subtask3_labels is not None:
            loss_s1 = self.bce_loss(logits_s1, subtask1_labels)
            loss_s2 = self.bce_loss(logits_s2, subtask2_labels)

            # Subtask 3 loss with language masking (Italian and Russian are masked out)
            raw_loss_s3 = nn.functional.binary_cross_entropy_with_logits(logits_s3, subtask3_labels, reduction="none")
            if subtask3_mask is not None:
                # subtask3_mask shape: [batch_size]
                mask = subtask3_mask.unsqueeze(-1)  # [batch_size, 1]
                loss_s3 = (raw_loss_s3 * mask).sum() / (mask.sum() * 6 + 1e-8)
            else:
                loss_s3 = raw_loss_s3.mean()

            total_loss = (
                self.loss_weights["subtask1"] * loss_s1
                + self.loss_weights["subtask2"] * loss_s2
                + self.loss_weights["subtask3"] * loss_s3
            )

            result.update({
                "loss": total_loss,
                "loss_subtask1": loss_s1.item(),
                "loss_subtask2": loss_s2.item(),
                "loss_subtask3": loss_s3.item(),
            })

        return result
