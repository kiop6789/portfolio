
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


# 1. Accuracy
def calculer_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    seuil: float = 0.5,
) -> float:
    preds_binaires = (predictions.cpu().numpy() >= seuil).astype(int)
    labels         = targets.cpu().numpy().astype(int)
    return float(accuracy_score(labels, preds_binaires))


# 2. AUC-ROC
def calculer_auc(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    scores = predictions.cpu().numpy().flatten()
    labels = targets.cpu().numpy().astype(int).flatten()

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        # Si une seule classe dans le batch (rare en début d'entraînement)
        logger.warning("AUC non calculable (une seule classe dans le batch)")
        auc = 0.0

    return float(auc)


# 3. F1-Score
def calculer_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    seuil: float = 0.5,
) -> float:
    preds_binaires = (predictions.cpu().numpy() >= seuil).astype(int)
    labels         = targets.cpu().numpy().astype(int)
    return float(f1_score(labels, preds_binaires, zero_division=0))


# 4. Toutes les métriques d'un coup
def calculer_metriques(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    seuil: float = 0.5,
) -> dict:
    return {
        "accuracy": calculer_accuracy(predictions, targets, seuil),
        "auc":      calculer_auc(predictions, targets),
        "f1":       calculer_f1(predictions, targets, seuil),
    }