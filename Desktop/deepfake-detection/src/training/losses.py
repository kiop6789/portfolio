
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# 1. Binary Cross Entropy standard
def bce_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = predictions.view(-1)
    targets     = targets.view(-1).float()
    return F.binary_cross_entropy(predictions, targets)


# 2. Focal Loss (dataset déséquilibré)
class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        logger.info(f"FocalLoss initialisée (alpha={alpha}, gamma={gamma})")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(-1)
        targets     = targets.view(-1).float()

        # Clip pour stabilité numérique
        p = torch.clamp(predictions, 1e-7, 1 - 1e-7)

        # BCE de base
        bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)

        # Facteur de modulation (1-p_t)^gamma
        p_t = targets * p + (1 - targets) * (1 - p)
        focal_weight = (1 - p_t) ** self.gamma

        # Poids alpha pour équilibrer les classes
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


# 3. Fonction utilitaire : choisir la loss
def get_loss(
    loss_type: str = "focal",
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    if loss_type == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == "bce":
        return bce_loss
    else:
        raise ValueError(f"loss_type inconnu : {loss_type}. Choisir 'bce' ou 'focal'.")