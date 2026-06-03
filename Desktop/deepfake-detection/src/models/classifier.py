import torch
import torch.nn as nn
import logging

from backbone import get_backbone, EfficientNetBackbone

logger = logging.getLogger(__name__)


# Modèle complet : Backbone + Classifier
class DeepfakeClassifier(nn.Module):

    def __init__(
        self,
        backbone: EfficientNetBackbone = None,
        hidden_dim: int = 512,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # Backbone (gelé par défaut)
        self.backbone = backbone if backbone is not None else get_backbone()
        feature_dim = self.backbone.feature_dim

        # Tête de classification (seule partie entraînée)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),

            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate * 0.75),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),          # sortie entre 0 et 1
        )

        # Initialisation des poids de la tête
        self._init_weights()

        logger.info(
            f"DeepfakeClassifier prêt | "
            f"feature_dim={feature_dim} → hidden={hidden_dim} → 1"
        )

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, 1792)
        scores   = self.classifier(features)  # (B, 1)
        return scores

    def predire(self, x: torch.Tensor, seuil: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            scores = self.forward(x)
        labels = (scores >= seuil).float()
        return scores, labels


# ─────────────────────────────────────────────
# Sauvegarder / Charger le modèle
# ─────────────────────────────────────────────
def sauvegarder_modele(modele: DeepfakeClassifier, chemin: str):
    torch.save(modele.state_dict(), chemin)
    logger.info(f"Modèle sauvegardé : {chemin}")


def charger_modele(chemin: str, device: str = "cpu") -> DeepfakeClassifier:
    modele = DeepfakeClassifier()
    modele.load_state_dict(torch.load(chemin, map_location=device))
    modele.to(device)
    modele.eval()
    logger.info(f"Modèle chargé depuis {chemin} sur {device}")
    return modele