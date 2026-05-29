import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)

# Backbone EfficientNet (feature extractor)
class EfficientNetBackbone(nn.Module):

    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        # Charger EfficientNet sans la tête de classification d'origine
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # supprime la couche FC finale
            global_pool="avg",  # global average pooling → vecteur 1D
        )

        self.feature_dim = self.backbone.num_features  # 1792 pour B4

        # Geler les poids si demandé
        if freeze:
            self.geler()

        logger.info(
            f"Backbone '{model_name}' chargé "
            f"(pretrained={pretrained}, frozen={freeze}, "
            f"feature_dim={self.feature_dim})"
        )

    def geler(self):
        """Gèle tous les paramètres du backbone (aucune mise à jour lors du backward)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone gelé ❄️")

    def degeler(self):
        """Dégèle tous les paramètres (fine-tuning complet)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone dégelé 🔥 (fine-tuning activé)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.backbone(x)


# Fonction utilitaire
def get_backbone(
    model_name: str = "efficientnet_b4",
    pretrained: bool = True,
    freeze: bool = True,
) -> EfficientNetBackbone:
    """Instancie et retourne le backbone."""
    return EfficientNetBackbone(model_name, pretrained, freeze)