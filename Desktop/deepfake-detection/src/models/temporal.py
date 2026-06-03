
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


# Modèle temporel LSTM
class TemporalModel(nn.Module):

    def __init__(
        self,
        input_dim: int = 1792,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,       # entrée : (B, T, input_dim)
            bidirectional=True,     # double la dim de sortie → hidden_dim * 2
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2  # bidirectionnel

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"TemporalModel prêt | "
            f"input={input_dim} → LSTM({hidden_dim}×2) → 1"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out : (B, T, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Mean pooling sur la dimension temporelle
        pooled = lstm_out.mean(dim=1)   # (B, hidden_dim * 2)

        score = self.head(pooled)        # (B, 1)
        return score


# ─────────────────────────────────────────────
# Modèle combiné : Spatial + Temporel
# ─────────────────────────────────────────────
class DeepfakeModelComplet(nn.Module):

    def __init__(
        self,
        backbone,
        classifier,
        temporal_model: TemporalModel,
        poids_spatial: float = 0.6,
        poids_temporel: float = 0.4,
    ):
        super().__init__()
        assert abs(poids_spatial + poids_temporel - 1.0) < 1e-6, \
            "Les poids doivent sommer à 1.0"

        self.backbone       = backbone
        self.classifier     = classifier
        self.temporal_model = temporal_model
        self.poids_spatial  = poids_spatial
        self.poids_temporel = poids_temporel

    def forward(
        self,
        frames: torch.Tensor,           # (B, T, 3, 224, 224)
    ) -> dict:
        B, T, C, H, W = frames.shape

        # ── Analyse spatiale (frame par frame) ──
        frames_flat = frames.view(B * T, C, H, W)       # (B*T, 3, H, W)
        features    = self.backbone(frames_flat)          # (B*T, 1792)
        scores_flat = self.classifier.classifier(features)  # (B*T, 1)
        scores_spatial = scores_flat.view(B, T, 1).mean(dim=1)  # (B, 1)

        # ── Analyse temporelle (LSTM) ──
        features_seq = features.view(B, T, -1)          # (B, T, 1792)
        score_temporel = self.temporal_model(features_seq)  # (B, 1)

        # ── Fusion pondérée ──
        score_final = (
            self.poids_spatial  * scores_spatial +
            self.poids_temporel * score_temporel
        )

        return {
            "score_final":    score_final,
            "score_spatial":  scores_spatial,
            "score_temporel": score_temporel,
        }