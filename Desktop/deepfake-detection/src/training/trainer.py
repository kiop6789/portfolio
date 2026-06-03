import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import logging
import time

from losses import get_loss
from metrics import calculer_metriques

logger = logging.getLogger(__name__)


# Boucle d'entraînement (1 epoch)
def train_epoch(
    modele: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> dict:
    modele.train()

    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # ── 1. Prédiction ──
        scores = modele(images).squeeze(1)  # (B,)

        # ── 2. Loss ──
        loss = loss_fn(scores, labels)

        # ── 3. Backpropagation ──
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (stabilité)
        torch.nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)

        # ── 4. Optimisation ──
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(scores.detach().cpu())
        all_targets.append(labels.detach().cpu())

        if batch_idx % 20 == 0:
            logger.debug(f"  Batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")

    # Concaténer tous les batchs
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metriques = calculer_metriques(all_preds, all_targets)
    metriques["loss"] = total_loss / len(loader)
    return metriques


# Boucle d'évaluation (validation / test)
@torch.no_grad()
def eval_epoch(
    modele: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict:
    modele.eval()

    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        scores = modele(images).squeeze(1)
        loss   = loss_fn(scores, labels)

        total_loss += loss.item()
        all_preds.append(scores.cpu())
        all_targets.append(labels.cpu())

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metriques = calculer_metriques(all_preds, all_targets)
    metriques["loss"] = total_loss / len(loader)
    return metriques


# Entraînement complet (toutes les epochs)
def entrainer(
    modele: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    loss_type: str = "focal",
    dossier_sauvegarde: str = "models/checkpoints",
    device_str: str = "auto",
) -> dict:
    # ── Device ──
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Device : {device}")

    modele = modele.to(device)

    # ── Optimizer & Scheduler ──
    # On n'optimise que les paramètres qui nécessitent un gradient (le classifier)
    params_a_entrainer = [p for p in modele.parameters() if p.requires_grad]
    optimizer = AdamW(params_a_entrainer, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ── Loss ──
    loss_fn = get_loss(loss_type)

    # ── Dossier de sauvegarde ──
    Path(dossier_sauvegarde).mkdir(parents=True, exist_ok=True)

    historique = {"train": [], "val": []}
    meilleur_auc  = 0.0
    meilleur_epoch = 0

    logger.info(f"Début entraînement : {num_epochs} epochs | lr={lr} | loss={loss_type}")
    logger.info("=" * 60)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Entraînement ──
        train_metrics = train_epoch(modele, train_loader, optimizer, loss_fn, device)

        # ── Validation ──
        val_metrics = eval_epoch(modele, val_loader, loss_fn, device)

        # ── Scheduler ──
        scheduler.step()

        duree = time.time() - t0

        historique["train"].append(train_metrics)
        historique["val"].append(val_metrics)

        logger.info(
            f"Epoch {epoch:02d}/{num_epochs} ({duree:.1f}s) | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} auc={train_metrics['auc']:.3f} | "
            f"Val   loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.3f}  auc={val_metrics['auc']:.3f}  f1={val_metrics['f1']:.3f}"
        )

        # ── Sauvegarder le meilleur modèle (critère : AUC validation) ──
        if val_metrics["auc"] > meilleur_auc:
            meilleur_auc   = val_metrics["auc"]
            meilleur_epoch = epoch
            chemin = Path(dossier_sauvegarde) / "best_model.pth"
            torch.save(modele.state_dict(), str(chemin))
            logger.info(f"  ✅ Meilleur modèle sauvegardé (AUC={meilleur_auc:.4f})")

    logger.info("=" * 60)
    logger.info(f"Entraînement terminé. Meilleur AUC={meilleur_auc:.4f} à l'epoch {meilleur_epoch}")
    return historique