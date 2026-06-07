"""
Chess piece type classification using ResNet-34 transfer learning.

Requires: torch, torchvision  (pip install torch torchvision)
These are pre-installed on Google Colab.

Labels (12 classes): bishop_b, bishop_w, king_b, king_w, knight_b, knight_w,
                     pawn_b, pawn_w, queen_b, queen_w, rook_b, rook_w
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PIECE_LABELS = sorted([
    "bishop_b", "bishop_w",
    "king_b",   "king_w",
    "knight_b", "knight_w",
    "pawn_b",   "pawn_w",
    "queen_b",  "queen_w",
    "rook_b",   "rook_w",
])
NUM_CLASSES = len(PIECE_LABELS)  # 12
LABEL_TO_IDX = {label: i for i, label in enumerate(PIECE_LABELS)}

# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def extract_piece_dataset(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_samples: int | None = None,
    dst_size: int = 480,
    cell_size: int = 64,
) -> dict[str, int]:
    """Extract labeled cell images from the chess dataset for classifier training.

    For each image in the dataset:
      - Warps board to top-down view using GT corners
      - Detects board orientation (finds which warped corner is A8)
      - Saves each occupied cell as output_dir/{label}/{img_id}_{square}.jpg

    Args:
        dataset_dir: path to the Kaggle dataset directory.
        output_dir:  root directory for labeled images (created if needed).
        max_samples: limit number of images processed (None = all).
        dst_size:    side length of the warped board image in pixels.
        cell_size:   side length of the saved cell images (resized).

    Returns:
        counts dict: {label: number_of_saved_images}
    """
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda x, **kw: x  # noqa: E731

    from src.chess import (
        list_samples, load_annotation, annotation_path_for,
        corners_to_pixels, compute_homography, warp_board,
        segment_grid, square_to_grid, cell_features,
    )

    samples = list_samples(dataset_dir)
    if max_samples is not None:
        samples = samples[:max_samples]

    output_dir = Path(output_dir)
    counts: dict[str, int] = {}
    skipped = 0

    for img_path in _tqdm(samples, desc="Extracting cells"):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        ann = load_annotation(annotation_path_for(img_path))
        if not ann.get("config") or not ann.get("corners"):
            skipped += 1
            continue

        # Warp board to top-down view using GT corners
        corners_px = corners_to_pixels(ann["corners"], img.shape)
        H = compute_homography(corners_px, dst_size)
        warped = warp_board(img, H, dst_size)
        cells = segment_grid(warped)

        # Find orientation: pick the rotation k where annotated squares
        # have the highest aggregate edge density (occupied cells are textured)
        k = _find_rotation_from_occupancy(cells, ann["config"], square_to_grid, cell_features)

        # Save each piece cell
        for square, piece in ann["config"].items():
            if piece not in LABEL_TO_IDX:
                continue
            warp_r, warp_c = _std_to_warped(square_to_grid(square), k)
            if not (0 <= warp_r < 8 and 0 <= warp_c < 8):
                continue

            cell = cells[warp_r][warp_c]
            if cell_size != cell.shape[0] or cell_size != cell.shape[1]:
                cell = cv2.resize(cell, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

            out_dir = output_dir / piece
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{img_path.stem}_{square}.jpg"
            cv2.imwrite(str(out_path), cell)
            counts[piece] = counts.get(piece, 0) + 1

    if skipped:
        print(f"  Skipped {skipped} images (read error or missing annotation).")
    return counts


def _std_to_warped(std_pos: tuple[int, int], k: int) -> tuple[int, int]:
    """Map standard grid (row, col) to the pre-rotation warped grid position.

    np.rot90(warped_occ, k) gives standard orientation, so this is the
    inverse: given (r, c) in standard, return the position in warped.

    Standard: row 0 = rank 8 (top), col 0 = file A (left).
    """
    r, c = std_pos
    N = 7  # max index for 8×8 grid
    if k == 0: return r, c
    if k == 1: return c, N - r
    if k == 2: return N - r, N - c
    if k == 3: return N - c, r
    return r, c


def _find_rotation_from_occupancy(
    cells: list[list[np.ndarray]],
    config: dict,
    square_to_grid_fn,
    cell_features_fn,
) -> int:
    """Find the rotation k that best aligns annotated squares with high-texture cells.

    For each candidate rotation, accumulates edge_density + texture_var from
    cells at the annotated positions.  The correct rotation maximises this
    score because occupied cells are much more textured than empty ones.
    """
    scores = [0.0] * 4
    for square in config:
        std_pos = square_to_grid_fn(square)
        for k in range(4):
            wr, wc = _std_to_warped(std_pos, k)
            if 0 <= wr < 8 and 0 <= wc < 8:
                feats = cell_features_fn(cells[wr][wc])
                scores[k] += feats["edge_density"] + feats["texture_var"] * 0.001
    return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Build a ResNet-34 with a modified classification head.

    Transfer learning setup (as taught in Aula 12.2):
      - Backbone loaded with ImageNet weights (frozen by default)
      - Final FC layer replaced to output num_classes logits
      - Call freeze_backbone() before phase-1 training
      - Call unfreeze_backbone() before fine-tuning (phase 2)

    Returns:
        model (torch.nn.Module)
    """
    import torch.nn as nn
    from torchvision import models

    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)

    # Replace the classification head
    in_features = model.fc.in_features  # 512 for ResNet-34
    model.fc = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model) -> None:
    """Freeze all layers except the classification head (phase-1 transfer learning)."""
    import torch.nn as nn
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_backbone(model) -> None:
    """Unfreeze all layers for fine-tuning (phase 2)."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_transforms(train: bool, img_size: int = 224):
    """Return torchvision transforms for training or validation."""
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet statistics
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def make_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    val_split: float = 0.15,
    img_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
):
    """Build train/val DataLoaders from a directory of labelled images.

    Expects data_dir/{label}/*.jpg structure (as produced by extract_piece_dataset).
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    full_dataset = ImageFolder(str(data_dir), transform=get_transforms(train=True, img_size=img_size))
    val_dataset  = ImageFolder(str(data_dir), transform=get_transforms(train=False, img_size=img_size))

    rng = np.random.default_rng(seed)
    n = len(full_dataset)
    indices = rng.permutation(n).tolist()
    n_val = int(n * val_split)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, full_dataset.classes


def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = correct = n = 0
    import torch
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


def train(
    data_dir: Union[str, Path],
    save_path: Union[str, Path],
    phase1_epochs: int = 10,
    phase2_epochs: int = 10,
    batch_size: int = 64,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    img_size: int = 224,
    num_workers: int = 2,
) -> list[dict]:
    """Train a ResNet-34 piece classifier with two-phase transfer learning.

    Phase 1: freeze backbone, train classification head (fast convergence).
    Phase 2: unfreeze all layers, fine-tune end-to-end (slower, lower lr).

    Args:
        data_dir:       directory with {label}/ sub-folders (from extract_piece_dataset).
        save_path:      path to save the best model weights (.pth).
        phase1_epochs:  epochs with frozen backbone.
        phase2_epochs:  epochs with full fine-tuning.
        batch_size:     images per gradient step.
        lr_phase1:      learning rate for phase 1.
        lr_phase2:      learning rate for phase 2.
        img_size:       input resolution fed to ResNet (224 matches ImageNet).
        num_workers:    DataLoader workers.

    Returns:
        history: list of dicts with keys epoch, phase, train_loss, val_loss,
                 train_acc, val_acc.
    """
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, classes = make_dataloaders(
        data_dir, batch_size=batch_size, img_size=img_size, num_workers=num_workers
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_val_acc = 0.0
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for phase, epochs, lr, frozen in [
        ("transfer", phase1_epochs, lr_phase1, True),
        ("finetune", phase2_epochs, lr_phase2, False),
    ]:
        if frozen:
            freeze_backbone(model)
        else:
            unfreeze_backbone(model)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"\n--- Phase: {phase} ({epochs} epochs, lr={lr}) ---")

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, training=True)
            vl_loss, vl_acc = _run_epoch(model, val_loader, criterion, optimizer, device, training=False)
            scheduler.step()

            rec = dict(epoch=epoch, phase=phase,
                       train_loss=tr_loss, val_loss=vl_loss,
                       train_acc=tr_acc, val_acc=vl_acc)
            history.append(rec)
            print(f"  Ep {epoch:3d} | loss {tr_loss:.4f}/{vl_loss:.4f} | acc {tr_acc:.3f}/{vl_acc:.3f}")

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "img_size": img_size,
                }, str(save_path))
                print(f"    Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_classifier(model_path: Union[str, Path]):
    """Load a trained piece classifier from a .pth checkpoint.

    Returns:
        model (torch.nn.Module in eval mode), img_size (int), classes (list[str])
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(model_path), map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)

    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, img_size, classes


def predict(
    model,
    cell_bgr: np.ndarray,
    img_size: int = 224,
    classes: list[str] | None = None,
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """Classify a single cell image (BGR numpy array from OpenCV).

    Args:
        model:    trained model (from load_classifier).
        cell_bgr: cell image as BGR numpy array.
        img_size: input resolution the model expects.
        classes:  class label list (from load_classifier).
        top_k:    number of top predictions to return.

    Returns:
        list of (label, probability) sorted by probability descending.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if classes is None:
        classes = PIECE_LABELS

    device = next(model.parameters()).device
    tf = get_transforms(train=False, img_size=img_size)

    cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cell_rgb)
    tensor = tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], float(probs[i])) for i in top_indices]


def predict_board(
    model,
    cells: list[list[np.ndarray]],
    occupancy: np.ndarray,
    img_size: int = 224,
    classes: list[str] | None = None,
    confidence_threshold: float = 0.3,
) -> dict[str, str]:
    """Classify all occupied cells on the board.

    Args:
        model:       trained classifier.
        cells:       8×8 list of cell BGR images (from segment_grid).
        occupancy:   8×8 bool array (True = occupied).
        img_size:    model input resolution.
        classes:     class labels.
        confidence_threshold: minimum probability to accept a prediction.

    Returns:
        dict mapping square name (e.g. "E4") to piece label (e.g. "pawn_w").
        Squares with low confidence are omitted.
    """
    from src.chess import cell_name

    result: dict[str, str] = {}
    for row in range(8):
        for col in range(8):
            if not occupancy[row, col]:
                continue
            preds = predict(model, cells[row][col], img_size=img_size, classes=classes, top_k=1)
            label, prob = preds[0]
            if prob >= confidence_threshold:
                result[cell_name(row, col)] = label
    return result


# ---------------------------------------------------------------------------
# Training history visualisation
# ---------------------------------------------------------------------------

def plot_history(history: list[dict]) -> None:
    """Plot training/validation loss and accuracy curves."""
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in history]
    tr_acc = [r["train_acc"] for r in history]
    vl_acc = [r["val_acc"]  for r in history]
    tr_loss = [r["train_loss"] for r in history]
    vl_loss = [r["val_loss"]  for r in history]

    # Mark phase boundary
    phase2_start = next((i for i, r in enumerate(history) if r["phase"] == "finetune"), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tr, vl, ylabel in [
        (ax1, tr_acc,  vl_acc,  "Accuracy"),
        (ax2, tr_loss, vl_loss, "Loss"),
    ]:
        ax.plot(epochs, tr, label="train")
        ax.plot(epochs, vl, label="val")
        if phase2_start:
            ax.axvline(x=history[phase2_start]["epoch"], color="gray",
                       linestyle="--", label="fine-tune start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Piece Classifier — Training History")
    plt.tight_layout()
    plt.show()
