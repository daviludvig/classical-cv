# Chess Board Reader

Computer vision pipeline for reading chessboards from images — board detection, piece occupancy, piece color, and piece type classification using a combination of classical CV and deep learning.

**Course:** INE410121 / TRV410001 — Visão Computacional · UFSC  
**Authors:** Davi Ludvig & Julia Macedo  
**Dataset:** [Synthetic Chess Board Images](https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images) (Kaggle, CC0)

---

## Pipeline

```
Image (1280×1280)
    │
    ▼  Gaussian blur + Canny + Hough lines  (classical)
Board corners detected
    │
    ▼  Homography  (classical)
Rectified board (480×480, top-down)
    │
    ▼  Feature voting — std, edge density, texture, center-border diff  (classical)
Occupancy map 8×8
    │
    ▼  ResNet-34 transfer learning  (deep learning)
Piece map: {A1: pawn_w, E4: queen_b, ...}
```

## Results

| Component | Approach | Result |
|---|---|---|
| Board detection | Hough + Homography | Robust |
| 8×8 segmentation | Uniform division | Exact |
| Occupancy | Feature voting | **F1 ≈ 63%** (end-to-end, 60 imgs) / 86% with GT corners |
| Piece color | Fixed HSV-value threshold | ~82% (200 imgs, GT corners) |
| **Piece type** | **ResNet-34 (TL + fine-tuning)** | **F1 = 91%** |

Evaluated on 50 images with GT occupancy (isolates the DL classifier from classical pipeline errors).

## Piece Classifier — Training

The ResNet-34 piece classifier is trained in two phases, both starting from ImageNet-pretrained weights:

1. **Transfer learning** (10 epochs) — the backbone is **frozen** and only the new classification head (12 piece classes) is trained. Fast, and it cannot damage the pretrained features while the head is still random.
2. **Fine-tuning** (15 epochs) — the **whole network is unfrozen** and refined with a much smaller learning rate (lower for early layers, higher for later ones), adapting the features to the chess domain without erasing what was learned.

Training cells (~62 000 labeled piece crops, roughly balanced across the 12 classes) are built by warping each board with its GT corners and saving every occupied square to `outputs/piece_cells/{label}/`. Data augmentation (flips, ±15° rotation, brightness/contrast/saturation jitter) makes the model robust to board orientation and lighting. See `src/piece_classifier.py`.

## Repository Structure

```
classical-cv/
├── notebooks/
│   └── main.ipynb              # Full pipeline walkthrough
├── src/
│   ├── setup.py                # Environment + dataset bootstrap (local & Colab)
│   ├── chess.py                # Board detection, perspective, segmentation, occupancy
│   ├── piece_classifier.py     # ResNet-34 training, inference, evaluation
│   └── utils.py                # Shared utilities
├── models/
│   └── piece_classifier_resnet34.pth  # Trained weights (download separately — see below)
├── docs/
│   ├── intro/                  # Project overview and initial presentation
│   └── andamento/              # Progress report — DL classifier results
└── outputs/                    # Generated figures and metrics (not versioned)
```

## Setup

**Requirements:** Python 3.10+, PyTorch (CUDA recommended for training), OpenCV, NumPy, Matplotlib, ipywidgets.

```bash
pip install torch torchvision opencv-python numpy matplotlib ipywidgets kagglehub python-dotenv
```

**Kaggle credentials** — create `.env` at the repo root (see `.env.example`):

```dotenv
KAGGLE_API_TOKEN=KGAT_your_token_here
```

The dataset (~457 MB) is downloaded automatically on first run via `kagglehub`.

## Trained Model

The trained ResNet-34 weights are not versioned in this repository (binary, ~85 MB).

> **Download link:** _to be defined — see [issue #1](https://github.com/daviludvig/classical-cv/issues/1)_

Place the downloaded file at `models/piece_classifier_resnet34.pth` before running the inference cells.  
To retrain from scratch, delete the file and run the training cell in the notebook.

## Running

Open `notebooks/main.ipynb` and run cells top to bottom. The notebook covers:

1. Dataset loading and sampling
2. Preprocessing and histogram analysis
3. Edge detection and Hough lines
4. Perspective correction (homography)
5. 8×8 segmentation
6. Occupancy detection (classical features + voting)
7. Piece color classification (HSV)
8. Piece type classification (ResNet-34 transfer learning)
9. Evaluation metrics

To run on **Google Colab**, remove the `sys.path.insert` line in the header cell and set `os.environ["KAGGLE_API_TOKEN"]` before calling `setup()`.

## Setup Kaggle credentials

1. Go to <https://www.kaggle.com/settings> → **API** → **Create New Token**
2. Copy the token (starts with `KGAT_...`)
3. `cp .env.example .env` and fill in your token

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch workflow and notebook filter setup.

## Authors

- Davi Ludvig
- Julia Macedo
