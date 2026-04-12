# Classical Computer Vision Project

Projects developed for the Computer Vision course (INE410121 / TRV410001) at UFSC,
focusing on classical image processing techniques — no deep learning.

## Current Project: Chessboard Detection & Occupancy Analysis

A classical computer vision system for understanding chessboard state from images.
The pipeline detects the board, corrects perspective via homography, segments an 8x8 grid,
classifies each square as occupied or empty using intensity/edge/texture features,
and detects moves by comparing two frames.

**Dataset:** [Synthetic Chess Board Images](https://www.kaggle.com/datasets/thefamousrat/synthetic-chess-board-images) (Kaggle)

## Topics Covered

- Image preprocessing & filtering (grayscale, Gaussian blur)
- Edge detection (Canny, Sobel)
- Line detection (Hough Transform)
- Contour detection & polygon approximation
- Perspective correction (homography)
- Morphological operations (opening, closing)
- Feature extraction (intensity, edge density, texture variance)
- Threshold-based occupancy classification
- Temporal analysis (move detection via frame comparison)

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/daviludvig/classical-cv.git
cd classical-cv
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Kaggle credentials

The dataset is downloaded automatically via `kagglehub`. To get your API token:

1. Go to <https://www.kaggle.com/settings>
2. Scroll to the **API** section
3. Click **Create New Token**
4. Copy the token shown (starts with `KGAT_...`)

Then configure your `.env`:

```bash
cp .env.example .env
# Open .env and replace KGAT_your_token_here with your token
```

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/main.ipynb
```

The first cell calls `setup()`, which downloads the dataset on first run and caches it locally.

### Export notebook to PDF (without TeX)

If `nbconvert --to pdf` fails due to missing `xelatex`, use:

```bash
bash scripts/export_notebook_pdf.sh notebooks/main.ipynb
```

This script exports to HTML first and then:
- generates PDF automatically if Chrome/Chromium is available
- otherwise leaves the HTML file in `outputs/exports/` for manual Print to PDF

---

## Running on Google Colab

Open `notebooks/main.ipynb` in Colab.
Cell 0 of the notebook handles credentials — fill in your token there before running.

---

## Contributing

Always work on a branch — **never commit directly to `main`**.

```bash
git checkout -b feature/my-feature   # create branch
# ... work ...
git push origin feature/my-feature
# open a Pull Request to main on GitHub
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Project Structure

```text
classical-cv/
├── notebooks/
│   └── main.ipynb          # Main project notebook (chessboard analysis)
├── src/
│   ├── setup.py            # Environment + dataset bootstrap
│   ├── chess.py             # Board detection, segmentation, occupancy, moves
│   └── utils.py            # Shared utilities (save_fig, save_metrics, ...)
├── docs/
│   ├── intro.md            # Detailed project introduction & methodology
│   └── apresentacao.md     # Presentation slides (Marp format)
├── outputs/
│   ├── figures/            # Generated plots (per-run folders, not versioned)
│   └── results/            # Experiment metrics (per-run folders, not versioned)
├── requirements.txt
└── .env.example            # Credentials template
```

> The dataset is downloaded automatically by `kagglehub` and cached at `~/.cache/kagglehub/`. No local `data/` folder needed.

## Authors

- Davi Ludvig
- Julia Macedo