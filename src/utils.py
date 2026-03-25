"""
Utility functions for saving figures and metrics with timestamped run directories.

Usage in notebooks:
    from src.utils import create_run_dir, save_fig, save_metrics

    RUN_DIR = create_run_dir()
    save_fig("samples", RUN_DIR)
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parent.parent


def create_run_dir(base_dir: str | Path | None = None) -> Path:
    """
    Creates a timestamped output directory under outputs/figures/.
    """
    if base_dir is None:
        base_dir = _REPO_ROOT / "outputs" / "figures"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output dir: {run_dir}")
    return run_dir


def save_metrics(metrics: dict, run_dir: Path) -> Path:
    """
    Saves a metrics dictionary as metrics.json inside outputs/results/<timestamp>/.
    """
    results_dir = _REPO_ROOT / "outputs" / "results" / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {"timestamp": run_dir.name, **metrics}
    path = results_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {path.relative_to(_REPO_ROOT)}")
    return path


def save_fig(name: str, run_dir: Path, fig=None, dpi: int = 150) -> Path:
    """
    Saves a matplotlib figure to run_dir/<name>.png.
    """
    if fig is None:
        fig = plt.gcf()

    path = run_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved → {path.relative_to(_REPO_ROOT)}")
    return path