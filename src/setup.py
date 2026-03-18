"""
Environment setup for local and Google Colab execution.

Usage (first cell of any notebook):
    import sys, os
    sys.path.insert(0, os.path.abspath(".."))
    from src.setup import setup
    DATA_DIR = setup()
"""

import os
import sys
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/daviludvig/classical-cv.git"
REPO_NAME = "classical-cv"
DATASET_SLUG = "raidathmane/corrosion-and-spalling-concrete-defect-segmentation"


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_package(package: str) -> None:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def _load_env() -> None:
    """Load .env from the repo root (local only — skipped on Colab)."""
    _ensure_package("dotenv")
    from dotenv import load_dotenv

    # Walk up from this file's location to find the repo root .env
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f".env loaded from {env_file}")
    else:
        print(f"No .env found at {env_file} — using existing environment variables.")


def _clone_repo(target: str = f"/content/{REPO_NAME}") -> str:
    if not os.path.exists(target):
        subprocess.check_call(["git", "clone", REPO_URL, target])
    return target


def _download_dataset() -> str:
    _ensure_package("kagglehub")
    import kagglehub
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset ready at: {path}")
    return path


def setup() -> str:
    """
    Sets up the environment and returns the path to the dataset root.

    Local:
        Reads KAGGLE_API_TOKEN from .env at the repo root.
        Copy .env.example to .env and fill in your token.
        Get your token at: https://www.kaggle.com/settings → API → Create New Token

    Colab:
        Clones the repo, adds it to sys.path.
        Set credentials before calling setup():
            import os
            os.environ["KAGGLE_API_TOKEN"] = "KGAT_your_token_here"
    """
    if _is_colab():
        repo_path = _clone_repo()
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        os.chdir(repo_path)
        print(f"Repo ready at: {repo_path}")
    else:
        _load_env()

    return _download_dataset()
