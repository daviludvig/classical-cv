"""
Compare classical corner detection methods against ground-truth annotations.

Evaluates the three main methods from chess.py and reports per-image errors.
Useful as a baseline reference before training a deep learning model.

Run:  python tests/compare_corners.py [--n 100] [--out results.json]
"""

import sys, os, argparse, json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.setup import setup
from src.chess import (
    list_samples, load_annotation, annotation_path_for, corners_to_pixels,
    order_corners, to_gray,
    detect_board_corners, detect_board_corners_combined, find_board_contour,
)


def _gt_corners(ann: dict, img_shape: tuple) -> np.ndarray:
    return order_corners(corners_to_pixels(ann["corners"], img_shape))


def _corner_error(pred: np.ndarray, gt: np.ndarray) -> float:
    dists = np.linalg.norm(order_corners(pred) - gt, axis=1)
    return float(np.mean(dists))


METHODS: dict[str, callable] = {
    "hough":          lambda img: detect_board_corners(to_gray(img)),
    "hough_combined": lambda img: detect_board_corners_combined(to_gray(img)),
    "contour":        lambda img: find_board_contour(to_gray(img)),
}


def run(images: list, n: int) -> list[dict]:
    results = []
    for i, img_path in enumerate(images[:n]):
        img = cv2.imread(str(img_path))
        ann = load_annotation(annotation_path_for(img_path))
        gt = _gt_corners(ann, img.shape)

        row: dict = {"image": img_path.name, "gt_corners": gt.tolist()}
        for name, fn in METHODS.items():
            pred = fn(img)
            if pred is not None:
                row[name] = {"corners": order_corners(pred).tolist(), "error_px": _corner_error(pred, gt)}
            else:
                row[name] = None

        results.append(row)
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n}", flush=True)

    return results


def print_summary(results: list, n: int) -> None:
    print(f"\n{'Method':22s} {'Detect%':>8} {'Mean':>8} {'Median':>8} {'P90':>8} {'<20px':>7} {'<50px':>7}")
    print("-" * 72)
    for name in METHODS:
        preds = [r[name] for r in results if r[name] is not None]
        errs = sorted(p["error_px"] for p in preds)
        det = len(preds) / n
        if errs:
            print(
                f"  {name:20s} {det:8.0%}"
                f" {np.mean(errs):8.1f} {np.median(errs):8.1f}"
                f" {np.percentile(errs, 90):8.1f}"
                f" {sum(e < 20 for e in errs) / n:7.0%}"
                f" {sum(e < 50 for e in errs) / n:7.0%}"
            )
        else:
            print(f"  {name:20s} {det:8.0%}  (no detections)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Number of images to evaluate (default: 100)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: compare_corners_results.json next to this script)")
    args = parser.parse_args()

    DATA_DIR = setup()
    images = list_samples(DATA_DIR)
    n = min(args.n, len(images))
    print(f"Evaluating {n} images ({len(METHODS)} methods)\n")

    results = run(images, n)
    print_summary(results, n)

    out_path = Path(args.out) if args.out else Path(__file__).parent / "compare_corners_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPer-image results saved to {out_path}")
    print("Each entry has 'gt_corners' (ground truth) and per-method 'corners' + 'error_px'.")
    print("Use 'gt_corners' as DL training labels and 'error_px' to filter hard/easy images.")


if __name__ == "__main__":
    main()
