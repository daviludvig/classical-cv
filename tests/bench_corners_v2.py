"""
Offline benchmark v2: multi-parameter-per-image corner detection.

Tests an adaptive approach: for each image, try many parameter combinations
and pick the one with the best geometric/checkerboard quality score.

Run:  python tests/bench_corners_v2.py [--n 50]
"""

import sys, os, argparse, json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.setup import setup
from src.chess import (
    list_samples, load_annotation, annotation_path_for, corners_to_pixels,
    order_corners, to_gray, blur, clahe, detect_edges, detect_lines_hough,
    classify_lines, hough_line_intersection,
    _cluster_lines_full, _best_grid_window,
    compute_homography, warp_board,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gt_corners(ann: dict, img_shape: tuple) -> np.ndarray:
    return order_corners(corners_to_pixels(ann["corners"], img_shape))


def corner_error(detected: np.ndarray, gt: np.ndarray) -> dict:
    detected = order_corners(detected)
    dists = np.linalg.norm(detected - gt, axis=1)
    return {"mean": float(np.mean(dists)), "max": float(np.max(dists))}


# ---------------------------------------------------------------------------
# Quality scoring (no GT needed)
# ---------------------------------------------------------------------------

def checkerboard_contrast_score(img: np.ndarray, corners: np.ndarray, dst_size: int = 200) -> float:
    """Warp using candidate corners and measure checkerboard contrast.

    A correct board warp should show alternating light/dark squares.
    Returns a score in [0, inf) — higher is better.
    """
    try:
        H = compute_homography(corners, dst_size=dst_size, border_frac=0.0)
        warped = warp_board(img, H, size=dst_size)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    except Exception:
        return 0.0

    cs = dst_size // 8
    margin = max(1, cs // 5)
    light_sum = 0.0
    dark_sum = 0.0
    n_cells = 0

    for r in range(8):
        for c in range(8):
            y1, y2 = r * cs + margin, (r + 1) * cs - margin
            x1, x2 = c * cs + margin, (c + 1) * cs - margin
            if y2 <= y1 or x2 <= x1:
                continue
            cell_mean = float(np.mean(gray[y1:y2, x1:x2]))
            if (r + c) % 2 == 0:
                light_sum += cell_mean
            else:
                dark_sum += cell_mean
            n_cells += 1

    if n_cells < 64:
        return 0.0

    # Contrast between light and dark squares
    return abs(light_sum - dark_sum) / 32.0


def quad_regularity_score(corners: np.ndarray, img_shape: tuple) -> float:
    """Score based on quadrilateral geometry."""
    pts = order_corners(corners)
    area = cv2.contourArea(pts)
    img_area = img_shape[0] * img_shape[1]
    area_ratio = area / img_area

    if area_ratio < 0.03 or area_ratio > 0.95:
        return 0.0

    if not cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)):
        return 0.0

    # Opposite sides should be similar length
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if min(sides) < 10:
        return 0.0
    ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2])
    ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3])
    return (ratio1 + ratio2) / 2


def combined_score(img: np.ndarray, corners: np.ndarray) -> float:
    """Combined quality score: geometry + checkerboard contrast."""
    geo = quad_regularity_score(corners, img.shape[:2])
    if geo < 0.1:
        return 0.0
    contrast = checkerboard_contrast_score(img, corners)
    # Normalize contrast (empirically, good boards have contrast > 20)
    contrast_norm = min(contrast / 50.0, 1.0)
    return 0.3 * geo + 0.7 * contrast_norm


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_hough(
    gray: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
    blur_ksize: int = 5,
    angle_tol: float = np.pi / 6,
    n_grid_lines: int = 9,
    min_gap: float = 20,
) -> np.ndarray | None:
    blurred = blur(gray, ksize=blur_ksize)
    edges = detect_edges(blurred, low=canny_low, high=canny_high)
    lines = detect_lines_hough(edges, threshold=hough_threshold)
    if lines is None:
        return None

    horiz, vert = classify_lines(lines, angle_tol)
    if len(horiz) < 2 or len(vert) < 2:
        return None

    h_clusters = _cluster_lines_full(horiz, n_target=999, min_gap=min_gap, img_shape=gray.shape, is_horizontal=True)
    v_clusters = _cluster_lines_full(vert, n_target=999, min_gap=min_gap, img_shape=gray.shape, is_horizontal=False)

    h_grid = _best_grid_window(h_clusters, n_grid_lines, gray.shape, is_horizontal=True)
    v_grid = _best_grid_window(v_clusters, n_grid_lines, gray.shape, is_horizontal=False)

    if len(h_grid) < 2 or len(v_grid) < 2:
        return None

    h_top, h_bottom = h_grid[0], h_grid[-1]
    v_left, v_right = v_grid[0], v_grid[-1]

    tl = hough_line_intersection(*h_top, *v_left)
    tr = hough_line_intersection(*h_top, *v_right)
    br = hough_line_intersection(*h_bottom, *v_right)
    bl = hough_line_intersection(*h_bottom, *v_left)

    if any(p is None for p in [tl, tr, br, bl]):
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_contour(
    gray: np.ndarray,
    blur_ksize: int = 7,
    canny_low: int = 30,
    canny_high: int = 100,
    dilate_iter: int = 2,
    min_area_ratio: float = 0.05,
) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * min_area_ratio

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def detect_contour_morph(
    gray: np.ndarray,
    blur_ksize: int = 7,
    canny_low: int = 30,
    canny_high: int = 100,
    close_ksize: int = 15,
    min_area_ratio: float = 0.05,
) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * min_area_ratio

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


# ---------------------------------------------------------------------------
# Multi-parameter adaptive detection
# ---------------------------------------------------------------------------

HOUGH_SWEEP = [
    {"canny_low": 30, "canny_high": 90, "hough_threshold": 80, "blur_ksize": 5},
    {"canny_low": 40, "canny_high": 120, "hough_threshold": 90, "blur_ksize": 5},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 100, "blur_ksize": 5},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 80, "blur_ksize": 5},
    {"canny_low": 60, "canny_high": 180, "hough_threshold": 120, "blur_ksize": 5},
    {"canny_low": 70, "canny_high": 200, "hough_threshold": 100, "blur_ksize": 5},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 100, "blur_ksize": 7},
    {"canny_low": 40, "canny_high": 120, "hough_threshold": 90, "blur_ksize": 7},
    {"canny_low": 60, "canny_high": 180, "hough_threshold": 100, "blur_ksize": 3},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 120, "blur_ksize": 5},
]

CONTOUR_SWEEP = [
    {"blur_ksize": 5, "canny_low": 20, "canny_high": 60, "dilate_iter": 2},
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "dilate_iter": 2},
    {"blur_ksize": 5, "canny_low": 30, "canny_high": 100, "dilate_iter": 2},
    {"blur_ksize": 7, "canny_low": 20, "canny_high": 80, "dilate_iter": 2},
    {"blur_ksize": 9, "canny_low": 40, "canny_high": 120, "dilate_iter": 2},
    {"blur_ksize": 5, "canny_low": 20, "canny_high": 60, "dilate_iter": 3},
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "dilate_iter": 3},
]

CONTOUR_MORPH_SWEEP = [
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "close_ksize": 11},
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "close_ksize": 15},
    {"blur_ksize": 5, "canny_low": 25, "canny_high": 90, "close_ksize": 21},
    {"blur_ksize": 7, "canny_low": 20, "canny_high": 80, "close_ksize": 15},
    {"blur_ksize": 9, "canny_low": 30, "canny_high": 100, "close_ksize": 15},
]


def detect_adaptive(img: np.ndarray) -> np.ndarray | None:
    """Try multiple detection strategies and pick the best by quality score."""
    gray_raw = to_gray(img)
    gray_bilateral = cv2.bilateralFilter(gray_raw, 9, 75, 75)

    candidates = []  # (corners, score)

    # Hough on raw and bilateral
    for gray in [gray_raw, gray_bilateral]:
        for params in HOUGH_SWEEP:
            c = detect_hough(gray, **params)
            if c is not None:
                score = combined_score(img, c)
                candidates.append((c, score))

    # Contour on raw and bilateral
    for gray in [gray_raw, gray_bilateral]:
        for params in CONTOUR_SWEEP:
            c = detect_contour(gray, **params)
            if c is not None:
                score = combined_score(img, c)
                candidates.append((c, score))

    # Contour morph on raw and bilateral
    for gray in [gray_raw, gray_bilateral]:
        for params in CONTOUR_MORPH_SWEEP:
            c = detect_contour_morph(gray, **params)
            if c is not None:
                score = combined_score(img, c)
                candidates.append((c, score))

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x[1])
    if best[1] < 0.05:
        return None
    return order_corners(best[0])


def detect_adaptive_hough_only(img: np.ndarray) -> np.ndarray | None:
    """Multi-param Hough only (faster, no contour)."""
    gray_raw = to_gray(img)

    candidates = []
    for params in HOUGH_SWEEP:
        c = detect_hough(gray_raw, **params)
        if c is not None:
            score = combined_score(img, c)
            candidates.append((c, score))

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x[1])
    if best[1] < 0.05:
        return None
    return order_corners(best[0])


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def detect_baseline(img: np.ndarray) -> np.ndarray | None:
    """Current baseline: raw gray, Canny(50,150), Hough threshold=100."""
    gray = to_gray(img)
    return detect_hough(gray, canny_low=50, canny_high=150, hough_threshold=100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(images, n, detect_fn, name):
    errors = []
    found = 0
    for i in range(n):
        img = cv2.imread(str(images[i]))
        ann = load_annotation(annotation_path_for(images[i]))
        gt = gt_corners(ann, img.shape)

        corners = detect_fn(img)
        if corners is not None:
            found += 1
            err = corner_error(corners, gt)
            errors.append(err["mean"])
        else:
            errors.append(None)

    valid = [e for e in errors if e is not None]
    det_rate = found / n
    mean_err = float(np.mean(valid)) if valid else float("inf")
    med_err = float(np.median(valid)) if valid else float("inf")
    p90_err = float(np.percentile(valid, 90)) if valid else float("inf")
    max_err = float(np.max(valid)) if valid else float("inf")
    under50 = sum(1 for e in valid if e < 50) / n
    under20 = sum(1 for e in valid if e < 20) / n

    print(f"  {name:40s} detect={det_rate:5.0%} mean={mean_err:7.1f}px med={med_err:7.1f}px p90={p90_err:7.1f}px max={max_err:7.1f}px <50px={under50:5.0%} <20px={under20:5.0%}")
    return {
        "name": name, "detect_rate": det_rate, "mean": mean_err,
        "median": med_err, "p90": p90_err, "max": max_err,
        "under_50": under50, "under_20": under20,
        "per_image": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    DATA_DIR = setup()
    images = list_samples(DATA_DIR)
    n = min(args.n, len(images))
    print(f"Benchmarking on {n} images\n")

    results = {}

    print("=" * 120)
    print("COMPARISON")
    print("=" * 120)

    results["baseline"] = evaluate(images, n, detect_baseline, "Baseline (current)")
    results["adaptive_hough"] = evaluate(images, n, detect_adaptive_hough_only, "Adaptive Hough (multi-param)")
    results["adaptive_full"] = evaluate(images, n, detect_adaptive, "Adaptive Full (hough+contour)")

    # Per-image comparison: where adaptive beats baseline
    print()
    print("=" * 120)
    print("PER-IMAGE COMPARISON (adaptive_full vs baseline)")
    print("=" * 120)
    print(f"  {'Image':12s} {'Baseline':>10s} {'Adaptive':>10s} {'Delta':>10s}")
    print("  " + "-" * 50)

    base = results["baseline"]["per_image"]
    adap = results["adaptive_full"]["per_image"]
    wins, losses, ties = 0, 0, 0
    for i in range(n):
        b = base[i]
        a = adap[i]
        b_str = f"{b:.1f}px" if b is not None else "MISS"
        a_str = f"{a:.1f}px" if a is not None else "MISS"

        if b is None and a is None:
            delta_str = "both miss"
            ties += 1
        elif b is None:
            delta_str = "RECOVERED"
            wins += 1
        elif a is None:
            delta_str = "LOST"
            losses += 1
        else:
            delta = a - b
            delta_str = f"{delta:+.1f}px"
            if delta < -5:
                wins += 1
            elif delta > 5:
                losses += 1
            else:
                ties += 1

        print(f"  {images[i].name:12s} {b_str:>10s} {a_str:>10s} {delta_str:>10s}")

    print(f"\n  Adaptive wins: {wins}, losses: {losses}, ties: {ties}")

    # Save results
    out_path = Path(__file__).parent / "bench_corners_v2_results.json"
    serializable = {}
    for k, v in results.items():
        sv = {kk: vv for kk, vv in v.items() if kk != "per_image"}
        serializable[k] = sv
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
