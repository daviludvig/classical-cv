"""
Offline benchmark: compare corner-detection strategies against GT corners.

Run:  python tests/bench_corners.py [--n 50]
"""

import sys, os, argparse, time, json
from pathlib import Path
from itertools import product

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.setup import setup
from src.chess import (
    list_samples, load_annotation, annotation_path_for, corners_to_pixels,
    order_corners, to_gray, blur, clahe, detect_edges, detect_lines_hough,
    classify_lines, find_board_contour, hough_line_intersection,
    _cluster_lines_full, _best_grid_window,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gt_corners(ann: dict, img_shape: tuple) -> np.ndarray:
    return order_corners(corners_to_pixels(ann["corners"], img_shape))


def corner_error(detected: np.ndarray, gt: np.ndarray) -> dict:
    """Per-corner and mean pixel error."""
    detected = order_corners(detected)
    dists = np.linalg.norm(detected - gt, axis=1)
    return {
        "mean": float(np.mean(dists)),
        "max": float(np.max(dists)),
        "per_corner": dists.tolist(),
    }


# ---------------------------------------------------------------------------
# Strategy: Hough-based (current baseline)
# ---------------------------------------------------------------------------

def detect_hough(
    gray: np.ndarray,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
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


# ---------------------------------------------------------------------------
# Strategy: Contour-based
# ---------------------------------------------------------------------------

def detect_contour(
    gray: np.ndarray,
    blur_ksize: int = 7,
    canny_low: int = 30,
    canny_high: int = 100,
    min_area_ratio: float = 0.05,
    dilate_iter: int = 2,
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


# ---------------------------------------------------------------------------
# Strategy: Contour with morphological close (helps close board outline gaps)
# ---------------------------------------------------------------------------

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
# Preprocessing variants
# ---------------------------------------------------------------------------

def preprocess_raw(img: np.ndarray) -> np.ndarray:
    return to_gray(img)


def preprocess_clahe(img: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    gray = to_gray(img)
    return clahe(gray, clip=clip, grid=grid)


def preprocess_bilateral(img: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75) -> np.ndarray:
    gray = to_gray(img)
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def preprocess_clahe_bilateral(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    cl = clahe(gray, clip=2.0, grid=8)
    return cv2.bilateralFilter(cl, 9, 75, 75)


def preprocess_bilateral_clahe(img: np.ndarray) -> np.ndarray:
    gray = to_gray(img)
    bf = cv2.bilateralFilter(gray, 9, 75, 75)
    return clahe(bf, clip=2.0, grid=8)


# ---------------------------------------------------------------------------
# Corner refinement
# ---------------------------------------------------------------------------

def refine_corners_subpix(gray: np.ndarray, corners: np.ndarray, win: int = 11) -> np.ndarray:
    """Refine corners to sub-pixel accuracy."""
    h, w = gray.shape[:2]
    pts = corners.reshape(-1, 2).astype(np.float32).copy()
    # Clamp to image bounds (with margin for the search window)
    margin = win + 1
    pts[:, 0] = np.clip(pts[:, 0], margin, w - margin - 1)
    pts[:, 1] = np.clip(pts[:, 1], margin, h - margin - 1)
    # Check all corners are inside image
    if np.any(pts[:, 0] < margin) or np.any(pts[:, 1] < margin):
        return corners  # can't refine
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    pts = pts.reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    return refined.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Geometric quality score (no GT needed)
# ---------------------------------------------------------------------------

def quad_quality_score(corners: np.ndarray, img_shape: tuple) -> float:
    """Score [0,1] based on how 'board-like' the quadrilateral is.

    Checks:
    - Area relative to image (boards typically occupy 20-80% of image)
    - Convexity
    - Aspect ratio of bounding rect (should be roughly square)
    """
    pts = order_corners(corners)
    area = cv2.contourArea(pts)
    img_area = img_shape[0] * img_shape[1]
    area_ratio = area / img_area

    # Area penalty
    if area_ratio < 0.05 or area_ratio > 0.95:
        return 0.0
    area_score = 1.0 - abs(area_ratio - 0.35) / 0.35  # peak at ~35%
    area_score = max(0.0, min(1.0, area_score))

    # Convexity
    if not cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)):
        return 0.0

    # Side-length regularity: opposite sides should be similar
    sides = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
    ratio1 = min(sides[0], sides[2]) / max(sides[0], sides[2]) if max(sides[0], sides[2]) > 0 else 0
    ratio2 = min(sides[1], sides[3]) / max(sides[1], sides[3]) if max(sides[1], sides[3]) > 0 else 0
    regularity = (ratio1 + ratio2) / 2

    return 0.4 * area_score + 0.6 * regularity


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

PREPROCESS = {
    "raw": preprocess_raw,
    "clahe": preprocess_clahe,
    "bilateral": preprocess_bilateral,
    "clahe+bilateral": preprocess_clahe_bilateral,
    "bilateral+clahe": preprocess_bilateral_clahe,
}

HOUGH_PARAMS = [
    {"canny_low": 30, "canny_high": 90, "hough_threshold": 80},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 100},  # current default
    {"canny_low": 40, "canny_high": 120, "hough_threshold": 90},
    {"canny_low": 60, "canny_high": 180, "hough_threshold": 120},
    {"canny_low": 30, "canny_high": 100, "hough_threshold": 70},
    {"canny_low": 50, "canny_high": 150, "hough_threshold": 80},
    {"canny_low": 70, "canny_high": 200, "hough_threshold": 100},
]

CONTOUR_PARAMS = [
    {"blur_ksize": 5, "canny_low": 20, "canny_high": 60},
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100},  # current default
    {"blur_ksize": 7, "canny_low": 20, "canny_high": 80},
    {"blur_ksize": 9, "canny_low": 40, "canny_high": 120},
    {"blur_ksize": 5, "canny_low": 30, "canny_high": 100},
    {"blur_ksize": 11, "canny_low": 25, "canny_high": 90},
]

CONTOUR_MORPH_PARAMS = [
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "close_ksize": 11},
    {"blur_ksize": 7, "canny_low": 30, "canny_high": 100, "close_ksize": 15},
    {"blur_ksize": 7, "canny_low": 20, "canny_high": 80, "close_ksize": 15},
    {"blur_ksize": 5, "canny_low": 25, "canny_high": 90, "close_ksize": 21},
    {"blur_ksize": 9, "canny_low": 30, "canny_high": 100, "close_ksize": 15},
]


def run_experiment(images, n_images, strategy_name, preproc_name, preproc_fn, detect_fn, detect_params, refine=False):
    """Run one experiment configuration and return summary."""
    errors = []
    detected_count = 0

    for i in range(n_images):
        img = cv2.imread(str(images[i]))
        ann = load_annotation(annotation_path_for(images[i]))
        gt = gt_corners(ann, img.shape)
        gray = preproc_fn(img)

        corners = detect_fn(gray, **detect_params)
        if corners is not None:
            if refine:
                raw_gray = to_gray(img)
                corners = refine_corners_subpix(raw_gray, corners)
            detected_count += 1
            err = corner_error(corners, gt)
            errors.append(err["mean"])
        else:
            errors.append(None)

    found = [e for e in errors if e is not None]
    missed = sum(1 for e in errors if e is None)

    return {
        "strategy": strategy_name,
        "preproc": preproc_name,
        "params": {k: v for k, v in detect_params.items()},
        "refine": refine,
        "n_images": n_images,
        "detected": detected_count,
        "missed": missed,
        "detect_rate": detected_count / n_images,
        "mean_err": float(np.mean(found)) if found else float("inf"),
        "median_err": float(np.median(found)) if found else float("inf"),
        "p90_err": float(np.percentile(found, 90)) if found else float("inf"),
        "max_err": float(np.max(found)) if found else float("inf"),
        "under_50px": sum(1 for e in found if e < 50) / n_images if found else 0,
        "under_20px": sum(1 for e in found if e < 20) / n_images if found else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of images to evaluate")
    args = parser.parse_args()

    DATA_DIR = setup()
    images = list_samples(DATA_DIR)
    n = min(args.n, len(images))
    print(f"Benchmarking on {n} images\n")

    all_results = []
    total_experiments = 0

    # -----------------------------------------------------------------------
    # Phase 1: Hough-based with different preprocessing and params
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: Hough-based detection")
    print("=" * 70)

    for preproc_name, preproc_fn in PREPROCESS.items():
        for hp in HOUGH_PARAMS:
            for refine in [False, True]:
                label = f"hough | {preproc_name} | canny({hp['canny_low']},{hp['canny_high']}) hough_t={hp['hough_threshold']}"
                if refine:
                    label += " +subpix"
                total_experiments += 1

                r = run_experiment(images, n, "hough", preproc_name, preproc_fn, detect_hough, hp, refine=refine)
                all_results.append(r)

                status = f"detect={r['detect_rate']:.0%} mean={r['mean_err']:.1f}px med={r['median_err']:.1f}px <50px={r['under_50px']:.0%}"
                print(f"  [{total_experiments:3d}] {label:75s} {status}")

    # -----------------------------------------------------------------------
    # Phase 2: Contour-based with different preprocessing and params
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PHASE 2: Contour-based detection")
    print("=" * 70)

    for preproc_name, preproc_fn in PREPROCESS.items():
        for cp in CONTOUR_PARAMS:
            for refine in [False, True]:
                label = f"contour | {preproc_name} | blur={cp['blur_ksize']} canny({cp['canny_low']},{cp['canny_high']})"
                if refine:
                    label += " +subpix"
                total_experiments += 1

                r = run_experiment(images, n, "contour", preproc_name, preproc_fn, detect_contour, cp, refine=refine)
                all_results.append(r)

                status = f"detect={r['detect_rate']:.0%} mean={r['mean_err']:.1f}px med={r['median_err']:.1f}px <50px={r['under_50px']:.0%}"
                print(f"  [{total_experiments:3d}] {label:75s} {status}")

    # -----------------------------------------------------------------------
    # Phase 3: Contour + morphological close
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PHASE 3: Contour + morphological close")
    print("=" * 70)

    for preproc_name, preproc_fn in PREPROCESS.items():
        for mp in CONTOUR_MORPH_PARAMS:
            for refine in [False, True]:
                label = f"contour_morph | {preproc_name} | blur={mp['blur_ksize']} canny({mp['canny_low']},{mp['canny_high']}) close={mp['close_ksize']}"
                if refine:
                    label += " +subpix"
                total_experiments += 1

                r = run_experiment(images, n, "contour_morph", preproc_name, preproc_fn, detect_contour_morph, mp, refine=refine)
                all_results.append(r)

                status = f"detect={r['detect_rate']:.0%} mean={r['mean_err']:.1f}px med={r['median_err']:.1f}px <50px={r['under_50px']:.0%}"
                print(f"  [{total_experiments:3d}] {label:75s} {status}")

    # -----------------------------------------------------------------------
    # Summary: top 20 by composite score
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"SUMMARY — Top 20 of {total_experiments} experiments (sorted by composite score)")
    print("=" * 70)
    print()
    print("Score = 0.4 * detect_rate + 0.3 * under_50px_rate + 0.3 * (1 - clamp(median_err/200))")
    print()

    for r in all_results:
        med_norm = min(r["median_err"], 200.0) / 200.0
        r["score"] = 0.4 * r["detect_rate"] + 0.3 * r["under_50px"] + 0.3 * (1.0 - med_norm)

    top = sorted(all_results, key=lambda r: r["score"], reverse=True)[:20]

    print(f"{'#':>3} {'Score':>6} {'Det%':>5} {'Mean':>7} {'Med':>7} {'P90':>7} {'<50px':>6} {'<20px':>6} {'Strategy'}")
    print("-" * 100)
    for i, r in enumerate(top):
        ref = "+spx" if r["refine"] else "    "
        p = r["params"]
        params_str = " ".join(f"{k}={v}" for k, v in p.items())
        print(f"{i+1:3d} {r['score']:6.3f} {r['detect_rate']:5.0%} {r['mean_err']:7.1f} {r['median_err']:7.1f} {r['p90_err']:7.1f} {r['under_50px']:6.0%} {r['under_20px']:6.0%} {r['strategy']}|{r['preproc']}|{params_str} {ref}")

    # Save full results
    out_path = Path(__file__).parent / "bench_corners_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    # -----------------------------------------------------------------------
    # Phase 4: Hybrid — best contour + best hough fallback
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PHASE 4: Hybrid (best strategies combined)")
    print("=" * 70)

    # Find best contour and best hough configs
    best_contour = sorted([r for r in all_results if "contour" in r["strategy"]], key=lambda r: r["score"], reverse=True)[0]
    best_hough = sorted([r for r in all_results if r["strategy"] == "hough"], key=lambda r: r["score"], reverse=True)[0]

    print(f"  Best contour: {best_contour['strategy']}|{best_contour['preproc']}|{best_contour['params']} (score={best_contour['score']:.3f})")
    print(f"  Best hough:   {best_hough['strategy']}|{best_hough['preproc']}|{best_hough['params']} (score={best_hough['score']:.3f})")
    print()

    # Hybrid: try contour first, fall back to hough, pick best by quality score
    hybrid_errors = []
    hybrid_found = 0
    hybrid_method_used = {"contour": 0, "hough": 0, "best_of_both": 0, "none": 0}

    bc_preproc = PREPROCESS[best_contour["preproc"]]
    bh_preproc = PREPROCESS[best_hough["preproc"]]
    bc_detect = detect_contour_morph if best_contour["strategy"] == "contour_morph" else detect_contour
    bc_refine = best_contour["refine"]
    bh_refine = best_hough["refine"]

    for i in range(n):
        img = cv2.imread(str(images[i]))
        ann = load_annotation(annotation_path_for(images[i]))
        gt_c = gt_corners(ann, img.shape)

        gray_c = bc_preproc(img)
        gray_h = bh_preproc(img)
        raw_gray = to_gray(img)

        c_corners = bc_detect(gray_c, **best_contour["params"])
        h_corners = detect_hough(gray_h, **best_hough["params"])

        if bc_refine and c_corners is not None:
            c_corners = refine_corners_subpix(raw_gray, c_corners)
        if bh_refine and h_corners is not None:
            h_corners = refine_corners_subpix(raw_gray, h_corners)

        # Pick best by quality score, or by GT error if both exist
        candidates = []
        if c_corners is not None:
            candidates.append(("contour", c_corners, quad_quality_score(c_corners, img.shape[:2])))
        if h_corners is not None:
            candidates.append(("hough", h_corners, quad_quality_score(h_corners, img.shape[:2])))

        if not candidates:
            hybrid_errors.append(None)
            hybrid_method_used["none"] += 1
            continue

        # If both exist, pick the one with lower GT error (oracle) for upper bound
        if len(candidates) == 2:
            err_c = corner_error(candidates[0][1], gt_c)["mean"]
            err_h = corner_error(candidates[1][1], gt_c)["mean"]
            best = candidates[0] if err_c <= err_h else candidates[1]
            hybrid_method_used["best_of_both"] += 1
        else:
            best = candidates[0]
            hybrid_method_used[best[0]] += 1

        hybrid_found += 1
        hybrid_errors.append(corner_error(best[1], gt_c)["mean"])

    found_errs = [e for e in hybrid_errors if e is not None]
    print(f"  Hybrid (oracle): detect={hybrid_found/n:.0%} mean={np.mean(found_errs):.1f}px med={np.median(found_errs):.1f}px <50px={sum(1 for e in found_errs if e < 50)/n:.0%}")
    print(f"  Method breakdown: {hybrid_method_used}")

    # Hybrid with quality score (no GT)
    hybrid_q_errors = []
    hybrid_q_found = 0

    for i in range(n):
        img = cv2.imread(str(images[i]))
        ann = load_annotation(annotation_path_for(images[i]))
        gt_c = gt_corners(ann, img.shape)

        gray_c = bc_preproc(img)
        gray_h = bh_preproc(img)
        raw_gray = to_gray(img)

        c_corners = bc_detect(gray_c, **best_contour["params"])
        h_corners = detect_hough(gray_h, **best_hough["params"])

        if bc_refine and c_corners is not None:
            c_corners = refine_corners_subpix(raw_gray, c_corners)
        if bh_refine and h_corners is not None:
            h_corners = refine_corners_subpix(raw_gray, h_corners)

        candidates = []
        if c_corners is not None:
            candidates.append((c_corners, quad_quality_score(c_corners, img.shape[:2])))
        if h_corners is not None:
            candidates.append((h_corners, quad_quality_score(h_corners, img.shape[:2])))

        if not candidates:
            hybrid_q_errors.append(None)
            continue

        best = max(candidates, key=lambda x: x[1])
        hybrid_q_found += 1
        hybrid_q_errors.append(corner_error(best[0], gt_c)["mean"])

    found_q = [e for e in hybrid_q_errors if e is not None]
    print(f"  Hybrid (quality): detect={hybrid_q_found/n:.0%} mean={np.mean(found_q):.1f}px med={np.median(found_q):.1f}px <50px={sum(1 for e in found_q if e < 50)/n:.0%}")


if __name__ == "__main__":
    main()
