"""
Debug script: inspect _best_grid_window for image 0.jpg.

Run from the repo root:
    python tests/debug_grid_window.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import cv2

from src.chess import (
    blur,
    detect_edges,
    detect_lines_hough,
    classify_lines,
    _cluster_lines_full,
    _best_grid_window,
    corners_to_pixels,
    order_corners,
)

DATA_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/thefamousrat/synthetic-chess-board-images/versions/1/data"
)
IMG_PATH = os.path.join(DATA_DIR, "0.jpg")
JSON_PATH = os.path.join(DATA_DIR, "0.json")

# ---------------------------------------------------------------------------
# Pipeline params (same defaults as detect_board_corners)
# ---------------------------------------------------------------------------
BLUR_KSIZE    = 5
CANNY_LOW     = 50
CANNY_HIGH    = 150
HOUGH_THRESH  = 100
ANGLE_TOL     = np.pi / 6
N_GRID        = 9
MIN_GAP       = 20.0


def _pos_h(rho, theta, cx):
    ct, st = np.cos(theta), np.sin(theta)
    return (rho - cx * ct) / st if abs(st) > 1e-6 else rho


def _pos_v(rho, theta, cy):
    ct, st = np.cos(theta), np.sin(theta)
    return (rho - cy * st) / ct if abs(ct) > 1e-6 else rho


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"ERROR: could not load {IMG_PATH}")
        sys.exit(1)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    print(f"Image size: {w} x {h}")

    # ---- ground truth -------------------------------------------------------
    with open(JSON_PATH) as f:
        ann = json.load(f)

    gt_corners_norm = ann["corners"]
    gt_px = corners_to_pixels(gt_corners_norm, img.shape)
    gt_ordered = order_corners(gt_px)
    print("\n--- Ground-truth corners (TL, TR, BR, BL) ---")
    labels = ["TL", "TR", "BR", "BL"]
    for lbl, pt in zip(labels, gt_ordered):
        print(f"  {lbl}: ({pt[0]:.1f}, {pt[1]:.1f})")

    # ---- Hough pipeline -----------------------------------------------------
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = blur(gray, ksize=BLUR_KSIZE)
    edges   = detect_edges(blurred, low=CANNY_LOW, high=CANNY_HIGH)
    lines   = detect_lines_hough(edges, threshold=HOUGH_THRESH)

    if lines is None:
        print("ERROR: HoughLines returned None")
        sys.exit(1)

    horiz, vert = classify_lines(lines, ANGLE_TOL)
    print(f"\nRaw Hough lines: {len(lines)} total, {len(horiz)} horiz, {len(vert)} vert")

    # ---- Cluster (uncapped) -------------------------------------------------
    h_clusters = _cluster_lines_full(
        horiz, n_target=999, min_gap=MIN_GAP,
        img_shape=gray.shape, is_horizontal=True,
    )
    v_clusters = _cluster_lines_full(
        vert, n_target=999, min_gap=MIN_GAP,
        img_shape=gray.shape, is_horizontal=False,
    )

    # ---- Print ALL horizontal cluster positions ----------------------------
    h_positions = [_pos_h(r, t, cx) for r, t in h_clusters]
    print(f"\n--- ALL h_clusters ({len(h_clusters)} total), sorted by position ---")
    for i, (pos, (rho, theta)) in enumerate(zip(h_positions, h_clusters)):
        print(f"  [{i:2d}] pos={pos:8.2f}  rho={rho:8.2f}  theta={np.degrees(theta):.2f}°")

    # ---- Print ALL vertical cluster positions ------------------------------
    v_positions = [_pos_v(r, t, cy) for r, t in v_clusters]
    print(f"\n--- ALL v_clusters ({len(v_clusters)} total), sorted by position ---")
    for i, (pos, (rho, theta)) in enumerate(zip(v_positions, v_clusters)):
        print(f"  [{i:2d}] pos={pos:8.2f}  rho={rho:8.2f}  theta={np.degrees(theta):.2f}°")

    # ---- _best_grid_window selection ----------------------------------------
    h_grid = _best_grid_window(h_clusters, N_GRID, gray.shape, is_horizontal=True)
    v_grid = _best_grid_window(v_clusters, N_GRID, gray.shape, is_horizontal=False)

    h_grid_pos = [_pos_h(r, t, cx) for r, t in h_grid]
    v_grid_pos = [_pos_v(r, t, cy) for r, t in v_grid]

    # Find which window was selected
    h_start_idx = None
    for start in range(len(h_clusters) - N_GRID + 1):
        if h_clusters[start : start + N_GRID] == h_grid:
            h_start_idx = start
            break

    v_start_idx = None
    for start in range(len(v_clusters) - N_GRID + 1):
        if v_clusters[start : start + N_GRID] == v_grid:
            v_start_idx = start
            break

    print(f"\n--- Selected h_grid window (start_idx={h_start_idx}) ---")
    h_spacings = np.diff(h_grid_pos)
    for i, (pos, (rho, theta)) in enumerate(zip(h_grid_pos, h_grid)):
        sp = f"  gap_to_next={h_spacings[i]:.2f}" if i < len(h_spacings) else ""
        print(f"  [{i}] pos={pos:8.2f}  rho={rho:8.2f}  theta={np.degrees(theta):.2f}°{sp}")
    print(f"  Spacings: {[f'{s:.2f}' for s in h_spacings]}")
    print(f"  Mean spacing: {np.mean(h_spacings):.2f}  Std: {np.std(h_spacings):.2f}  CV: {np.std(h_spacings)/np.mean(h_spacings):.4f}")
    monotone_h = all(h_spacings[i] <= h_spacings[i+1] for i in range(len(h_spacings)-1))
    print(f"  Spacings monotonically increasing: {monotone_h}")

    # Check each possible window CV for horizontal
    print("\n--- CV for ALL h windows of size 9 ---")
    for start in range(len(h_clusters) - N_GRID + 1):
        pos_win = [_pos_h(r, t, cx) for r, t in h_clusters[start:start+N_GRID]]
        sp = np.diff(pos_win)
        mean_sp = np.mean(sp)
        cv = float(np.std(sp) / mean_sp) if mean_sp > 0 else float("inf")
        marker = " <-- SELECTED" if start == h_start_idx else ""
        print(f"  start={start}: pos=[{pos_win[0]:.1f} .. {pos_win[-1]:.1f}]  CV={cv:.4f}{marker}")

    print(f"\n--- Selected v_grid window (start_idx={v_start_idx}) ---")
    v_spacings = np.diff(v_grid_pos)
    for i, (pos, (rho, theta)) in enumerate(zip(v_grid_pos, v_grid)):
        sp = f"  gap_to_next={v_spacings[i]:.2f}" if i < len(v_spacings) else ""
        print(f"  [{i}] pos={pos:8.2f}  rho={rho:8.2f}  theta={np.degrees(theta):.2f}°{sp}")
    print(f"  Spacings: {[f'{s:.2f}' for s in v_spacings]}")
    print(f"  Mean spacing: {np.mean(v_spacings):.2f}  Std: {np.std(v_spacings):.2f}  CV: {np.std(v_spacings)/np.mean(v_spacings):.4f}")

    # Check each possible window CV for vertical
    print("\n--- CV for ALL v windows of size 9 ---")
    for start in range(len(v_clusters) - N_GRID + 1):
        pos_win = [_pos_v(r, t, cy) for r, t in v_clusters[start:start+N_GRID]]
        sp = np.diff(pos_win)
        mean_sp = np.mean(sp)
        cv = float(np.std(sp) / mean_sp) if mean_sp > 0 else float("inf")
        marker = " <-- SELECTED" if start == v_start_idx else ""
        print(f"  start={start}: pos=[{pos_win[0]:.1f} .. {pos_win[-1]:.1f}]  CV={cv:.4f}{marker}")

    # ---- Detected corners ---------------------------------------------------
    from src.chess import hough_line_intersection, order_corners as oc

    h_top_line, h_bot_line = h_grid[0], h_grid[-1]
    v_left_line, v_right_line = v_grid[0], v_grid[-1]

    tl = hough_line_intersection(*h_top_line, *v_left_line)
    tr = hough_line_intersection(*h_top_line, *v_right_line)
    br = hough_line_intersection(*h_bot_line, *v_right_line)
    bl = hough_line_intersection(*h_bot_line, *v_left_line)

    det_corners = oc(np.array([tl, tr, br, bl], dtype=np.float32))
    print("\n--- Detected corners (TL, TR, BR, BL) ---")
    for lbl, det, gt in zip(labels, det_corners, gt_ordered):
        err = np.linalg.norm(det - gt)
        print(f"  {lbl}: det=({det[0]:.1f}, {det[1]:.1f})  gt=({gt[0]:.1f}, {gt[1]:.1f})  err={err:.1f}px")

    # ---- Bottom line diagnostic ---------------------------------------------
    h_bot_pos  = h_grid_pos[-1]
    h_pre_pos  = h_grid_pos[-2]
    gap_last   = h_bot_pos - h_pre_pos
    gap_others = h_spacings[:-1]
    print(f"\n--- Bottom-line diagnostic ---")
    print(f"  Last spacing (pre-bottom -> bottom): {gap_last:.2f}")
    print(f"  Mean of other spacings:              {np.mean(gap_others):.2f}")
    ratio = gap_last / np.mean(gap_others) if np.mean(gap_others) > 0 else float("inf")
    print(f"  Ratio (last / mean_others):          {ratio:.3f}")
    if ratio > 1.5:
        print("  SUSPICIOUS: last gap is >1.5x the mean — likely a spurious floor/background line.")
    elif ratio < 0.5:
        print("  SUSPICIOUS: last gap is <0.5x the mean — likely a missed boundary line.")
    else:
        print("  OK: last gap is within normal range.")

    # Also print all candidates beyond the selected window
    if h_start_idx is not None and h_start_idx + N_GRID < len(h_clusters):
        next_cluster_pos = _pos_h(*h_clusters[h_start_idx + N_GRID], cx)
        print(f"  Next cluster beyond window: pos={next_cluster_pos:.2f}")
        print(f"  Distance from grid[-1]={h_bot_pos:.2f} to next={next_cluster_pos:.2f}: {next_cluster_pos - h_bot_pos:.2f}")


if __name__ == "__main__":
    main()
