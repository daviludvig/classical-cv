"""Reproducible metric verification for the chessboard pipeline.

Run:  .venv/Scripts/python.exe scripts/eval_metrics.py [N]

Reports, over N random dataset images (seed 0):
  - occupancy F1 with auto-detected and with GT corners
  - piece type+color accuracy under GT occupancy
  - orientation accuracy without GT (and the 180-degree flip rate)
  - fully-automatic end-to-end accuracy

Use this instead of quoting numbers from memory.
"""
import sys as _sys
_N = int(_sys.argv[1]) if len(_sys.argv) > 1 else 60
import os, glob, random, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
import numpy as np
import cv2
from src.chess import (
    list_samples, load_annotation, annotation_path_for, corners_to_pixels,
    order_corners, compute_homography, warp_board, segment_grid,
    ground_truth_occupancy, square_to_grid, build_occupancy_map,
    best_rotation_vs_gt, detect_board_rotation, to_gray,
    detect_board_corners_combined,
)
from src.piece_classifier import load_classifier, predict_board

random.seed(0)
DATA = os.path.expanduser("~/.cache/kagglehub/datasets/thefamousrat/"
                          "synthetic-chess-board-images/versions/1/data")
imgs = sorted(glob.glob(DATA + "/*.jpg"))
sample = random.sample(imgs, _N)
DST = 480
model, img_size, classes = load_classifier("models/piece_classifier_resnet34.pth")


def f1_of(occ, gt):
    tp = int(np.sum(occ & gt)); fp = int(np.sum(occ & ~gt)); fn = int(np.sum(~occ & gt))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = float(np.mean(occ == gt))
    return acc, prec, rec, f1


occ_auto, occ_gt = [], []          # (acc,prec,rec,f1)
type_hits = type_tot = 0
type_f1s = []
rot_match = rot_180 = rot_other = 0
e2e_hits = e2e_tot = 0

for p in sample:
    ann = load_annotation(annotation_path_for(p))
    img = cv2.imread(p)
    gt = ground_truth_occupancy(ann["config"])
    cfg = ann["config"]
    gt_color = {square_to_grid(sq): ("w" if v.endswith("_w") else "b")
                for sq, v in cfg.items()}

    # ---- GT corners branch -------------------------------------------------
    gtc = order_corners(corners_to_pixels(ann["corners"], img.shape))
    Hg = compute_homography(gtc, dst_size=DST)
    wg = warp_board(img, Hg, size=DST)
    cells_g = segment_grid(wg)
    occg, _ = build_occupancy_map(cells_g)
    occg_al, kg = best_rotation_vs_gt(occg, gt)          # GT-optimal rotation
    occ_gt.append(f1_of(occg_al, gt))

    # orientation without GT vs GT-optimal
    k_nogt = detect_board_rotation(wg)
    if k_nogt == kg:
        rot_match += 1
    elif (k_nogt - kg) % 4 == 2:
        rot_180 += 1
    else:
        rot_other += 1

    # ---- type+color accuracy (GT corners + GT occupancy + GT rotation) ------
    wg_al = np.rot90(wg, kg); cells_al = segment_grid(wg_al)
    occ_al = np.rot90(occg, kg)
    # use GT occupancy to isolate the classifier
    pmap = predict_board(model, cells_al, gt, img_size=img_size, classes=classes,
                         confidence_threshold=0.1)
    hits = sum(1 for sq, x in pmap.items() if cfg.get(sq) == x)
    type_hits += hits; type_tot += len(cfg)
    if len(pmap):
        type_f1s.append(hits / len(pmap))

    # ---- end-to-end with no-GT orientation ---------------------------------
    w_e = np.rot90(wg, k_nogt); cells_e = segment_grid(w_e)
    occ_e, _ = build_occupancy_map(cells_e)
    pmap_e = predict_board(model, cells_e, occ_e, img_size=img_size,
                           classes=classes, confidence_threshold=0.3)
    e2e_hits += sum(1 for sq, x in pmap_e.items() if cfg.get(sq) == x)
    e2e_tot += len(cfg)

    # ---- auto corners occupancy -------------------------------------------
    ac = detect_board_corners_combined(to_gray(img))
    if ac is None:
        ac = gtc
    Ha = compute_homography(ac, dst_size=DST)
    wa = warp_board(img, Ha, size=DST)
    occa, _ = build_occupancy_map(segment_grid(wa))
    occa_al, _ = best_rotation_vs_gt(occa, gt)
    occ_auto.append(f1_of(occa_al, gt))


def summ(name, rows):
    a = np.array(rows)
    print(f"{name}: acc={a[:,0].mean():.1%}  prec={a[:,1].mean():.1%}  "
          f"rec={a[:,2].mean():.1%}  f1={a[:,3].mean():.1%}")


def full_stats(name, rows):
    a = np.array(rows)
    lab = ["Acuracia", "Precisao", "Recall", "F1"]
    print(f"  [{name}] mean / median / min / max")
    for i, l in enumerate(lab):
        col = a[:, i]
        print(f"    {l:9s} {col.mean():.1%} / {np.median(col):.1%} / "
              f"{col.min():.1%} / {col.max():.1%}")


n = len(sample)
print(f"=== N = {n} images ===\n")
print("-- Occupancy --")
summ("  GT corners  ", occ_gt)
summ("  auto corners", occ_auto)
print()
full_stats("auto corners", occ_auto)
full_stats("GT corners", occ_gt)
print(f"\n-- Piece type+color (GT occupancy) --")
print(f"  exact matches: {type_hits}/{type_tot} = {type_hits/type_tot:.1%}")
print(f"  per-image mean: {np.mean(type_f1s):.1%}")
print(f"\n-- Orientation without GT (detect_board_rotation vs GT-optimal) --")
print(f"  correct:      {rot_match}/{n} = {rot_match/n:.0%}")
print(f"  180-flipped:  {rot_180}/{n} = {rot_180/n:.0%}")
print(f"  other error:  {rot_other}/{n} = {rot_other/n:.0%}")
print(f"\n-- End-to-end (auto occupancy + no-GT orientation) --")
print(f"  exact matches: {e2e_hits}/{e2e_tot} = {e2e_hits/e2e_tot:.1%}")
