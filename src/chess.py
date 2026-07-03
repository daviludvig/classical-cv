"""
Chessboard detection and analysis using classical computer vision.

Provides functions for:
- Board detection via edge detection, Hough lines, and contour analysis
- Perspective correction using homography
- Grid segmentation into 8x8 cells
- Occupancy detection via intensity, edge density, and texture features
- Move detection by comparing two occupancy maps
- Visualization helpers (grid overlay, occupancy map, move diff)

Dataset: Synthetic Chess Board Images (thefamousrat/synthetic-chess-board-images)
  - 1280x1280 JPEG images rendered at an angle
  - JSON annotations with piece positions and normalized corner coordinates
"""

import json
from pathlib import Path
from typing import Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

SQUARE_NAMES = [f"{c}{r}" for r in range(8, 0, -1) for c in "ABCDEFGH"]

PIECE_TYPES = ["pawn", "rook", "knight", "bishop", "queen", "king"]

# The dataset annotations provide the outer *frame* corners, not the inner
# playing-area corners.  This fraction represents the frame border width
# as a proportion of the board diagonal, measured empirically.
FRAME_BORDER_FRAC = 0.0


def _resolve_data_dir(dataset_dir: Union[str, Path]) -> Path:
    """Return the directory that actually contains the images.

    kagglehub may place files in a ``data/`` sub-folder; this helper
    transparently resolves that.
    """
    d = Path(dataset_dir)
    data_sub = d / "data"
    if data_sub.is_dir() and any(data_sub.glob("*.jpg")):
        return data_sub
    return d


def list_samples(dataset_dir: Union[str, Path]) -> list[Path]:
    """Return sorted list of image paths in the dataset directory."""
    return sorted(_resolve_data_dir(dataset_dir).glob("*.jpg"))


def load_annotation(json_path: Union[str, Path]) -> dict:
    """Load a per-image annotation JSON.

    Returns dict with keys:
        config  – dict mapping square names (e.g. "A1") to piece type strings
        corners – list of 4 [x, y] pairs in normalised [0, 1] coordinates
    """
    with open(json_path) as f:
        return json.load(f)


def annotation_path_for(image_path: Union[str, Path]) -> Path:
    """Return the .json annotation path corresponding to an image path."""
    return Path(image_path).with_suffix(".json")


def corners_to_pixels(corners_norm: list, img_shape: tuple) -> np.ndarray:
    """Convert normalised [0,1] corner coordinates to pixel coordinates.

    The dataset annotations store corners as ``[y_norm, x_norm]`` (row, col).
    This function converts them to ``[x, y]`` pixel coordinates suitable for
    OpenCV drawing functions.

    Args:
        corners_norm: list of 4 [y_norm, x_norm] pairs, values in [0, 1].
        img_shape: (height, width, ...) of the target image.

    Returns:
        (4, 2) float32 array of (x, y) pixel coordinates.
    """
    h, w = img_shape[:2]
    pts = np.array(corners_norm, dtype=np.float32)
    px_x = pts[:, 1] * w   # second element is x (col)
    px_y = pts[:, 0] * h   # first element is y (row)
    return np.column_stack([px_x, px_y]).astype(np.float32)


def frame_to_grid_corners(
    frame_corners: np.ndarray, border_frac: float = FRAME_BORDER_FRAC
) -> np.ndarray:
    """Shrink frame corners inward to approximate the inner playing-area corners.

    Each corner is moved towards the opposite corner by *border_frac* of the
    diagonal, compensating for the wooden frame border.
    """
    ordered = order_corners(frame_corners)
    inner = ordered.copy()
    for i in range(4):
        opposite = (i + 2) % 4
        inner[i] = ordered[i] + 2 * border_frac * (ordered[opposite] - ordered[i])
    return inner


def ground_truth_occupancy(config: dict) -> np.ndarray:
    """Build an 8x8 boolean occupancy matrix from annotation config.

    Row 0 = rank 8 (top), col 0 = file A (left).
    """
    occ = np.zeros((8, 8), dtype=bool)
    for square, _piece in config.items():
        row, col = square_to_grid(square)
        occ[row, col] = True
    return occ


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cell_name(row: int, col: int) -> str:
    """Convert grid indices to chess notation (e.g. row=0, col=0 -> 'A8')."""
    return f"{chr(65 + col)}{8 - row}"


def square_to_grid(notation: str) -> tuple[int, int]:
    """Convert chess notation (e.g. 'E2') to grid indices (row, col)."""
    col = ord(notation[0].upper()) - ord("A")
    row = 8 - int(notation[1])
    return row, col


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def adaptive_threshold(gray: np.ndarray, block: int = 15, C: int = 3) -> np.ndarray:
    """Apply adaptive Gaussian thresholding."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C
    )


def morph_clean(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply morphological opening then closing to clean a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def equalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization."""
    obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return obj.apply(gray)


# ---------------------------------------------------------------------------
# Edge & line detection
# ---------------------------------------------------------------------------

def detect_edges_roberts(gray: np.ndarray, thresh: int = 30) -> np.ndarray:
    """Roberts cross-gradient edge detection."""
    kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kx)
    gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return (mag > thresh).astype(np.uint8) * 255


def detect_edges_prewitt(gray: np.ndarray, thresh: int = 30) -> np.ndarray:
    """Prewitt edge detection."""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, kx)
    gy = cv2.filter2D(gray.astype(np.float64), cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return (mag > thresh).astype(np.uint8) * 255


def detect_edges_sobel(gray: np.ndarray, ksize: int = 3, thresh: int = 30) -> np.ndarray:
    """Sobel edge detection (magnitude of x and y gradients)."""
    gx = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_norm = (mag / mag.max() * 255).astype(np.uint8) if mag.max() > 0 else mag.astype(np.uint8)
    _, binary = cv2.threshold(mag_norm, thresh, 255, cv2.THRESH_BINARY)
    return binary


def detect_edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge detection."""
    return cv2.Canny(gray, low, high)


def detect_lines_hough(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 100,
) -> np.ndarray | None:
    """Standard Hough Transform — returns (N, 1, 2) array of (rho, theta)."""
    return cv2.HoughLines(edges, rho, theta, threshold)


def detect_lines_houghp(
    edges: np.ndarray,
    rho: float = 1,
    theta: float = np.pi / 180,
    threshold: int = 80,
    min_length: int = 80,
    max_gap: int = 10,
) -> np.ndarray | None:
    """Probabilistic Hough Transform — returns (N, 1, 4) line segments."""
    return cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_length, maxLineGap=max_gap)


def classify_lines(lines_hough: np.ndarray, angle_tol: float = np.pi / 6):
    """Split Standard-Hough lines into near-horizontal and near-vertical groups.

    Args:
        lines_hough: (N, 1, 2) array from cv2.HoughLines (rho, theta).
        angle_tol: tolerance in radians around 0/pi (horizontal) and pi/2 (vertical).

    Returns:
        (horizontal, vertical) — each a list of (rho, theta) tuples.
    """
    horizontal, vertical = [], []
    if lines_hough is None:
        return horizontal, vertical
    for line in lines_hough:
        rho, theta = line[0]
        if theta < angle_tol or theta > (np.pi - angle_tol):
            vertical.append((rho, theta))
        elif abs(theta - np.pi / 2) < angle_tol:
            horizontal.append((rho, theta))
    return horizontal, vertical


def cluster_line_positions(lines: list, n_target: int = 9, min_gap: int = 20) -> np.ndarray:
    """Cluster a set of Hough lines by their intercept and return representative positions.

    For horizontal lines (theta ~ pi/2), the intercept is rho (≈ y).
    For vertical lines (theta ~ 0), the intercept is rho (≈ x).

    Returns sorted array of up to *n_target* representative intercept values.
    """
    if not lines:
        return np.array([])
    intercepts = sorted([rho for rho, _theta in lines])
    clusters: list[float] = []
    current: list[float] = [intercepts[0]]
    for i in range(1, len(intercepts)):
        if abs(intercepts[i] - intercepts[i - 1]) < min_gap:
            current.append(intercepts[i])
        else:
            clusters.append(float(np.mean(current)))
            current = [intercepts[i]]
    clusters.append(float(np.mean(current)))

    if len(clusters) > n_target:
        clusters = _pick_evenly_spaced(clusters, n_target)
    return np.array(sorted(clusters))


def _pick_evenly_spaced(values: list[float], n: int) -> list[float]:
    """Select *n* values from a sorted list that are most evenly spaced."""
    if len(values) <= n:
        return values
    indices = np.round(np.linspace(0, len(values) - 1, n)).astype(int)
    return [values[i] for i in indices]


# ---------------------------------------------------------------------------
# Hough-based board corner detection
# ---------------------------------------------------------------------------

def hough_line_intersection(
    rho1: float, theta1: float, rho2: float, theta2: float
) -> np.ndarray | None:
    """Compute the intersection point of two Hough lines (rho, theta).

    Returns (x, y) as float32 array, or None if lines are parallel.
    """
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)],
    ])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-8:
        return None
    b = np.array([rho1, rho2])
    x, y = np.linalg.solve(A, b)
    return np.array([x, y], dtype=np.float32)


def _cluster_lines_full(
    lines: list[tuple[float, float]],
    n_target: int = 9,
    min_gap: float = 20,
    img_shape: tuple = (0, 0),
    is_horizontal: bool = True,
) -> list[tuple[float, float]]:
    """Cluster Hough lines by effective image position.

    Unlike :func:`cluster_line_positions` (which returns intercept scalars),
    this function returns a representative ``(rho, theta)`` pair per cluster
    so that line intersections can be computed later.

    Args:
        lines: list of (rho, theta) tuples from :func:`classify_lines`.
        n_target: desired number of output clusters.
        min_gap: minimum pixel distance between cluster centres.
        img_shape: (height, width, ...) used to compute the effective position.
        is_horizontal: True for horizontal lines (position = y), False for
            vertical lines (position = x).
    """
    if not lines:
        return []

    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2

    def _pos(rho: float, theta: float) -> float:
        ct, st = np.cos(theta), np.sin(theta)
        if is_horizontal:
            return (rho - cx * ct) / st if abs(st) > 1e-6 else rho
        else:
            return (rho - cy * st) / ct if abs(ct) > 1e-6 else rho

    positioned = sorted(
        [(_pos(rho, theta), rho, theta) for rho, theta in lines],
        key=lambda x: x[0],
    )

    # Merge lines that are closer than *min_gap*
    clusters: list[tuple[float, float]] = []
    current: list[tuple[float, float, float]] = [positioned[0]]
    for i in range(1, len(positioned)):
        if abs(positioned[i][0] - current[-1][0]) < min_gap:
            current.append(positioned[i])
        else:
            rho_med = float(np.median([c[1] for c in current]))
            theta_med = float(np.median([c[2] for c in current]))
            clusters.append((rho_med, theta_med))
            current = [positioned[i]]
    rho_med = float(np.median([c[1] for c in current]))
    theta_med = float(np.median([c[2] for c in current]))
    clusters.append((rho_med, theta_med))

    if len(clusters) > n_target:
        indices = np.round(np.linspace(0, len(clusters) - 1, n_target)).astype(int)
        clusters = [clusters[int(i)] for i in indices]

    return clusters


def _best_grid_window(
    clusters: list[tuple[float, float]],
    n_grid: int,
    img_shape: tuple,
    is_horizontal: bool,
) -> list[tuple[float, float]]:
    """Select the *n_grid* consecutive clusters that form the most regular grid.

    For a perspective chessboard, the true grid lines should have the most
    uniform spacing among all candidate windows.  The window with the lowest
    coefficient-of-variation of consecutive spacings wins.
    """
    if len(clusters) <= n_grid:
        return clusters

    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2

    def _pos(rho: float, theta: float) -> float:
        ct, st = np.cos(theta), np.sin(theta)
        if is_horizontal:
            return (rho - cx * ct) / st if abs(st) > 1e-6 else rho
        else:
            return (rho - cy * st) / ct if abs(ct) > 1e-6 else rho

    positions = [_pos(r, t) for r, t in clusters]

    best_cv = float("inf")
    best_start = 0
    for start in range(len(clusters) - n_grid + 1):
        spacings = np.diff(positions[start : start + n_grid])
        mean_sp = np.mean(spacings)
        if mean_sp > 0:
            cv = float(np.std(spacings) / mean_sp)
        else:
            cv = float("inf")
        if cv < best_cv:
            best_cv = cv
            best_start = start

    return clusters[best_start : best_start + n_grid]


def detect_board_corners(
    gray: np.ndarray,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
    angle_tol: float = np.pi / 6,
    n_grid_lines: int = 9,
    min_gap: float = 20,
) -> np.ndarray | None:
    """Detect the 4 corners of a chessboard using Hough line intersections.

    Pipeline:
        1. Gaussian blur + Canny edge detection
        2. Standard Hough Transform
        3. Classify lines into horizontal / vertical groups
        4. Cluster each group (no cap — capture all natural clusters)
        5. Select the 9 consecutive clusters with most regular spacing
        6. Intersect the outermost horizontal and vertical lines → 4 corners

    Returns:
        (4, 2) float32 array ordered TL, TR, BR, BL — or *None*.
    """
    blurred = blur(gray, ksize=blur_ksize)
    edges = detect_edges(blurred, low=canny_low, high=canny_high)
    lines = detect_lines_hough(edges, threshold=hough_threshold)
    if lines is None:
        return None

    horiz, vert = classify_lines(lines, angle_tol)
    if len(horiz) < 2 or len(vert) < 2:
        return None

    # Cluster without capping — keep all natural clusters
    h_clusters = _cluster_lines_full(
        horiz, n_target=999, min_gap=min_gap,
        img_shape=gray.shape, is_horizontal=True,
    )
    v_clusters = _cluster_lines_full(
        vert, n_target=999, min_gap=min_gap,
        img_shape=gray.shape, is_horizontal=False,
    )

    # Pick the best window of n_grid_lines consecutive clusters
    h_grid = _best_grid_window(h_clusters, n_grid_lines, gray.shape, is_horizontal=True)
    v_grid = _best_grid_window(v_clusters, n_grid_lines, gray.shape, is_horizontal=False)

    if len(h_grid) < 2 or len(v_grid) < 2:
        return None

    # Outermost lines of the selected grid
    h_top, h_bottom = h_grid[0], h_grid[-1]
    v_left, v_right = v_grid[0], v_grid[-1]

    tl = hough_line_intersection(*h_top, *v_left)
    tr = hough_line_intersection(*h_top, *v_right)
    br = hough_line_intersection(*h_bottom, *v_right)
    bl = hough_line_intersection(*h_bottom, *v_left)

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    return order_corners(corners)


def _checkerboard_score(gray: np.ndarray, corners: np.ndarray, dst_size: int = 160) -> float:
    """Score candidate corners by checkerboard contrast after perspective warp.

    Warps the image using the candidate corners and measures the mean intensity
    difference between light and dark squares.  A correct detection yields
    clearly alternating squares; a wrong detection yields near-zero contrast.
    """
    try:
        src = order_corners(corners)
        dst_pts = np.array(
            [[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(src, dst_pts)
        warped = cv2.warpPerspective(gray, H, (dst_size, dst_size))
    except Exception:
        return 0.0

    cs = dst_size // 8
    margin = max(1, cs // 5)
    light, dark = 0.0, 0.0
    for r in range(8):
        for c in range(8):
            cell = warped[r * cs + margin:(r + 1) * cs - margin,
                          c * cs + margin:(c + 1) * cs - margin]
            if (r + c) % 2 == 0:
                light += float(np.mean(cell))
            else:
                dark += float(np.mean(cell))
    return abs(light - dark) / 32.0


def detect_board_corners_robust(
    img_bgr: np.ndarray,
    **hough_kw,
) -> np.ndarray | None:
    """Detect board corners trying multiple preprocessings, pick best by quality.

    Tries four preprocessings in order (raw, CLAHE, bilateral, CLAHE+bilateral).
    Each is passed to :func:`detect_board_corners`; the candidate with the
    highest checkerboard contrast score is returned.

    Using CLAHE helps suppress strong background patterns (floor tiles, wood
    grain) that compete with the board lines in the Hough transform.

    Args:
        img_bgr: original BGR image (not pre-converted to gray).
        **hough_kw: forwarded to :func:`detect_board_corners`.

    Returns:
        (4, 2) float32 array ordered TL, TR, BR, BL — or *None*.
    """
    gray_raw = to_gray(img_bgr)
    gray_clahe = clahe(gray_raw)
    gray_bilateral = cv2.bilateralFilter(gray_raw, 9, 75, 75)
    gray_clahe_bilateral = cv2.bilateralFilter(gray_clahe, 9, 75, 75)

    best_corners, best_score = None, -1.0
    for gray in (gray_raw, gray_clahe, gray_bilateral, gray_clahe_bilateral):
        corners = detect_board_corners(gray, **hough_kw)
        if corners is None:
            continue
        score = _checkerboard_score(gray_raw, corners)
        if score > best_score:
            best_score, best_corners = score, corners

    return best_corners


def detect_board_corners_combined(
    gray: np.ndarray,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    angle_tol: float = np.pi / 6,
    n_grid_lines: int = 9,
    min_gap: float = 20,
) -> np.ndarray | None:
    """Detect board corners using Hough line intersections.

    Pipeline:
        1. Gaussian blur + Canny (full image)
        2. Standard Hough → cluster into h_clusters / v_clusters
        3. _best_grid_window selects the 9 most regularly-spaced clusters
        4. Intersect outermost H and V lines → 4 corners

    Returns:
        (4, 2) float32 array ordered TL, TR, BR, BL — or *None*.
    """
    blurred = blur(gray, ksize=blur_ksize)
    edges = detect_edges(blurred, low=canny_low, high=canny_high)

    lines = detect_lines_hough(edges, threshold=hough_threshold)
    if lines is None:
        return None

    horiz, vert = classify_lines(lines, angle_tol)
    if len(horiz) < 2 or len(vert) < 2:
        return None

    h_clusters = _cluster_lines_full(
        horiz, n_target=999, min_gap=min_gap,
        img_shape=gray.shape, is_horizontal=True,
    )
    v_clusters = _cluster_lines_full(
        vert, n_target=999, min_gap=min_gap,
        img_shape=gray.shape, is_horizontal=False,
    )

    h_grid = _best_grid_window(h_clusters, n_grid_lines, gray.shape, is_horizontal=True)
    v_grid = _best_grid_window(v_clusters, n_grid_lines, gray.shape, is_horizontal=False)

    if len(h_grid) < 3 or len(v_grid) < 3:
        return None

    # h_grid[-1] is the board's wooden frame edge (clearly visible as the last
    # Hough line at the bottom).  The GT annotation marks the playing-area
    # boundary, which corresponds to h_grid[-2] — one step above the frame.
    tl = hough_line_intersection(*h_grid[0],  *v_grid[0])
    tr = hough_line_intersection(*h_grid[0],  *v_grid[-1])
    br = hough_line_intersection(*h_grid[-2], *v_grid[-1])
    bl = hough_line_intersection(*h_grid[-2], *v_grid[0])

    if any(p is None for p in [tl, tr, br, bl]):
        return None
    return order_corners(np.array([tl, tr, br, bl], dtype=np.float32))


# ---------------------------------------------------------------------------
# Board contour detection
# ---------------------------------------------------------------------------

def find_board_contour(
    gray: np.ndarray,
    blur_ksize: int = 7,
    canny_low: int = 30,
    canny_high: int = 100,
    min_area_ratio: float = 0.05,
) -> np.ndarray | None:
    """Detect the chessboard quadrilateral via contour analysis.

    Returns 4 corner points as (4, 2) float32 array, or None if not found.
    """
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * min_area_ratio

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < min_area:
            break
        # Try increasing epsilon on the raw contour first
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        # Fall back: convex hull tends to give a cleaner quadrilateral
        hull = cv2.convexHull(cnt)
        hull_peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
            approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    return None


def detect_grid_in_warped(
    warped_gray: np.ndarray, dst_size: int = 480
) -> tuple[int, int, int]:
    """Detect the 8x8 grid origin and cell size in a warped board image.

    Maximises **checkerboard contrast** — the absolute difference between
    the mean intensity of cells on light squares vs. dark squares.  An
    integral-image is used so that each candidate *(x0, y0, cs)* is
    evaluated in O(64) additions.

    Cell size is constrained to keep the grid within 85–100 % of
    *dst_size*.

    Returns:
        (x0, y0, cell_size) — top-left offset and square side in pixels.
    """
    gray = warped_gray.astype(np.float64)
    integral = cv2.integral(gray)  # shape (h+1, w+1)

    def _rect_sum(y1: int, x1: int, y2: int, x2: int) -> float:
        return float(
            integral[y2, x2] - integral[y1, x2]
            - integral[y2, x1] + integral[y1, x1]
        )

    min_cs = max(1, int(dst_size * 0.85) // 8)
    max_cs = dst_size // 8

    best_score = -1.0
    best_params = (0, 0, max_cs)

    for cs in range(min_cs, max_cs + 1):
        span = 8 * cs
        if span > dst_size:
            continue
        max_origin = dst_size - span
        margin = max(1, cs // 5)
        cell_area = (cs - 2 * margin) ** 2
        if cell_area <= 0:
            continue

        for x0 in range(max_origin + 1):
            for y0 in range(max_origin + 1):
                light = 0.0
                dark = 0.0
                for r in range(8):
                    ry1 = y0 + r * cs + margin
                    ry2 = y0 + (r + 1) * cs - margin
                    for c in range(8):
                        cx1 = x0 + c * cs + margin
                        cx2 = x0 + (c + 1) * cs - margin
                        s = _rect_sum(ry1, cx1, ry2, cx2)
                        if (r + c) % 2 == 0:
                            light += s
                        else:
                            dark += s

                contrast = abs(light - dark) / cell_area
                if contrast > best_score:
                    best_score = contrast
                    best_params = (x0, y0, cs)

    return best_params


def grid_corners_from_warped(
    frame_corners: np.ndarray, x0: int, y0: int, cs: int, dst_size: int = 480
) -> np.ndarray:
    """Map detected grid corners back to the original image coordinate system.

    Given the frame-level *frame_corners* (used to produce the warped image)
    and the detected grid offset *(x0, y0, cs)* inside that warp, return the
    four playing-area corners in original-image pixel coordinates, ordered as
    TL, TR, BR, BL.
    """
    src = order_corners(frame_corners)
    dst_pts = np.array(
        [[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]],
        dtype=np.float32,
    )
    H_fwd, _ = cv2.findHomography(src, dst_pts)
    H_inv = np.linalg.inv(H_fwd)

    # Grid corners in warped-image coordinates
    grid_w = np.array(
        [[x0, y0], [x0 + 8 * cs, y0],
         [x0 + 8 * cs, y0 + 8 * cs], [x0, y0 + 8 * cs]],
        dtype=np.float32,
    )

    # Project back to original image
    grid_orig = np.zeros((4, 2), dtype=np.float32)
    for i, pt in enumerate(grid_w):
        p = H_inv @ np.array([pt[0], pt[1], 1.0])
        grid_orig[i] = p[:2] / p[2]
    return grid_orig


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------

def order_corners(corners: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum and difference of coordinates to determine ordering.
    """
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(d)]   # top-right: smallest x-y
    ordered[3] = pts[np.argmax(d)]   # bottom-left: largest x-y
    return ordered


def compute_homography(
    src_corners: np.ndarray,
    dst_size: int = 480,
    border_frac: float = FRAME_BORDER_FRAC,
) -> np.ndarray:
    """Compute a homography that maps *src_corners* to a square of *dst_size*.

    When *border_frac* > 0 the destination rectangle is expanded outward so
    that the wooden frame border falls outside the output image and only the
    inner playing area is captured.

    Args:
        src_corners: 4 corner points (will be ordered automatically).
        dst_size: side length of the output square in pixels.
        border_frac: fraction of *dst_size* occupied by the frame border on
            each side.  Set to 0 to disable the correction.

    Returns:
        3x3 homography matrix.
    """
    src = order_corners(src_corners)
    b = int(dst_size * border_frac)
    dst = np.array(
        [[-b, -b], [dst_size + b, -b],
         [dst_size + b, dst_size + b], [-b, dst_size + b]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst)
    return H


def warp_board(img: np.ndarray, H: np.ndarray, size: int = 480) -> np.ndarray:
    """Apply homography to produce a top-down view of the board."""
    return cv2.warpPerspective(img, H, (size, size))


# ---------------------------------------------------------------------------
# Grid segmentation
# ---------------------------------------------------------------------------

def detect_board_rotation(warped: np.ndarray, grid_size: int = 8) -> int:
    """Detect how many 90° rotations align the warped board with standard orientation.

    Standard chess convention (row 0 = rank 8, col 0 = file A):

        (0,0)=A8=light   (0,7)=H8=dark
        (7,0)=A1=dark    (7,7)=H1=light

    A light square satisfies ``(row + col) % 2 == 0``.

    The function samples cells across the board (not just corners, for
    robustness against pieces) and picks the rotation *k* such that
    ``np.rot90(occupancy, k)`` best matches the standard colour pattern.

    Returns:
        *k* (0–3) to pass directly to ``np.rot90(array, k)``.
    """
    n = grid_size
    cs_h, cs_w = warped.shape[0] // n, warped.shape[1] // n
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    margin = max(1, min(cs_h, cs_w) // 5)

    def _cell_mean(row: int, col: int) -> float:
        y1 = row * cs_h + margin
        y2 = (row + 1) * cs_h - margin
        x1 = col * cs_w + margin
        x2 = (col + 1) * cs_w - margin
        return float(np.mean(gray[y1:y2, x1:x2]))

    # Sample ALL 64 cells — pieces affect individual cells but the
    # majority of the board still follows the checkerboard pattern.
    sample_positions = [(r, c) for r in range(n) for c in range(n)]
    intensities = {(r, c): _cell_mean(r, c) for r, c in sample_positions}

    # np.rot90(data, k) source mappings — result[r,c] = data[src_r, src_c]:
    #   k=0: (r, c)
    #   k=1: (c, n-1-r)
    #   k=2: (n-1-r, n-1-c)
    #   k=3: (n-1-c, r)
    _src = [
        lambda r, c: (r, c),
        lambda r, c: (c, n - 1 - r),
        lambda r, c: (n - 1 - r, n - 1 - c),
        lambda r, c: (n - 1 - c, r),
    ]

    best_k, best_score = 0, -np.inf
    for k in range(4):
        score = 0.0
        for (std_r, std_c) in sample_positions:
            src_r, src_c = _src[k](std_r, std_c)
            if (src_r, src_c) not in intensities:
                continue
            val = intensities[(src_r, src_c)]
            expected_light = (std_r + std_c) % 2 == 0
            score += val if expected_light else -val
        if score > best_score:
            best_score, best_k = score, k

    return best_k


def best_rotation_vs_gt(
    occupancy: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, int]:
    """Try all 4 rotations and return the one that best matches *gt*.

    Useful for evaluation — eliminates the 180° ambiguity that colour-based
    detection cannot resolve.

    Returns:
        (aligned_occupancy, k) — the rotated map and the rotation used.
    """
    best_k, best_acc = 0, -1.0
    for k in range(4):
        acc = float(np.mean(np.rot90(occupancy, k) == gt))
        if acc > best_acc:
            best_acc, best_k = acc, k
    return np.rot90(occupancy, best_k), best_k


def segment_grid(warped: np.ndarray, grid_size: int = 8) -> list[list[np.ndarray]]:
    """Divide a square warped board image into *grid_size* x *grid_size* cells.

    Returns cells[row][col] — row 0 is rank 8 (top), col 0 is file A (left).
    """
    h, w = warped.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    cells: list[list[np.ndarray]] = []
    for row in range(grid_size):
        row_cells: list[np.ndarray] = []
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            row_cells.append(warped[y1:y2, x1:x2])
        cells.append(row_cells)
    return cells


# ---------------------------------------------------------------------------
# Occupancy detection
# ---------------------------------------------------------------------------

def cell_features(cell_bgr: np.ndarray, margin: float = 0.15) -> dict:
    """Extract classical features from a single cell image.

    Features returned:
        mean_intensity  – average grayscale
        std_intensity   – standard deviation of grayscale
        edge_density    – fraction of Canny edge pixels
        texture_var     – variance of Laplacian (focus/texture measure)
        center_diff     – absolute difference between centre and border mean
    """
    h, w = cell_bgr.shape[:2]
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    mh, mw = int(h * margin), int(w * margin)
    roi = gray[mh : h - mh, mw : w - mw]

    mean_int = float(np.mean(roi))
    std_int = float(np.std(roi))

    edges = cv2.Canny(roi, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    lap = cv2.Laplacian(roi, cv2.CV_64F)
    texture_var = float(lap.var())

    ch, cw = roi.shape
    center = roi[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
    center_mean = float(np.mean(center))
    border_total = float(np.sum(roi)) - float(np.sum(center))
    border_size = roi.size - center.size
    border_mean = border_total / border_size if border_size > 0 else mean_int
    center_diff = abs(center_mean - border_mean)

    return {
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "edge_density": edge_density,
        "texture_var": texture_var,
        "center_diff": center_diff,
    }


def is_occupied(
    feats: dict,
    edge_thresh: float = 0.03,
    texture_thresh: float = 50.0,
    std_thresh: float = 12.0,
    center_thresh: float = 12.0,
    min_votes: int = 3,
) -> bool:
    """Simple vote-based occupancy classifier.

    A cell is deemed occupied when at least *min_votes* features exceed their
    respective thresholds.
    """
    votes = (
        int(feats["edge_density"] > edge_thresh)
        + int(feats["texture_var"] > texture_thresh)
        + int(feats["std_intensity"] > std_thresh)
        + int(feats["center_diff"] > center_thresh)
    )
    return votes >= min_votes


def build_occupancy_map(cells: list[list[np.ndarray]], **occ_kw) -> tuple[np.ndarray, list[list[dict]]]:
    """Classify every cell in the grid and return the occupancy matrix.

    Extra keyword arguments are forwarded to :func:`is_occupied`.

    Returns:
        occupancy – (8, 8) bool array (True = occupied).
        features  – 8x8 nested list of feature dicts.
    """
    n = len(cells)
    occupancy = np.zeros((n, n), dtype=bool)
    features: list[list[dict]] = []
    for row in range(n):
        row_feats: list[dict] = []
        for col in range(n):
            f = cell_features(cells[row][col])
            occupancy[row, col] = is_occupied(f, **occ_kw)
            row_feats.append(f)
        features.append(row_feats)
    return occupancy, features


def classify_piece_color(cell_bgr: np.ndarray, margin: float = 0.25) -> str:
    """Classify a piece as light ('w') or dark ('b') using HSV value channel.

    Assumes the cell is already known to be occupied.  Dark (ebony) pieces
    have lower HSV-Value than light (boxwood) ones.  Threshold 85 was tuned
    on 30 GT-corrected images (white mean≈121, black mean≈54).

    Returns ``'w'`` or ``'b'``.
    """
    h, w = cell_bgr.shape[:2]
    m = int(min(h, w) * margin)
    roi = cell_bgr[m : h - m, m : w - m]
    ch, cw = roi.shape[:2]
    center = roi[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
    center_v = float(np.mean(cv2.cvtColor(center, cv2.COLOR_BGR2HSV)[:, :, 2]))
    return "b" if center_v < 85 else "w"


# ---------------------------------------------------------------------------
# Move detection
# ---------------------------------------------------------------------------

def detect_moves(
    occ_before: np.ndarray, occ_after: np.ndarray
) -> list[tuple[int, int, str]]:
    """Compare two 8x8 occupancy maps and return a list of changes.

    Each entry is (row, col, change_type) where change_type is
    ``'vacated'`` or ``'occupied'``.
    """
    changes: list[tuple[int, int, str]] = []
    for r in range(8):
        for c in range(8):
            if occ_before[r, c] and not occ_after[r, c]:
                changes.append((r, c, "vacated"))
            elif not occ_before[r, c] and occ_after[r, c]:
                changes.append((r, c, "occupied"))
    return changes


# ---------------------------------------------------------------------------
# FEN notation and move inference
# ---------------------------------------------------------------------------

_FEN_TYPE = {"pawn": "p", "knight": "n", "bishop": "b",
             "rook": "r", "queen": "q", "king": "k"}
_PT_NAME = {"pawn": "Peão", "knight": "Cavalo", "bishop": "Bispo",
            "rook": "Torre", "queen": "Dama", "king": "Rei"}


def piece_to_fen_symbol(label: str) -> str:
    """Map a piece label like ``'knight_w'`` to its FEN symbol (``'N'``).

    White pieces are uppercase, black pieces lowercase.
    """
    type_part, _, color = label.partition("_")
    sym = _FEN_TYPE[type_part]
    return sym.upper() if color == "w" else sym


def board_to_fen(piece_map: dict, side: str = "w") -> str:
    """Convert a ``{square: label}`` map (e.g. ``{"E1": "king_w"}``) into FEN.

    Only the piece-placement field can be inferred from a single image; the
    remaining FEN fields (side to move, castling rights, en passant target,
    half/full-move clocks) are filled with neutral defaults.
    """
    ranks = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for file in "ABCDEFGH":
            label = piece_map.get(f"{file}{rank}")
            if label is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += piece_to_fen_symbol(label)
        if empty:
            row += str(empty)
        ranks.append(row)
    return f"{'/'.join(ranks)} {side} - - 0 1"


def _uci(square: str) -> str:
    """``'E2'`` -> ``'e2'`` (UCI uses lowercase files)."""
    return square[0].lower() + square[1]


def _describe_move(before: dict, after: dict, frm: str, to: str, capture: bool) -> dict:
    piece = before[frm]
    ptype, color = piece.split("_")
    dest_type = after[to].split("_")[0]
    promo = _FEN_TYPE[dest_type] if (ptype == "pawn" and dest_type != "pawn") else ""
    verb = "captura em" if capture else "para"
    desc = f"{_PT_NAME[ptype]} {'branco' if color == 'w' else 'preto'} {verb} {_uci(to)}"
    if promo:
        desc += f", promove a {_PT_NAME[dest_type]}"
    mtype = "promotion" if promo else ("capture" if capture else "move")
    return {"type": mtype, "from": frm, "to": to,
            "uci": _uci(frm) + _uci(to) + promo, "description": desc}


def classify_move(before: dict, after: dict) -> dict:
    """Infer a single chess move from two piece maps (``{square: label}``).

    Handles quiet moves, captures, castling, en passant and promotion by
    comparing which squares were emptied, filled or changed identity.
    Returns a dict with ``type``, ``from``, ``to``, ``uci`` and a
    human-readable ``description``. If the difference does not correspond to
    one legal move (e.g. two unrelated positions), ``type`` is ``'ambiguous'``.
    """
    squares = set(before) | set(after)
    vacated = sorted(s for s in squares if s in before and s not in after)
    appeared = sorted(s for s in squares if s in after and s not in before)
    changed = sorted(s for s in squares
                     if s in before and s in after and before[s] != after[s])

    # Castling — king and rook both move (2 emptied, 2 filled)
    if len(vacated) == 2 and len(appeared) == 2 and not changed:
        king_from = next((s for s in vacated if before[s].startswith("king")), None)
        king_to = (next((s for s in appeared
                         if abs(ord(s[0]) - ord(king_from[0])) == 2), None)
                   if king_from else None)
        if king_from and king_to:
            side = "O-O" if king_to[0] > king_from[0] else "O-O-O"
            return {"type": "castle", "from": king_from, "to": king_to,
                    "uci": _uci(king_from) + _uci(king_to), "description": f"Roque {side}"}

    # En passant — pawn moves diagonally to an empty square, capturing a pawn
    if len(vacated) == 2 and len(appeared) == 1 and not changed:
        dest = appeared[0]
        mover = next((s for s in vacated
                      if before[s].startswith("pawn") and s[0] != dest[0]), None)
        captured = next((s for s in vacated if s != mover), None)
        if mover and after[dest].startswith("pawn"):
            return {"type": "en_passant", "from": mover, "to": dest,
                    "uci": _uci(mover) + _uci(dest),
                    "description": f"En passant {_uci(mover)}x{_uci(dest)} "
                                   f"(captura peão em {_uci(captured)})"}

    # Quiet move — one square emptied, one filled
    if len(vacated) == 1 and len(appeared) == 1 and not changed:
        return _describe_move(before, after, vacated[0], appeared[0], capture=False)

    # Capture — origin emptied, destination changes identity
    if len(vacated) == 1 and not appeared and len(changed) == 1:
        return _describe_move(before, after, vacated[0], changed[0], capture=True)

    return {"type": "ambiguous", "from": None, "to": None, "uci": None,
            "description": (f"{len(vacated)} esvaziada(s), {len(appeared)} preenchida(s), "
                            f"{len(changed)} alterada(s) — não corresponde a um lance único")}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_corners(
    img: np.ndarray,
    corners: np.ndarray,
    radius: int = 12,
    thickness: int = 3,
) -> np.ndarray:
    """Draw the four corners as a labelled quadrilateral.

    Draws the outline connecting the corners plus coloured circles and
    TL / TR / BR / BL labels so the result is visible even at small
    display sizes.
    """
    vis = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    pts = corners.astype(int)

    # Quadrilateral outline
    for i in range(4):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), thickness)

    # Corner circles + labels
    for i, pt in enumerate(pts):
        cv2.circle(vis, tuple(pt), radius, colors[i], -1)
        cv2.putText(
            vis, labels[i], (pt[0] + radius + 4, pt[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[i], 2, cv2.LINE_AA,
        )
    return vis


def draw_grid_overlay(
    img: np.ndarray,
    corners: np.ndarray,
    grid_size: int = 8,
    color: tuple = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw the projected grid lines on the original (perspective) image.

    Uses a homography to correctly project equidistant grid lines from
    board-space into the perspective image, so that lines converge towards
    the vanishing points instead of being linearly interpolated.
    """
    src = order_corners(corners)
    # Homography: grid-space (0..grid_size) → image-space
    grid_dst = np.array(
        [[0, 0], [grid_size, 0], [grid_size, grid_size], [0, grid_size]],
        dtype=np.float32,
    )
    H_to_img, _ = cv2.findHomography(grid_dst, src)

    def _project(gx: float, gy: float):
        p = H_to_img @ np.array([gx, gy, 1.0])
        return int(round(p[0] / p[2])), int(round(p[1] / p[2]))

    vis = img.copy()
    for i in range(grid_size + 1):
        # Horizontal rank lines: y = i, x goes 0 → grid_size
        pts = [_project(x * 0.25, i) for x in range(grid_size * 4 + 1)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, color, thickness)
        # Vertical file lines: x = i, y goes 0 → grid_size
        pts = [_project(i, y * 0.25) for y in range(grid_size * 4 + 1)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(vis, a, b, color, thickness)
    return vis


def draw_occupancy_image(
    occupancy: np.ndarray,
    cell_size: int = 60,
    occupied_color: tuple = (0, 0, 200),
    empty_color: tuple = (0, 200, 0),
) -> np.ndarray:
    """Render an 8x8 occupancy map as a colour-coded image."""
    n = occupancy.shape[0]
    size = n * cell_size
    vis = np.zeros((size, size, 3), dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            color = occupied_color if occupancy[row, col] else empty_color
            cv2.rectangle(vis, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, -1)
            label = cell_name(row, col)
            cv2.putText(
                vis, label, (x1 + 12, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return vis


def draw_move_diff(
    occ_before: np.ndarray,
    occ_after: np.ndarray,
    cell_size: int = 60,
) -> np.ndarray:
    """Visualise changes between two occupancy maps.

    Yellow = vacated, Cyan = newly occupied, Grey = unchanged.
    """
    changes = detect_moves(occ_before, occ_after)
    change_map = {(r, c): t for r, c, t in changes}
    n = occ_before.shape[0]
    size = n * cell_size
    vis = np.zeros((size, size, 3), dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            y1, y2 = row * cell_size, (row + 1) * cell_size
            x1, x2 = col * cell_size, (col + 1) * cell_size
            key = (row, col)
            if key in change_map:
                color = (0, 255, 255) if change_map[key] == "vacated" else (255, 255, 0)
            else:
                color = (80, 80, 80)
            cv2.rectangle(vis, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, -1)
            label = cell_name(row, col)
            cv2.putText(
                vis, label, (x1 + 12, y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return vis


def draw_hough_lines(
    img: np.ndarray,
    lines_hough: np.ndarray | None,
    color: tuple = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw Standard-Hough lines on an image copy."""
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines_hough is None:
        return vis
    for line in lines_hough:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        length = max(vis.shape[:2])
        pt1 = (int(x0 + length * (-b)), int(y0 + length * a))
        pt2 = (int(x0 - length * (-b)), int(y0 - length * a))
        cv2.line(vis, pt1, pt2, color, thickness)
    return vis
