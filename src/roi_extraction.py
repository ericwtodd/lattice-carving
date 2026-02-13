"""
ROI extraction via SAM segmentation and mask-to-centerline conversion.

Provides two main workflows:
  1. SAM-based segmentation: segment_with_sam() → binary mask
  2. Mask-to-centerline: mask_to_centerline() → ordered control points

Combined convenience: segment_river() → control points ready for
Lattice2D.from_curve_points().

Requires:
  pip install segment-anything scikit-image
  Download checkpoint: models/sam_vit_b_01ec64.pth
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List


# Default SAM checkpoint path (relative to project root)
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "sam_vit_b_01ec64.pth"


def segment_with_sam(
    image_path: str,
    point_prompts: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    model_type: str = "vit_b",
    darkness_threshold: float = 0.45,
    min_area_fraction: float = 0.02,
) -> np.ndarray:
    """Segment an object (e.g. river) from an image using SAM.

    Args:
        image_path: Path to the input image.
        point_prompts: (N, 2) array of (x, y) click points for SamPredictor.
            If None, uses SamAutomaticMaskGenerator with heuristic filtering.
        point_labels: (N,) array of labels (1=foreground, 0=background).
            Required if point_prompts is given.
        model_path: Path to SAM checkpoint. Defaults to models/sam_vit_b_01ec64.pth.
        model_type: SAM model type ('vit_b', 'vit_l', 'vit_h').
        darkness_threshold: For auto mode — max mean brightness (0-1) to consider
            a mask as "dark" (like water). Higher = more permissive.
        min_area_fraction: For auto mode — minimum mask area as fraction of image.

    Returns:
        Binary mask (H, W) as numpy bool array.
    """
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    from PIL import Image

    if model_path is None:
        model_path = str(_DEFAULT_MODEL_PATH)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device)

    # Load image
    pil_img = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_img)  # (H, W, 3) uint8

    if point_prompts is not None:
        # Point-prompted segmentation
        predictor = SamPredictor(sam)
        predictor.set_image(image_np)

        if point_labels is None:
            point_labels = np.ones(len(point_prompts), dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_prompts,
            point_labels=point_labels,
            multimask_output=True,
        )
        # Pick highest-scoring mask
        best_idx = np.argmax(scores)
        return masks[best_idx]  # (H, W) bool

    else:
        # Automatic segmentation with heuristic filtering
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )
        masks = mask_generator.generate(image_np)

        if not masks:
            raise RuntimeError("SAM generated no masks")

        H, W = image_np.shape[:2]
        total_pixels = H * W
        image_gray = image_np.astype(np.float32).mean(axis=2) / 255.0

        # Score each mask by darkness + size + elongation
        best_mask = None
        best_score = -1

        for m in masks:
            seg = m["segmentation"]  # (H, W) bool
            area = seg.sum()
            area_frac = area / total_pixels

            if area_frac < min_area_fraction:
                continue

            # Darkness: mean brightness of masked pixels (lower = darker = more river-like)
            mean_brightness = image_gray[seg].mean()
            if mean_brightness > darkness_threshold:
                continue

            # Elongation: ratio of bounding box width to height (prefer elongated)
            ys, xs = np.where(seg)
            bbox_w = xs.max() - xs.min() + 1
            bbox_h = ys.max() - ys.min() + 1
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

            # Score: larger + darker + more elongated
            score = area_frac * (1.0 - mean_brightness) * np.sqrt(aspect)
            if score > best_score:
                best_score = score
                best_mask = seg

        if best_mask is None:
            # Fallback: just use the largest mask
            best_mask = max(masks, key=lambda m: m["area"])["segmentation"]

        return best_mask


def mask_to_centerline(
    mask: np.ndarray,
    n_control_points: int = 40,
) -> torch.Tensor:
    """Extract an ordered centerline from a binary mask.

    Skeletonizes the mask, orders the skeleton pixels along the curve
    (walking from one endpoint to the nearest unvisited neighbor), and
    resamples to n_control_points evenly spaced along arc length.

    Args:
        mask: Binary mask (H, W), bool or 0/1.
        n_control_points: Number of output control points.

    Returns:
        Tensor of shape (n_control_points, 2) as (x, y) coordinates.
    """
    from skimage.morphology import skeletonize

    # Ensure bool
    mask_bool = mask.astype(bool)

    # Skeletonize
    skeleton = skeletonize(mask_bool)
    ys, xs = np.where(skeleton)

    if len(xs) < 2:
        raise RuntimeError(f"Skeleton has only {len(xs)} pixels — mask may be too small")

    # Order skeleton points by walking from an endpoint
    points = np.stack([xs, ys], axis=1).astype(np.float64)  # (N, 2)
    ordered = _order_skeleton_points(points)

    # Arc-length resample to n_control_points
    resampled = _arc_length_resample(ordered, n_control_points)

    return torch.from_numpy(resampled).float()


def _order_skeleton_points(points: np.ndarray) -> np.ndarray:
    """Order skeleton pixel coordinates by walking nearest-neighbor from an endpoint.

    Finds an endpoint (pixel with fewest skeleton neighbors), then greedily
    walks to the nearest unvisited point.

    Args:
        points: (N, 2) array of (x, y) skeleton pixel coordinates.

    Returns:
        (N, 2) ordered array.
    """
    from scipy.spatial import KDTree

    N = len(points)
    tree = KDTree(points)

    # Find an endpoint: point with fewest neighbors within sqrt(2)+0.1 distance
    neighbor_counts = tree.query_ball_point(points, r=1.5, return_length=True)
    start_idx = int(np.argmin(neighbor_counts))

    # Greedy walk
    visited = np.zeros(N, dtype=bool)
    order = np.zeros(N, dtype=int)
    order[0] = start_idx
    visited[start_idx] = True

    for i in range(1, N):
        current = order[i - 1]
        # Query increasingly large neighborhoods until we find an unvisited point
        dists, idxs = tree.query(points[current], k=min(20, N))
        if isinstance(dists, float):
            dists = np.array([dists])
            idxs = np.array([idxs])
        found = False
        for d, idx in zip(dists, idxs):
            if not visited[idx]:
                order[i] = idx
                visited[idx] = True
                found = True
                break
        if not found:
            # Find nearest unvisited globally
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                order = order[:i]
                break
            dists_all = np.linalg.norm(points[unvisited] - points[current], axis=1)
            nearest = unvisited[np.argmin(dists_all)]
            order[i] = nearest
            visited[nearest] = True

    return points[order]


def _arc_length_resample(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a polyline to n_points evenly spaced along arc length.

    Args:
        points: (N, 2) ordered polyline coordinates.
        n_points: Number of output points.

    Returns:
        (n_points, 2) resampled coordinates.
    """
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < 1e-6:
        # Degenerate: all points are the same
        return np.tile(points[0], (n_points, 1))

    # Target arc lengths
    target = np.linspace(0, total_length, n_points)

    # Interpolate
    resampled = np.zeros((n_points, 2))
    resampled[:, 0] = np.interp(target, cum_length, points[:, 0])
    resampled[:, 1] = np.interp(target, cum_length, points[:, 1])

    return resampled


def segment_river(
    image_path: str,
    n_control_points: int = 40,
    **kwargs,
) -> torch.Tensor:
    """Convenience: segment river with SAM, then extract centerline.

    Args:
        image_path: Path to the river image.
        n_control_points: Number of centerline control points.
        **kwargs: Passed to segment_with_sam().

    Returns:
        Tensor of shape (n_control_points, 2) as (x, y) coordinates,
        ready for Lattice2D.from_curve_points().
    """
    mask = segment_with_sam(image_path, **kwargs)
    centerline = mask_to_centerline(mask, n_control_points=n_control_points)
    return centerline
