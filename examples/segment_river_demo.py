"""
SAM segmentation demo: extract a river centerline from river.jpg.

Produces a 4-panel diagnostic figure:
  1. Original image
  2. SAM mask
  3. Skeleton + resampled centerline
  4. Lattice overlay

Also saves the centerline as output/river_centerline.json for reuse.

Run:
    conda run -n lattice-carving python examples/segment_river_demo.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.lattice import Lattice2D
from src.roi_extraction import segment_with_sam, mask_to_centerline

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
RIVER_PATH = PROJECT_ROOT / "river.jpg"


def run_demo(
    image_path: Path = RIVER_PATH,
    n_control_points: int = 40,
    perp_extent: float = 80,
    n_lines: int = 512,
    point_prompts: np.ndarray = None,
):
    """Run the full SAM → centerline → lattice pipeline."""

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please place river.jpg in the project root.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load image for display
    pil_img = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_img)
    H, W = image_np.shape[:2]
    print(f"Image: {W}x{H}")

    # --- Step 1: SAM segmentation ---
    print("Running SAM segmentation...")
    mask = segment_with_sam(str(image_path), point_prompts=point_prompts)
    print(f"  Mask: {mask.sum()} pixels ({100 * mask.mean():.1f}% of image)")

    # --- Step 2: Extract centerline ---
    print("Extracting centerline...")
    centerline = mask_to_centerline(mask, n_control_points=n_control_points)
    print(f"  Centerline: {centerline.shape[0]} control points")

    # Save centerline JSON
    centerline_list = centerline.tolist()
    json_path = OUTPUT_DIR / "river_centerline.json"
    with open(json_path, "w") as f:
        json.dump(centerline_list, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- Step 3: Build lattice ---
    print("Building lattice...")
    lattice = Lattice2D.from_curve_points(
        centerline, n_lines=n_lines, perp_extent=perp_extent)
    lattice.smooth(max_iterations=100)
    print(f"  Lattice: {lattice.n_lines} scanlines")

    # --- Step 4: Skeleton for visualization ---
    from skimage.morphology import skeletonize
    skeleton = skeletonize(mask.astype(bool))

    # --- Step 5: Generate 4-panel figure ---
    print("Generating diagnostic figure...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: Original
    axes[0].imshow(image_np)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: SAM mask
    axes[1].imshow(image_np)
    mask_overlay = np.zeros((*mask.shape, 4))
    mask_overlay[mask, :] = [0, 0.8, 0.8, 0.4]  # cyan overlay
    axes[1].imshow(mask_overlay)
    axes[1].set_title("SAM Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Skeleton + centerline
    axes[2].imshow(image_np, alpha=0.5)
    skel_ys, skel_xs = np.where(skeleton)
    axes[2].scatter(skel_xs, skel_ys, s=0.3, c='yellow', alpha=0.5, label='Skeleton')
    cp = centerline.numpy()
    axes[2].plot(cp[:, 0], cp[:, 1], 'r-', linewidth=2, label='Centerline')
    axes[2].scatter(cp[:, 0], cp[:, 1], s=20, c='red', zorder=5)
    axes[2].legend(fontsize=9, loc='upper right')
    axes[2].set_title("Skeleton + Centerline", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Panel 4: Lattice overlay
    axes[3].imshow(image_np)
    lattice_w = int(2 * perp_extent)
    n_max = lattice.n_lines
    # Draw scanlines
    for i in range(0, n_max, max(1, n_max // 48)):
        u_vals = torch.linspace(0, float(lattice_w), 80)
        pts = torch.stack([u_vals, torch.full_like(u_vals, float(i))], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        axes[3].plot(world_pts[:, 0], world_pts[:, 1], 'cyan', alpha=0.3, linewidth=0.6)
    # Draw perpendicular lines
    for u_val in np.linspace(0, lattice_w, 12):
        n_vals = torch.linspace(0, float(n_max - 1), 100)
        pts = torch.stack([torch.full_like(n_vals, u_val), n_vals], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        axes[3].plot(world_pts[:, 0], world_pts[:, 1], 'yellow', alpha=0.25, linewidth=0.6)
    axes[3].set_title("Lattice Overlay", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    fig.suptitle("SAM River Segmentation Pipeline", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "segment_river_demo.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {fig_path}")
    print("\nDone!")


if __name__ == "__main__":
    run_demo()
