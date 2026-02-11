"""
Reproduce paper figures and validate carving visually.

Generates before/after images for:
  1. Traditional vs lattice-guided carving on an arch (Figure 3 analog)
  2. Seam pairs on synthetic bagel (Figure 10/22 analog — shrink the hole)
  3. Seam pairs on real double-bagel (the demo: shrink/grow left bagel)

Run:
    conda run -n lattice-carving python examples/reproduce_figures.py

Output goes to output/ directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image

from src.lattice import Lattice2D
from src.carving import (
    carve_image_traditional,
    carve_image_lattice_guided,
    carve_seam_pairs,
    _precompute_forward_mapping,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import greedy_seam, greedy_seam_windowed

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def tensor_to_numpy(t):
    """Convert (C,H,W) or (H,W) tensor to numpy for display."""
    if t.dim() == 3:
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return t.clamp(0, 1).cpu().numpy()


def save_comparison(images, titles, filename, suptitle=None):
    """Save a row of images side-by-side."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Figure 1: Traditional carving on a simple gradient + edge image
# Validates that traditional carving works and seams avoid edges
# -----------------------------------------------------------------------

def figure_traditional_carving():
    """Traditional seam carving: remove seams from a gradient image with
    a vertical edge. Seams should come from uniform regions, not cross edges."""
    print("\n--- Figure: Traditional Carving ---")
    torch.manual_seed(42)
    H, W = 100, 150

    # Create image: gradient background with a bright vertical stripe
    image = torch.zeros(3, H, W)
    grad = torch.linspace(0.1, 0.4, W).unsqueeze(0).expand(H, W)
    image[0] = grad
    image[1] = grad * 0.8
    image[2] = grad * 0.6
    # Add a vertical bright stripe at column 75
    image[:, :, 70:80] = 0.9

    original_np = tensor_to_numpy(image)
    carved = carve_image_traditional(image, n_seams=30, direction='vertical')
    carved_np = tensor_to_numpy(carved)

    save_comparison(
        [original_np, carved_np],
        [f"Original ({W}px wide)", f"After 30 seams ({carved.shape[2]}px wide)"],
        "fig_traditional_carving.png",
        "Traditional Seam Carving — seams avoid the bright stripe"
    )


# -----------------------------------------------------------------------
# Figure 2: Lattice-guided carving on synthetic arch
# Shows the core value: lattice-guided preserves arch silhouette
# -----------------------------------------------------------------------

def figure_arch_carving():
    """Create a semicircular arch and carve it:
    - Traditional: distorts the arch shape
    - Lattice-guided: preserves the arch by carving along the curve"""
    print("\n--- Figure: Arch Carving (Figure 3 analog) ---")
    torch.manual_seed(42)
    H, W = 200, 300

    # Create arch image
    cy, cx = H - 20, W // 2
    outer_r, inner_r = 80, 55
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    image = torch.full((3, H, W), 0.15)  # dark background
    arch_mask = (dist >= inner_r) & (dist <= outer_r) & (yy < cy)
    image[0][arch_mask] = 0.85
    image[1][arch_mask] = 0.65
    image[2][arch_mask] = 0.35
    # Add noise for texture
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    n_seams = 20

    # Traditional carving
    trad = carve_image_traditional(image, n_seams=n_seams, direction='vertical')

    # Lattice-guided: build a lattice following the arch
    # Semicircular curve for the arch centerline
    angles = torch.linspace(np.pi, 0, 60)
    mid_r = (inner_r + outer_r) / 2
    arch_x = cx + mid_r * torch.cos(angles)
    arch_y = cy + mid_r * torch.sin(angles)
    curve_pts = torch.stack([arch_x, arch_y], dim=1)

    lat = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=H / 2)
    lattice_carved = carve_image_lattice_guided(
        image, lat, n_seams=n_seams, lattice_width=W,
        roi_bounds=(0.0, float(W))
    )

    save_comparison(
        [tensor_to_numpy(image), tensor_to_numpy(trad), tensor_to_numpy(lattice_carved)],
        [f"Original ({W}px)", f"Traditional ({trad.shape[2]}px)", f"Lattice-guided ({W}px)"],
        "fig_arch_carving.png",
        "Arch Carving: Traditional vs Lattice-Guided"
    )


# -----------------------------------------------------------------------
# Figure 3: Seam pairs on synthetic bagel (shrink the hole)
# This is the Figure 10/22 analog from the paper
# -----------------------------------------------------------------------

def figure_synthetic_bagel_seam_pairs():
    """Create a synthetic bagel and use seam pairs to shrink the hole
    while keeping the outer boundary unchanged."""
    print("\n--- Figure: Synthetic Bagel Seam Pairs (Figure 10 analog) ---")
    torch.manual_seed(42)
    H, W = 200, 200
    cx, cy = 100.0, 100.0
    inner_r, outer_r = 30, 80

    # Create bagel image
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    image = torch.full((3, H, W), 0.15)
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    image[0][ring_mask] = 0.85
    image[1][ring_mask] = 0.70
    image[2][ring_mask] = 0.40
    # Texture
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    # Build circular lattice from curve points
    theta = torch.linspace(0, 2 * np.pi, 80)
    mid_r = (inner_r + outer_r) / 2
    circle_x = cx + mid_r * torch.cos(theta)
    circle_y = cy + mid_r * torch.sin(theta)
    curve_pts = torch.stack([circle_x, circle_y], dim=1)

    perp = (outer_r - inner_r) / 2 + 10
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=64, perp_extent=perp, cyclic=True)

    # Seam pairs: ROI targets inner ring edge, pair targets outer ring edge.
    # In lattice u-coordinates:
    #   u = perp_extent → centerline (on the circle at mid_r)
    #   u = perp_extent - (mid_r - inner_r) → inner ring edge
    #   u = perp_extent + (outer_r - mid_r) → outer ring edge
    lattice_w = int(2 * perp)
    inner_u = int(perp - (mid_r - inner_r))  # ≈ u of inner edge
    outer_u = int(perp + (outer_r - mid_r))  # ≈ u of outer edge
    # Tight windows around the edges so seams follow the ring boundaries
    roi_u = (max(0, inner_u - 8), inner_u + 8)
    pair_u = (outer_u - 8, min(lattice_w, outer_u + 8))

    n_seams = 5
    carved = carve_seam_pairs(image, lat, n_seams=n_seams,
                               roi_range=roi_u, pair_range=pair_u,
                               lattice_width=lattice_w)

    save_comparison(
        [tensor_to_numpy(image), tensor_to_numpy(carved)],
        ["Original", f"After {n_seams} seam pairs (hole shrinks)"],
        "fig_synthetic_bagel_pairs.png",
        "Seam Pairs on Synthetic Bagel — Shrink Hole, Preserve Boundary"
    )

    # Also show the lattice space view
    energy = gradient_magnitude_energy(image)
    energy_3d = energy.unsqueeze(0)
    lattice_energy = lat.resample_to_lattice_space(energy_3d, lattice_w).squeeze(0)
    lattice_img = lat.resample_to_lattice_space(image, lattice_w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(tensor_to_numpy(lattice_img), aspect='auto')
    axes[0].set_title("Image in Lattice Space")
    axes[0].set_xlabel("u (radial)")
    axes[0].set_ylabel("n (angular)")

    axes[1].imshow(lattice_energy.cpu().numpy(), cmap='hot', aspect='auto')
    axes[1].axvline(roi_u[0], color='cyan', linestyle='--', label='ROI')
    axes[1].axvline(roi_u[1], color='cyan', linestyle='--')
    axes[1].axvline(pair_u[0], color='magenta', linestyle='--', label='Pair')
    axes[1].axvline(pair_u[1], color='magenta', linestyle='--')
    axes[1].legend()
    axes[1].set_title("Energy in Lattice Space")
    axes[1].set_xlabel("u (radial)")

    # Show seam on energy
    seam = greedy_seam_windowed(normalize_energy(lattice_energy), roi_u)
    n_idx = np.arange(len(seam))
    axes[2].imshow(lattice_energy.cpu().numpy(), cmap='hot', aspect='auto', alpha=0.6)
    axes[2].plot(seam.cpu().numpy(), n_idx, 'cyan', linewidth=2, label='ROI seam')
    pair_seam = greedy_seam_windowed(normalize_energy(lattice_energy), pair_u)
    axes[2].plot(pair_seam.cpu().numpy(), n_idx, 'magenta', linewidth=2, label='Pair seam')
    axes[2].legend()
    axes[2].set_title("Seams in Lattice Space")

    plt.tight_layout()
    path = OUTPUT_DIR / "fig_synthetic_bagel_lattice_view.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Figure 4: Seam pairs on real double-bagel image
# THE DEMO: shrink/grow the left bagel without affecting the right
# -----------------------------------------------------------------------

def figure_real_bagel_seam_pairs():
    """Load the real double-bagel image and use seam pairs to
    shrink the left bagel while the right stays fixed."""
    print("\n--- Figure: Real Double Bagel Seam Pairs ---")

    bagel_path = Path(__file__).parent.parent / "bagel_double.jpg"
    if not bagel_path.exists():
        print(f"  SKIPPED: {bagel_path} not found")
        return

    # Load image
    pil_img = Image.open(bagel_path).convert('RGB')
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    # Convert to (C, H, W) tensor
    image = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    C, H, W = image.shape
    print(f"  Image: {W}x{H}, {C} channels")

    # The left bagel is roughly centered at (W*0.27, H*0.50) with radius ~H*0.35
    # We'll define a circular lattice around it
    left_cx = W * 0.27
    left_cy = H * 0.50
    left_radius = H * 0.30  # approximate radius of the left bagel

    # Circular curve around the left bagel
    theta = torch.linspace(0, 2 * np.pi, 100)
    circle_x = left_cx + left_radius * torch.cos(theta)
    circle_y = left_cy + left_radius * torch.sin(theta)
    curve_pts = torch.stack([circle_x, circle_y], dim=1)

    perp = left_radius * 0.6  # how far scanlines extend
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=48, perp_extent=perp, cyclic=True)

    lattice_w = int(2 * perp)
    # ROI: inner portion (bagel body), Pair: outer portion (background)
    roi_u = (int(perp * 0.3), int(perp * 0.9))
    pair_u = (int(perp * 1.3), int(perp * 1.8))

    results = [tensor_to_numpy(image)]
    titles = ["Original"]

    for n_seams in [3, 8, 15]:
        carved = carve_seam_pairs(image, lat, n_seams=n_seams,
                                   roi_range=roi_u, pair_range=pair_u,
                                   lattice_width=lattice_w)
        results.append(tensor_to_numpy(carved))
        titles.append(f"{n_seams} seam pairs")

    save_comparison(
        results, titles,
        "fig_real_bagel_seam_pairs.png",
        "Real Bagel: Shrink Left Half via Seam Pairs"
    )

    # Also show lattice overlay on original
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(tensor_to_numpy(image))

    # Draw a few scanlines to show lattice coverage
    for i in range(0, lat.n_lines, lat.n_lines // 16):
        u_vals = torch.linspace(0, float(lattice_w), 50)
        n_val = float(i)
        pts = torch.stack([u_vals, torch.full_like(u_vals, n_val)], dim=1)
        world_pts = lat.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'cyan', alpha=0.5, linewidth=0.8)

    # Draw perpendicular lines (constant u)
    n_max = lat.n_lines
    for u_val in np.linspace(0, lattice_w, 12):
        n_vals = torch.linspace(0, float(n_max), 100)
        pts = torch.stack([torch.full_like(n_vals, u_val), n_vals], dim=1)
        world_pts = lat.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'yellow', alpha=0.3, linewidth=0.8)

    ax.set_title("Lattice overlay on double bagel", fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    path = OUTPUT_DIR / "fig_real_bagel_lattice_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# -----------------------------------------------------------------------
# Figure 5: No-blur comparison (naive vs correct)
# -----------------------------------------------------------------------

def figure_no_blur_comparison():
    """Show that carving-the-mapping avoids blur from double interpolation."""
    print("\n--- Figure: No-Blur Comparison (Section 3.3) ---")
    torch.manual_seed(42)
    H, W = 80, 120

    # Checkerboard with fine detail
    image = torch.zeros(1, H, W)
    for y in range(H):
        for x in range(W):
            image[0, y, x] = float(((x // 6) + (y // 6)) % 2)

    # Gentle curve
    curve_pts = torch.tensor([
        [0.0, H / 2 - 5], [W / 3, H / 2 + 5],
        [2 * W / 3, H / 2 - 5], [float(W - 1), H / 2 + 5]
    ])
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=H / 2)

    from src.carving import _carve_image_lattice_naive
    n_seams = 8
    naive = _carve_image_lattice_naive(image, lat, n_seams=n_seams, lattice_width=W)
    correct = carve_image_lattice_guided(image, lat, n_seams=n_seams, lattice_width=W)

    save_comparison(
        [tensor_to_numpy(image), tensor_to_numpy(naive), tensor_to_numpy(correct)],
        ["Original", f"Naive (double interp, {n_seams} seams)", f"Correct (carve mapping, {n_seams} seams)"],
        "fig_no_blur_comparison.png",
        "Section 3.3: Naive vs Correct — Double Interpolation Causes Blur"
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    figure_traditional_carving()
    figure_arch_carving()
    figure_no_blur_comparison()
    figure_synthetic_bagel_seam_pairs()
    figure_real_bagel_seam_pairs()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Open them to visually validate the carving algorithms.")
