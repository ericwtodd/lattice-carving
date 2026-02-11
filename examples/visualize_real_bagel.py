"""
Comprehensive visualization of real bagel image with lattice-guided carving.

Combines both visualization approaches:
1. World-space view: Original + lattice overlay + seams
2. Lattice-space view: Image + energy + seams in (u,n) coordinates

This helps validate the full pipeline on a real image.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lattice import Lattice2D
from energy import gradient_magnitude_energy
from seam import greedy_seam_windowed, greedy_seam_cyclic


def visualize_lattice_grid(ax, image, lattice, title="Lattice Overlay"):
    """Draw lattice grid on image."""
    ax.imshow(image, cmap='gray', origin='upper')

    # Draw scanlines (n-direction)
    for i in range(len(lattice.origins)):
        # Get lattice coordinates for this scanline
        n_val = float(i)
        u_vals = torch.linspace(-lattice._perp_extent, lattice._perp_extent, 50, device=lattice.origins.device)

        # Map to world space
        world_points = []
        for u_val in u_vals:
            lattice_coord = torch.tensor([u_val, n_val], device=lattice.origins.device)
            world_point = lattice.inverse_mapping(lattice_coord.unsqueeze(0))
            world_points.append(world_point.cpu().numpy()[0])

        world_points = np.array(world_points)
        ax.plot(world_points[:, 0], world_points[:, 1], 'cyan', alpha=0.3, linewidth=0.5)

    # Draw perpendicular lines (u-direction)
    n_lines = len(lattice.origins)
    is_cyclic = getattr(lattice, '_cyclic', False)
    n_max = n_lines if is_cyclic else n_lines - 1

    for u_val in np.linspace(-lattice._perp_extent, lattice._perp_extent, 10):
        n_vals = torch.linspace(0, n_max, 100, device=lattice.origins.device)

        world_points = []
        for n_val in n_vals:
            lattice_coord = torch.tensor([u_val, n_val], device=lattice.origins.device)
            world_point = lattice.inverse_mapping(lattice_coord.unsqueeze(0))
            world_points.append(world_point.cpu().numpy()[0])

        world_points = np.array(world_points)
        ax.plot(world_points[:, 0], world_points[:, 1], 'yellow', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')


def visualize_seams_world(ax, image, lattice, roi_seam, pair_seam,
                         roi_window, pair_window, title="Seams in World Space"):
    """Draw seams mapped back to world space."""
    ax.imshow(image, cmap='gray', origin='upper')

    # Map ROI seam to world space
    world_points_roi = []
    for n_idx, u_idx in enumerate(roi_seam.cpu().numpy()):
        lattice_coord = torch.tensor([u_idx, float(n_idx)], device=lattice.origins.device)
        world_point = lattice.inverse_mapping(lattice_coord.unsqueeze(0))
        world_points_roi.append(world_point.cpu().numpy()[0])

    world_points_roi = np.array(world_points_roi)
    ax.plot(world_points_roi[:, 0], world_points_roi[:, 1], 'cyan',
           linewidth=2.5, label='ROI seam')
    ax.scatter(world_points_roi[::3, 0], world_points_roi[::3, 1],
              color='cyan', s=20, zorder=5, edgecolors='white', linewidths=0.5)

    # Map Pair seam to world space
    world_points_pair = []
    for n_idx, u_idx in enumerate(pair_seam.cpu().numpy()):
        lattice_coord = torch.tensor([u_idx, float(n_idx)], device=lattice.origins.device)
        world_point = lattice.inverse_mapping(lattice_coord.unsqueeze(0))
        world_points_pair.append(world_point.cpu().numpy()[0])

    world_points_pair = np.array(world_points_pair)
    ax.plot(world_points_pair[:, 0], world_points_pair[:, 1], 'magenta',
           linewidth=2.5, label='Pair seam')
    ax.scatter(world_points_pair[::3, 0], world_points_pair[::3, 1],
              color='magenta', s=20, zorder=5, edgecolors='white', linewidths=0.5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.axis('off')


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real bagel image
    bagel_path = Path(__file__).parent.parent / "bagel.jpg"
    print(f"Loading {bagel_path}...")

    bagel_img = Image.open(bagel_path).convert('L')
    image_np = np.array(bagel_img)
    image = torch.from_numpy(image_np).float().to(device) / 255.0
    image_tensor = image.unsqueeze(0)  # Add channel dimension

    height, width = image.shape
    print(f"Image size: {width}x{height}")

    # Create circular lattice
    # User should adjust these parameters based on the real bagel!
    center = (width // 2, height // 2)
    middle_radius = min(width, height) // 3  # Adjust based on bagel size

    print(f"\nCreating circular lattice at center ({center[0]}, {center[1]}), radius {middle_radius}...")

    theta = torch.linspace(0, 2 * np.pi, 100, device=device)
    circle_x = center[0] + middle_radius * torch.cos(theta)
    circle_y = center[1] + middle_radius * torch.sin(theta)
    curve_points = torch.stack([circle_x, circle_y], dim=1)

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=32,
        perp_extent=min(width, height) // 4,  # Adjust based on bagel size
        cyclic=True,
        device=device
    )

    print(f"Lattice: {len(lattice.origins)} scanlines, perp_extent={lattice._perp_extent}")

    # Resample to lattice space
    print("\nResampling to lattice space...")
    lattice_width = int(2 * lattice._perp_extent)
    lattice_image = lattice.resample_to_lattice_space(image_tensor, lattice_width).squeeze(0)

    print(f"Lattice space shape: {lattice_image.shape} (n={lattice_image.shape[0]}, u={lattice_image.shape[1]})")

    # Compute energy
    print("Computing energy...")
    energy = gradient_magnitude_energy(lattice_image)

    # Compute seam pair
    # User should adjust these windows based on where the bagel is!
    roi_window = (int(lattice_width * 0.2), int(lattice_width * 0.7))  # Adjust for real bagel
    pair_window = (int(lattice_width * 0.75), int(lattice_width * 0.95))

    print(f"Computing seams with cyclic Gaussian guide...")
    print(f"  ROI window: {roi_window}")
    print(f"  Pair window: {pair_window}")

    # Use cyclic seam computation for circular lattice
    roi_seam = greedy_seam_cyclic(energy, roi_window, direction='vertical', guide_width=10.0)
    pair_seam = greedy_seam_cyclic(energy, pair_window, direction='vertical', guide_width=10.0)

    # Create comprehensive visualization
    print("\nCreating visualization...")
    fig = plt.figure(figsize=(20, 10))

    # Top row: World space views
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image_np, cmap='gray', origin='upper')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    visualize_lattice_grid(ax2, image_np, lattice, "Lattice Grid Overlay")

    ax3 = plt.subplot(2, 3, 3)
    visualize_seams_world(ax3, image_np, lattice, roi_seam, pair_seam,
                         roi_window, pair_window, "Seams in World Space")

    # Bottom row: Lattice space views
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(lattice_image.cpu().numpy(), cmap='gray', aspect='auto', origin='lower')
    ax4.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='ROI')
    ax4.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    ax4.axvline(pair_window[0], color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Pair')
    ax4.axvline(pair_window[1], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_title('Image in Lattice Space', fontsize=12, fontweight='bold')
    ax4.set_xlabel('u (radius)', fontsize=10)
    ax4.set_ylabel('n (angle)', fontsize=10)
    ax4.legend(fontsize=9, loc='upper right')

    ax5 = plt.subplot(2, 3, 5)
    energy_np = energy.cpu().numpy()
    im = ax5.imshow(energy_np, cmap='hot', aspect='auto', origin='lower')
    ax5.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    ax5.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    ax5.axvline(pair_window[0], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax5.axvline(pair_window[1], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_title('Energy in Lattice Space', fontsize=12, fontweight='bold')
    ax5.set_xlabel('u (radius)', fontsize=10)
    ax5.set_ylabel('n (angle)', fontsize=10)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(energy_np, cmap='hot', aspect='auto', origin='lower', alpha=0.6)
    n_indices = np.arange(len(roi_seam))
    ax6.plot(roi_seam.cpu().numpy(), n_indices, 'cyan', linewidth=2.5, label='ROI seam')
    ax6.plot(pair_seam.cpu().numpy(), n_indices, 'magenta', linewidth=2.5, label='Pair seam')
    ax6.scatter(roi_seam.cpu().numpy()[::2], n_indices[::2], color='cyan', s=30,
               edgecolors='white', linewidths=0.5, zorder=5)
    ax6.scatter(pair_seam.cpu().numpy()[::2], n_indices[::2], color='magenta', s=30,
               edgecolors='white', linewidths=0.5, zorder=5)
    ax6.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.axvline(pair_window[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.axvline(pair_window[1], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6.set_title('Seams in Lattice Space', fontsize=12, fontweight='bold')
    ax6.set_xlabel('u (radius)', fontsize=10)
    ax6.set_ylabel('n (angle)', fontsize=10)
    ax6.legend(fontsize=9, loc='upper right')

    plt.suptitle('Real Bagel - Lattice-Guided Carving Visualization',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "real_bagel_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print("\nTop row (World Space):")
    print("  1. Original bagel image")
    print("  2. Lattice grid overlay (cyan=scanlines, yellow=perpendicular)")
    print("  3. Seams mapped to world space (cyan=ROI, magenta=Pair)")
    print("\nBottom row (Lattice Space):")
    print("  4. Image in (u,n) coordinates - should show bagel as horizontal bands")
    print("  5. Energy map - high at edges")
    print("  6. Computed seams with Gaussian guide - should connect (cyclic)")
    print("\nNOTE: You may need to adjust:")
    print("  - middle_radius: Where to place the circular lattice")
    print("  - perp_extent: How far the lattice extends perpendicular to circle")
    print("  - roi_window/pair_window: Where seams should be constrained")
    print("="*70)
