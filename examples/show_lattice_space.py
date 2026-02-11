"""
Simple lattice-space visualization (like Figure 12).
Shows test images in lattice index space with energy and seams.

First run: python create_test_images.py (creates test images)
Then run: python show_lattice_space.py (visualizes them)
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
from seam import greedy_seam_windowed


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(__file__).parent.parent / "output"

    # Check if test images exist
    bagel_path = output_dir / "test_bagel.png"
    if not bagel_path.exists():
        print("Test images not found. Running create_test_images.py...")
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "create_test_images.py")])

    print("\n" + "="*70)
    print("LATTICE-SPACE VISUALIZATION (Figure 12 style)")
    print("="*70)

    # Load bagel image
    print("\nLoading test_bagel.png...")
    bagel_img = Image.open(bagel_path)
    image = torch.from_numpy(np.array(bagel_img)).float().to(device) / 255.0
    image = image.unsqueeze(0)  # Add channel dimension

    height, width = image.shape[1], image.shape[2]
    center = (width // 2, height // 2)

    # Create circular lattice at middle radius
    print("Creating circular lattice...")
    theta = torch.linspace(0, 2 * np.pi, 100, device=device)
    circle_x = center[0] + 150 * torch.cos(theta)
    circle_y = center[1] + 150 * torch.sin(theta)
    curve_points = torch.stack([circle_x, circle_y], dim=1)

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=32,
        perp_extent=80,
        cyclic=True,
        device=device
    )

    # Resample to lattice space
    print("Resampling to lattice space...")
    lattice_width = 160
    lattice_image = lattice.resample_to_lattice_space(image, lattice_width).squeeze(0)

    print(f"Lattice space shape: {lattice_image.shape}")
    print(f"  n (scanlines): {lattice_image.shape[0]}")
    print(f"  u (perpendicular): {lattice_image.shape[1]}")

    # Compute energy
    print("Computing energy...")
    energy = gradient_magnitude_energy(lattice_image)

    # Compute seam pair
    print("Computing seams...")
    roi_window = (30, 130)  # Bagel donut region
    pair_window = (140, 160)  # Background outside

    roi_seam = greedy_seam_windowed(energy, roi_window, direction='vertical')
    pair_seam = greedy_seam_windowed(energy, pair_window, direction='vertical')

    # Visualize
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Image in lattice space
    axes[0].imshow(lattice_image.cpu().numpy(), cmap='gray', aspect='auto', origin='lower')
    axes[0].axvline(roi_window[0], color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='ROI')
    axes[0].axvline(roi_window[1], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].axvline(pair_window[0], color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Pair')
    axes[0].axvline(pair_window[1], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_title('Image in Lattice Space', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('u (radius from centerline)', fontsize=10)
    axes[0].set_ylabel('n (scanline index)', fontsize=10)
    axes[0].legend(fontsize=9, loc='upper right')

    # Panel 2: Energy
    energy_np = energy.cpu().numpy()
    im = axes[1].imshow(energy_np, cmap='hot', aspect='auto', origin='lower')
    axes[1].axvline(roi_window[0], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(roi_window[1], color='yellow', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(pair_window[0], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].axvline(pair_window[1], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    axes[1].set_title('Energy Map', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('u (radius)', fontsize=10)
    axes[1].set_ylabel('n (scanline index)', fontsize=10)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: ROI seam
    axes[2].imshow(energy_np, cmap='hot', aspect='auto', origin='lower', alpha=0.6)
    n_indices = np.arange(len(roi_seam))
    axes[2].plot(roi_seam.cpu().numpy(), n_indices, 'cyan', linewidth=2.5, label='ROI seam')
    axes[2].scatter(roi_seam.cpu().numpy()[::2], n_indices[::2], color='cyan', s=30,
                   edgecolors='white', linewidths=0.5, zorder=5)
    axes[2].axvline(roi_window[0], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[2].axvline(roi_window[1], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[2].set_title('ROI Seam', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('u (radius)', fontsize=10)
    axes[2].legend(fontsize=9)

    # Panel 4: Pair seam
    axes[3].imshow(energy_np, cmap='hot', aspect='auto', origin='lower', alpha=0.6)
    axes[3].plot(pair_seam.cpu().numpy(), n_indices, 'magenta', linewidth=2.5, label='Pair seam')
    axes[3].scatter(pair_seam.cpu().numpy()[::2], n_indices[::2], color='magenta', s=30,
                   edgecolors='white', linewidths=0.5, zorder=5)
    axes[3].axvline(pair_window[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[3].axvline(pair_window[1], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[3].set_title('Pair Seam', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('u (radius)', fontsize=10)
    axes[3].legend(fontsize=9)

    plt.suptitle('Bagel in Circular Lattice Space (sesame seeds visible!)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "lattice_space_bagel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    print("\n" + "="*70)
    print("What to look for:")
    print("  - Panel 1: Bagel appears as horizontal bands (constant radius)")
    print("  - Sesame seeds should be visible as small bright spots")
    print("  - Panel 2: High energy at edges (u~30 and u~130)")
    print("  - Panels 3&4: Seams follow low-energy paths")
    print("  - Check for artifacts or distortions (indicates overlapping lattice)")
    print("="*70)
