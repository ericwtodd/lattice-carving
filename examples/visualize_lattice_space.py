"""
Visualize seam computation in lattice index space (like Figure 12).

Shows:
1. Image resampled to lattice index space
2. Energy map in lattice index space
3. Computed seam path in lattice space

This helps validate that our resampling and seam computation are working correctly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lattice import Lattice2D
from energy import gradient_magnitude_energy
from seam import greedy_seam, greedy_seam_windowed


def visualize_lattice_space_computation(
    image: torch.Tensor,
    lattice: Lattice2D,
    title: str,
    roi_window=None,
    pair_window=None,
    output_path=None
):
    """
    Visualize seam computation in lattice index space.

    Args:
        image: Input image (C, H, W) or (H, W)
        lattice: Lattice structure
        title: Title for the plot
        roi_window: Optional (u_start, u_end) for ROI seam
        pair_window: Optional (u_start, u_end) for pair seam
        output_path: Where to save the visualization
    """
    device = image.device

    # Handle grayscale vs color
    if image.dim() == 2:
        image = image.unsqueeze(0)

    # Resample image to lattice space
    # Get lattice dimensions
    n_lines = len(lattice.origins)
    lattice_width = int(2 * lattice._perp_extent)

    lattice_image = lattice.resample_to_lattice_space(image, lattice_width)

    # Compute energy in lattice space
    # Convert to grayscale if needed for energy computation
    if lattice_image.shape[0] == 3:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)
        lattice_gray = (lattice_image * weights).sum(dim=0)
    else:
        lattice_gray = lattice_image.squeeze(0)

    energy = gradient_magnitude_energy(lattice_gray)

    # Compute seams
    if roi_window is not None and pair_window is not None:
        # Compute seam pair
        roi_seam = greedy_seam_windowed(energy, roi_window, direction='vertical')
        pair_seam = greedy_seam_windowed(energy, pair_window, direction='vertical')
        seams = [('ROI', roi_seam, 'cyan'), ('Pair', pair_seam, 'magenta')]
    else:
        # Compute single seam
        seam = greedy_seam(energy, direction='vertical')
        seams = [('Seam', seam, 'yellow')]

    # Create visualization
    n_panels = 2 + len(seams)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 1: Image in lattice space
    ax = axes[0]
    if lattice_image.shape[0] == 3:
        # RGB
        display_img = lattice_image.permute(1, 2, 0).cpu().numpy()
    else:
        # Grayscale
        display_img = lattice_image.squeeze(0).cpu().numpy()

    ax.imshow(display_img, cmap='gray' if lattice_image.shape[0] == 1 else None,
              aspect='auto', origin='lower')
    ax.set_xlabel('u (perpendicular)', fontsize=10)
    ax.set_ylabel('n (scanline index)', fontsize=10)
    ax.set_title('Image in Lattice Space', fontsize=11, fontweight='bold')

    # Add window boundaries if provided
    if roi_window is not None:
        ax.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='ROI window')
        ax.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    if pair_window is not None:
        ax.axvline(pair_window[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Pair window')
        ax.axvline(pair_window[1], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    if roi_window is not None or pair_window is not None:
        ax.legend(fontsize=8, loc='upper right')

    # Panel 2: Energy in lattice space
    ax = axes[1]
    energy_np = energy.cpu().numpy()
    im = ax.imshow(energy_np, cmap='hot', aspect='auto', origin='lower')
    ax.set_xlabel('u (perpendicular)', fontsize=10)
    ax.set_ylabel('n (scanline index)', fontsize=10)
    ax.set_title('Energy in Lattice Space', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add window boundaries
    if roi_window is not None:
        ax.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    if pair_window is not None:
        ax.axvline(pair_window[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(pair_window[1], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

    # Panels 3+: Seams overlaid on energy
    for i, (seam_name, seam, color) in enumerate(seams):
        ax = axes[2 + i]
        ax.imshow(energy_np, cmap='hot', aspect='auto', origin='lower', alpha=0.6)

        # Plot seam
        n_indices = np.arange(len(seam))
        u_indices = seam.cpu().numpy()
        ax.plot(u_indices, n_indices, color=color, linewidth=2, label=f'{seam_name} seam')
        ax.scatter(u_indices[::max(1, len(seam)//20)], n_indices[::max(1, len(seam)//20)],
                  color=color, s=20, zorder=5)

        ax.set_xlabel('u (perpendicular)', fontsize=10)
        ax.set_ylabel('n (scanline index)', fontsize=10)
        ax.set_title(f'{seam_name} Seam in Lattice Space', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')

        # Add window boundaries
        if roi_window is not None:
            ax.axvline(roi_window[0], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.axvline(roi_window[1], color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
        if pair_window is not None:
            ax.axvline(pair_window[0], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
            ax.axvline(pair_window[1], color='orange', linestyle='--', linewidth=1.5, alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {output_path}")

    plt.close()


def create_test_images():
    """Create simple test images for visualization."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test 1: Vertical stripes (should map to horizontal stripes in lattice space)
    print("\n" + "="*70)
    print("TEST 1: Vertical Stripes with Circular Lattice")
    print("="*70)

    height, width = 400, 400
    image = torch.zeros(height, width, device=device)
    for i in range(0, width, 40):
        image[:, i:i+20] = 0.8

    # Add noise
    image += (torch.rand(height, width, device=device) - 0.5) * 0.1
    image = torch.clamp(image, 0, 1)

    # Circular lattice (radial scanlines)
    center = (width // 2, height // 2)
    radius = 150
    n_lines = 32
    perp_extent = 80

    # Create circle points
    theta = torch.linspace(0, 2 * np.pi, 100, device=device)
    circle_x = center[0] + radius * torch.cos(theta)
    circle_y = center[1] + radius * torch.sin(theta)
    curve_points = torch.stack([circle_x, circle_y], dim=1)

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=n_lines,
        perp_extent=perp_extent,
        cyclic=True,
        device=device
    )

    visualize_lattice_space_computation(
        image=image,
        lattice=lattice,
        title="Vertical Stripes → Circular Lattice Space",
        roi_window=(30, 130),
        pair_window=(140, 160),
        output_path="../output/lattice_space_stripes.png"
    )

    # Test 2: Concentric circles (should map to vertical stripes in lattice space)
    print("\n" + "="*70)
    print("TEST 2: Concentric Circles with Circular Lattice")
    print("="*70)

    xx, yy = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing='xy'
    )

    dist = torch.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    # Create concentric circles
    image = torch.zeros(height, width, device=device)
    for r in range(50, 250, 30):
        mask = (dist >= r) & (dist < r + 15)
        image[mask] = 0.8

    # Add noise
    image += (torch.rand(height, width, device=device) - 0.5) * 0.1
    image = torch.clamp(image, 0, 1)

    visualize_lattice_space_computation(
        image=image,
        lattice=lattice,
        title="Concentric Circles → Circular Lattice Space",
        roi_window=(30, 130),
        pair_window=(140, 160),
        output_path="../output/lattice_space_circles.png"
    )

    # Test 3: Sinusoidal curve with curved lattice
    print("\n" + "="*70)
    print("TEST 3: Horizontal Band with Curved Lattice")
    print("="*70)

    height, width = 400, 500
    image = torch.full((height, width), 0.2, device=device)

    # Create sinusoidal curve
    x_vals = torch.linspace(50, width - 50, 40, device=device)
    y_vals = height / 2 + 60 * torch.sin(x_vals / 80)
    curve_points = torch.stack([x_vals, y_vals], dim=1)

    # Create band along curve
    curve_np = curve_points.cpu().numpy()
    xx_np, yy_np = np.meshgrid(np.arange(width), np.arange(height))
    min_dist = np.full((height, width), float('inf'), dtype=np.float32)

    for cx, cy in curve_np:
        dist = np.sqrt((xx_np - cx)**2 + (yy_np - cy)**2)
        min_dist = np.minimum(min_dist, dist)

    min_dist_torch = torch.from_numpy(min_dist).to(device)
    image = torch.where(min_dist_torch < 30, torch.tensor(0.8, device=device), image)

    # Add noise
    image += (torch.rand(height, width, device=device) - 0.5) * 0.1
    image = torch.clamp(image, 0, 1)

    # Create curved lattice
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=35,
        perp_extent=120,
        cyclic=False,
        device=device
    )

    visualize_lattice_space_computation(
        image=image,
        lattice=lattice,
        title="Sinusoidal Band → Curved Lattice Space",
        roi_window=(100, 140),
        pair_window=(160, 200),
        output_path="../output/lattice_space_sine.png"
    )

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated lattice-space visualizations in ../output/:")
    print("  - lattice_space_stripes.png (vertical stripes → circular)")
    print("  - lattice_space_circles.png (concentric circles → circular)")
    print("  - lattice_space_sine.png (sinusoidal band → curved)")


if __name__ == "__main__":
    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    create_test_images()
