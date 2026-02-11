"""
Debug visualization for lattice-guided seam carving.

Shows step-by-step what's happening:
1. Original image
2. Lattice structure overlayed on image
3. Energy map with seam pairs overlayed
4. Final result
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.lattice import Lattice2D
from src.energy import gradient_magnitude_energy
from src.seam import greedy_seam_windowed


def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs('../output', exist_ok=True)


def create_bagel(size=512, hole_radius=0.2, bagel_radius=0.5, device='cpu'):
    """Create a bagel test image."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing='ij'
    )
    r = torch.sqrt(x**2 + y**2)

    # Bagel body with gradient
    image = torch.zeros(size, size, device=device)
    in_bagel = (r > hole_radius) & (r < bagel_radius)
    image[in_bagel] = 0.7 + 0.2 * torch.cos(10 * r[in_bagel])

    # Add some texture
    texture = 0.05 * torch.sin(20 * x) * torch.sin(20 * y)
    image[in_bagel] += texture[in_bagel]

    return image.clamp(0, 1)


def create_river(size=512, device='cpu'):
    """Create a river test image - horizontal flowing river."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing='ij'
    )

    # River path: sinusoidal curve
    river_center = 0.3 * torch.sin(3 * x)
    river_width = 0.15

    # Distance from river centerline
    dist_from_river = torch.abs(y - river_center)

    # River is dark, background is light
    image = torch.ones(size, size, device=device) * 0.8
    in_river = dist_from_river < river_width
    image[in_river] = 0.3 + 0.1 * torch.sin(10 * x[in_river])

    # Add some texture
    texture = 0.05 * torch.sin(15 * x) * torch.sin(15 * y)
    image += texture

    return image.clamp(0, 1)


def visualize_lattice_on_image(image, lattice, n_samples=20):
    """
    Overlay lattice structure on image.

    Shows the scanlines of the lattice.
    """
    H, W = image.shape
    device = image.device

    # Create RGB version of image
    if image.dim() == 2:
        img_rgb = torch.stack([image, image, image], dim=0)
    else:
        img_rgb = image.clone()

    # Sample points along each scanline
    u_vals = torch.linspace(0, W * 0.5, n_samples, device=device)

    # Draw every Nth scanline
    n_lines_to_draw = min(32, lattice.n_lines)
    line_indices = torch.linspace(0, lattice.n_lines - 1, n_lines_to_draw, dtype=torch.long)

    for n_idx in line_indices:
        # Points along this scanline
        lattice_pts = torch.stack([
            u_vals,
            torch.ones_like(u_vals) * n_idx
        ], dim=1)

        # Map to world space
        world_pts = lattice.inverse_mapping(lattice_pts)

        # Draw the line
        x_coords = world_pts[:, 0].cpu().numpy()
        y_coords = world_pts[:, 1].cpu().numpy()

        # Mark pixels along the line in red
        for x, y in zip(x_coords, y_coords):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                img_rgb[0, yi, xi] = 1.0  # Red channel
                img_rgb[1, yi, xi] = 0.0
                img_rgb[2, yi, xi] = 0.0

    return img_rgb


def visualize_seams_on_world_energy(energy, u_map, roi_seam, pair_seam, n_map):
    """
    Overlay seam positions onto world-space energy map (VECTORIZED).

    Args:
        energy: (H, W) energy map
        u_map: (H, W) precomputed u-coordinates for each pixel
        roi_seam: (n_lines,) seam positions in lattice space
        pair_seam: (n_lines,) seam positions in lattice space
        n_map: (H, W) precomputed n-coordinates for each pixel
    """
    H, W = energy.shape
    device = energy.device

    # Create RGB version of energy map
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy_rgb = torch.stack([energy_norm, energy_norm, energy_norm], dim=0)

    # Interpolate the seam positions at each pixel's n coordinate
    from src.carving import _interpolate_seam
    roi_seam_interp = _interpolate_seam(roi_seam, n_map)  # (H, W)
    pair_seam_interp = _interpolate_seam(pair_seam, n_map)  # (H, W)

    # Find pixels near each seam (vectorized)
    threshold = 3.0

    # ROI seam (yellow)
    roi_mask = torch.abs(u_map - roi_seam_interp) < threshold
    energy_rgb[0, roi_mask] = 1.0  # Red
    energy_rgb[1, roi_mask] = 1.0  # Green (Red + Green = Yellow)
    energy_rgb[2, roi_mask] = 0.0  # Blue

    # Pair seam (green)
    pair_mask = torch.abs(u_map - pair_seam_interp) < threshold
    energy_rgb[0, pair_mask] = 0.0  # Red
    energy_rgb[1, pair_mask] = 1.0  # Green
    energy_rgb[2, pair_mask] = 0.0  # Blue

    return energy_rgb


def debug_bagel():
    """Debug bagel seam pairs step-by-step."""
    print("="*70)
    print("DEBUGGING: Bagel Seam Pairs")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create bagel
    print("1. Creating bagel image...")
    image = create_bagel(size=512, hole_radius=0.2, bagel_radius=0.5, device=device)
    H, W = image.shape
    center = (W // 2, H // 2)
    print(f"   Image size: {H}×{W}")

    # Create circular lattice
    print("\n2. Creating circular lattice...")
    radius = W * 0.4
    n_lines = 256
    lattice = Lattice2D.circular(
        center=center,
        radius=radius,
        n_lines=n_lines,
        device=device
    )
    print(f"   Radius: {radius:.1f}")
    print(f"   Number of scanlines: {n_lines}")

    # Define ROI and pair ranges
    roi_range = (0, 60)       # Shrink hole
    pair_range = (140, 200)   # Expand background
    print(f"\n3. Seam pair regions:")
    print(f"   ROI (hole): u ∈ [{roi_range[0]}, {roi_range[1]}]")
    print(f"   Pair (background): u ∈ [{pair_range[0]}, {pair_range[1]}]")

    # Compute energy
    print("\n4. Computing energy...")
    energy = gradient_magnitude_energy(image)

    # Resample energy to lattice space
    print("5. Resampling energy to lattice space...")
    lattice_width = 256
    energy_3d = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
    lattice_energy = lattice_energy.squeeze(0)
    print(f"   Lattice energy shape: {lattice_energy.shape}")

    # Find seams in windowed regions
    print("\n6. Finding seams in each region...")
    roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction='vertical')
    print(f"   ROI seam shape: {roi_seam.shape}")
    print(f"   Pair seam shape: {pair_seam.shape}")
    print(f"   ROI seam range: [{roi_seam.min().item():.0f}, {roi_seam.max().item():.0f}]")
    print(f"   Pair seam range: [{pair_seam.min().item():.0f}, {pair_seam.max().item():.0f}]")

    # Compute forward mapping for world pixels
    print("\n7. Computing forward mapping for visualization...")
    from src.carving import _precompute_forward_mapping
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)

    # Create visualizations
    print("8. Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1, Col 1: Original image
    axes[0, 0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Bagel Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Lattice overlay
    img_with_lattice = visualize_lattice_on_image(image, lattice, n_samples=30)
    axes[0, 1].imshow(img_with_lattice.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Circular Lattice Structure\n(Red lines = scanlines)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Energy map with seam pairs overlayed
    print("   Creating world-space seam visualization...")
    energy_with_seams = visualize_seams_on_world_energy(energy, u_map, roi_seam, pair_seam, n_map)
    axes[0, 2].imshow(energy_with_seams.permute(1, 2, 0).cpu().numpy())
    axes[0, 2].set_title('Energy + Seam Pairs (World Space)\nYellow=ROI (shrink), Green=Pair (expand)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2, Col 1: Lattice energy
    axes[1, 0].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    axes[1, 0].set_title(f'Energy in Lattice Space\n({lattice_energy.shape[0]} scanlines × {lattice_energy.shape[1]} samples)', fontsize=12)
    axes[1, 0].set_xlabel('u (radial distance)', fontsize=10)
    axes[1, 0].set_ylabel('n (scanline index)', fontsize=10)

    # Row 2, Col 2: Lattice energy with ROI seam
    axes[1, 1].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    # Plot ROI seam
    scanline_indices = np.arange(len(roi_seam))
    axes[1, 1].plot(roi_seam.cpu().numpy(), scanline_indices, 'c-', linewidth=2, label='ROI seam')
    axes[1, 1].axvline(roi_range[0], color='cyan', linestyle='--', alpha=0.5, label='ROI window')
    axes[1, 1].axvline(roi_range[1], color='cyan', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Lattice Energy + ROI Seam\n(Cyan = remove to shrink hole)', fontsize=12)
    axes[1, 1].set_xlabel('u (radial distance)', fontsize=10)
    axes[1, 1].set_ylabel('n (scanline index)', fontsize=10)
    axes[1, 1].legend(fontsize=8)

    # Row 2, Col 3: Lattice energy with both seams
    axes[1, 2].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    axes[1, 2].plot(roi_seam.cpu().numpy(), scanline_indices, 'c-', linewidth=2, label='ROI seam (remove)')
    axes[1, 2].plot(pair_seam.cpu().numpy(), scanline_indices, 'lime', linewidth=2, label='Pair seam (insert)')
    axes[1, 2].axvline(roi_range[0], color='cyan', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(roi_range[1], color='cyan', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(pair_range[0], color='lime', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(pair_range[1], color='lime', linestyle='--', alpha=0.3)
    axes[1, 2].set_title('Both Seam Pairs\n(Cyan=shrink hole, Lime=expand background)', fontsize=12)
    axes[1, 2].set_xlabel('u (radial distance)', fontsize=10)
    axes[1, 2].set_ylabel('n (scanline index)', fontsize=10)
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../output/debug_bagel_setup.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/debug_bagel_setup.png")

    print("\n" + "="*70)
    print("✓ Debug visualization complete!")
    print("="*70)
    print("\nCheck ../output/debug_bagel_setup.png")
    print("\nThis shows:")
    print("  - Original bagel image")
    print("  - Circular lattice structure (red scanlines)")
    print("  - Energy map in world space")
    print("  - Energy map in lattice space")
    print("  - Seam pairs found in lattice space")


def debug_river():
    """Debug river lattice carving step-by-step."""
    print("="*70)
    print("DEBUGGING: River Lattice Carving")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create river
    print("1. Creating river image...")
    image = create_river(size=512, device=device)
    H, W = image.shape
    print(f"   Image size: {H}×{W}")

    # For river, use rectangular lattice (horizontal scanlines)
    print("\n2. Creating rectangular lattice...")
    lattice = Lattice2D.rectangular(height=H, width=W, device=device)
    print(f"   Number of scanlines: {lattice.n_lines}")

    # Compute energy
    print("\n3. Computing energy...")
    energy = gradient_magnitude_energy(image)

    # Resample energy to lattice space
    print("4. Resampling energy to lattice space...")
    lattice_width = W
    energy_3d = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
    lattice_energy = lattice_energy.squeeze(0)
    print(f"   Lattice energy shape: {lattice_energy.shape}")

    # Find one seam
    from src.seam import greedy_seam
    print("\n5. Finding vertical seam...")
    seam = greedy_seam(lattice_energy, direction='vertical')
    print(f"   Seam shape: {seam.shape}")

    # Create visualizations
    print("\n6. Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original image
    axes[0, 0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original River Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Lattice overlay (just show it's rectangular)
    img_with_lattice = visualize_lattice_on_image(image, lattice, n_samples=50)
    axes[0, 1].imshow(img_with_lattice.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Rectangular Lattice\n(Red = horizontal scanlines)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Energy map
    axes[1, 0].imshow(energy.cpu().numpy(), cmap='hot')
    axes[1, 0].set_title('Energy Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Lattice energy with seam
    axes[1, 1].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    scanline_indices = np.arange(len(seam))
    axes[1, 1].plot(seam.cpu().numpy(), scanline_indices, 'cyan', linewidth=2, label='Greedy seam')
    axes[1, 1].set_title('Lattice Energy + Seam', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('u (column)', fontsize=10)
    axes[1, 1].set_ylabel('n (row/scanline)', fontsize=10)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('../output/debug_river_setup.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/debug_river_setup.png")

    print("\n" + "="*70)
    print("✓ Debug visualization complete!")
    print("="*70)
    print("\nCheck ../output/debug_river_setup.png")


def main():
    create_output_dir()

    print("\nWhich test would you like to debug?")
    print("1. Bagel (seam pairs)")
    print("2. River (rectangular lattice)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        debug_bagel()
    elif choice == '2':
        debug_river()
    else:
        print("Running both...")
        debug_bagel()
        print("\n\n")
        debug_river()


if __name__ == '__main__':
    # Just run both for now
    create_output_dir()
    debug_bagel()
    print("\n\n")
    debug_river()
