"""
Test lattice construction and seam computation with visualization.

For each test case (sine, arch, river, bagel), we:
1. Define centerline points
2. Build fine-grained lattice
3. Visualize lattice structure
4. Compute seams in lattice space
5. Visualize seams mapped back to world space

Shows:
- Panel 1: Original image + centerline
- Panel 2: Image + lattice grid overlay
- Panel 3: Image + computed seam(s) overlay
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from lattice import Lattice2D
from energy import gradient_magnitude_energy


def create_test_image_from_curve(curve_points, height, width, band_width=60, device='cpu'):
    """Create a test image with a bright band following the curve, with texture.

    Args:
        curve_points: Centerline points defining the feature
        height, width: Image dimensions
        band_width: Width of the bright band around the curve
        device: torch device

    Returns:
        Image tensor (H, W) with values in [0, 1]
    """
    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Compute distance from each pixel to the curve
    curve_np = curve_points.cpu().numpy().astype(np.float32)
    xx_np = xx.cpu().numpy().astype(np.float32)
    yy_np = yy.cpu().numpy().astype(np.float32)

    # For each pixel, find minimum distance to any curve point
    min_dist = np.full((height, width), float('inf'), dtype=np.float32)
    for cx, cy in curve_np:
        dist = np.sqrt((xx_np - cx)**2 + (yy_np - cy)**2)
        min_dist = np.minimum(min_dist, dist)

    # Create bright band around curve
    min_dist_torch = torch.from_numpy(min_dist).to(dtype=torch.float32, device=device)
    image = torch.where(min_dist_torch < band_width,
                       torch.ones_like(min_dist_torch) * 0.8,  # Bright band
                       torch.ones_like(min_dist_torch) * 0.2)  # Dark background

    # Add texture (small noise variation)
    torch.manual_seed(42)
    noise = (torch.rand(height, width, dtype=torch.float32, device=device) - 0.5) * 0.15
    image = torch.clamp(image + noise, 0, 1)

    return image


def create_bagel_image(center, inner_radius, outer_radius, height, width, device='cpu'):
    """Create a bagel/donut image with a hole and sesame seed texture.

    Args:
        center: (cx, cy) center of bagel
        inner_radius: Radius of hole
        outer_radius: Outer radius of bagel
        height, width: Image dimensions
        device: torch device

    Returns:
        Image tensor (H, W) with values in [0, 1]
    """
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    cx, cy = float(center[0]), float(center[1])
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Base bagel ring
    image = torch.where((dist >= inner_radius) & (dist <= outer_radius),
                       torch.tensor(0.8, dtype=torch.float32, device=device),  # Bagel
                       torch.tensor(0.2, dtype=torch.float32, device=device))  # Background + hole

    # Add sesame seed texture (small bright ovals)
    torch.manual_seed(42)  # Reproducible seeds
    n_seeds = 100
    for _ in range(n_seeds):
        # Random position along bagel (ensure tensors are on correct device)
        angle = torch.rand(1, device=device) * 2 * np.pi
        radius = inner_radius + torch.rand(1, device=device) * (outer_radius - inner_radius)
        seed_x = cx + radius * torch.cos(angle)
        seed_y = cy + radius * torch.sin(angle)

        # Small oval shape (rotated to follow bagel curve)
        seed_dist = torch.sqrt((xx - seed_x)**2 + (yy - seed_y)**2)
        seed_mask = seed_dist < 3.0  # Small seeds
        image = torch.where(seed_mask,
                          torch.tensor(0.95, dtype=torch.float32, device=device),  # Bright seed
                          image)

    return image


def visualize_seams(image, curve_points, lattice, seam, seam2=None,
                    roi_range=None, pair_range=None, title="Seam Visualization", filename="seams.png"):
    """Visualize image with lattice and computed seams.

    Args:
        image: Original image (H, W)
        curve_points: Centerline points
        lattice: Lattice2D structure
        seam: First seam (n_lines,) in lattice coordinates
        seam2: Optional second seam for seam pairs
        roi_range: Optional ROI range to highlight
        pair_range: Optional pair range to highlight
        title: Plot title
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    curve_np = curve_points.cpu().numpy()
    image_np = image.cpu().numpy()
    device = image.device
    H, W = image.shape

    # Check if lattice is cyclic
    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic

    # Panel 1: Original image with centerline
    ax = axes[0]
    ax.imshow(image_np, cmap='gray', aspect='auto', vmin=0, vmax=1)

    # For cyclic curves, close the loop by connecting back to start
    if is_cyclic:
        curve_closed = np.vstack([curve_np, curve_np[0:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], 'r-', linewidth=2, label='Centerline')
    else:
        ax.plot(curve_np[:, 0], curve_np[:, 1], 'r-', linewidth=2, label='Centerline')
    ax.scatter(curve_np[:, 0], curve_np[:, 1], c='red', s=30, zorder=10)
    ax.set_title(f'{title}\nOriginal Image + Centerline', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # Panel 2: Image with lattice grid overlay
    ax = axes[1]
    ax.imshow(image_np, cmap='gray', aspect='auto', vmin=0, vmax=1, alpha=0.7)

    # Use 2*perp_extent for lattice width (full extent in u direction)
    if hasattr(lattice, '_perp_extent'):
        lattice_width = int(2 * lattice._perp_extent)
    else:
        lattice_width = 100
    n_lines = lattice.n_lines

    # Draw more grid lines to show cell structure
    grid_spacing_n = max(1, n_lines // 20)
    grid_spacing_u = max(1, lattice_width // 20)

    # Check if lattice is cyclic
    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic
    n_max = n_lines if is_cyclic else n_lines - 1  # For cyclic, extend to show wrap-around

    for n_idx in range(0, n_lines, grid_spacing_n):
        u_vals = torch.linspace(0, lattice_width, 100, dtype=torch.float32, device=device)
        n_vals = torch.full_like(u_vals, float(n_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'r-', alpha=0.5, linewidth=0.8)

    for u_idx in range(0, lattice_width, grid_spacing_u):
        n_vals = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
        u_vals = torch.full_like(n_vals, float(u_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'g-', alpha=0.5, linewidth=0.8)

    # Draw centerline (close loop for cyclic)
    if is_cyclic:
        curve_closed = np.vstack([curve_np, curve_np[0:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], 'b-', linewidth=2, label='Centerline')
    else:
        ax.plot(curve_np[:, 0], curve_np[:, 1], 'b-', linewidth=2, label='Centerline')

    # Draw ROI and pair range boundaries if provided
    if roi_range is not None:
        for u_boundary in [roi_range[0], roi_range[1]]:
            n_vals = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
            u_vals = torch.full_like(n_vals, float(u_boundary))
            lattice_pts = torch.stack([u_vals, n_vals], dim=1)
            world_pts = lattice.inverse_mapping(lattice_pts)
            world_np = world_pts.cpu().numpy()
            ax.plot(world_np[:, 0], world_np[:, 1], 'y--', alpha=0.8, linewidth=2, label='ROI boundary' if u_boundary == roi_range[0] else '')

    if pair_range is not None:
        for u_boundary in [pair_range[0], pair_range[1]]:
            n_vals = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
            u_vals = torch.full_like(n_vals, float(u_boundary))
            lattice_pts = torch.stack([u_vals, n_vals], dim=1)
            world_pts = lattice.inverse_mapping(lattice_pts)
            world_np = world_pts.cpu().numpy()
            ax.plot(world_np[:, 0], world_np[:, 1], 'orange', linestyle='--', alpha=0.8, linewidth=2, label='Pair boundary' if u_boundary == pair_range[0] else '')

    ax.set_title('Image + Lattice Grid', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    # Panel 3: Image with computed seam(s)
    ax = axes[2]
    ax.imshow(image_np, cmap='gray', aspect='auto', vmin=0, vmax=1, alpha=0.7)

    # Draw seam in world coordinates
    # For cyclic lattices, extend to n_lines to show wrap-around
    n_max = n_lines if is_cyclic else n_lines - 1
    n_vals = torch.linspace(0, n_max, 200, dtype=torch.float32, device=device)

    # Interpolate seam at fractional n values
    seam_float = seam.float()
    if is_cyclic:
        n_floor = torch.floor(n_vals).long() % n_lines
        n_ceil = (n_floor + 1) % n_lines
    else:
        n_floor = torch.floor(n_vals).long().clamp(0, n_lines - 1)
        n_ceil = (n_floor + 1).clamp(0, n_lines - 1)
    n_frac = n_vals - torch.floor(n_vals)
    seam_interp = (1.0 - n_frac) * seam_float[n_floor] + n_frac * seam_float[n_ceil]

    # Map to world space
    lattice_pts = torch.stack([seam_interp, n_vals], dim=1)
    world_pts = lattice.inverse_mapping(lattice_pts)
    world_np = world_pts.cpu().numpy()
    ax.plot(world_np[:, 0], world_np[:, 1], 'cyan', linewidth=3, label='Seam 1', zorder=10)

    if seam2 is not None:
        seam2_float = seam2.float()
        seam2_interp = (1.0 - n_frac) * seam2_float[n_floor] + n_frac * seam2_float[n_ceil]
        lattice_pts2 = torch.stack([seam2_interp, n_vals], dim=1)
        world_pts2 = lattice.inverse_mapping(lattice_pts2)
        world_np2 = world_pts2.cpu().numpy()
        ax.plot(world_np2[:, 0], world_np2[:, 1], 'magenta', linewidth=3, label='Seam 2', zorder=10)

    # Draw centerline (close loop for cyclic)
    if is_cyclic:
        curve_closed = np.vstack([curve_np, curve_np[0:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], 'r-', linewidth=2, label='Centerline', alpha=0.5)
    else:
        ax.plot(curve_np[:, 0], curve_np[:, 1], 'r-', linewidth=2, label='Centerline', alpha=0.5)

    # Draw ROI and pair range boundaries if provided
    if roi_range is not None:
        for u_boundary in [roi_range[0], roi_range[1]]:
            n_vals_b = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
            u_vals_b = torch.full_like(n_vals_b, float(u_boundary))
            lattice_pts_b = torch.stack([u_vals_b, n_vals_b], dim=1)
            world_pts_b = lattice.inverse_mapping(lattice_pts_b)
            world_np_b = world_pts_b.cpu().numpy()
            # Debug: print first point to see where it's actually drawn
            if u_boundary == roi_range[0]:
                print(f"      DEBUG: ROI inner boundary u={u_boundary} maps to world point {world_np_b[0]}")
            ax.plot(world_np_b[:, 0], world_np_b[:, 1], 'y--', alpha=0.8, linewidth=2, label='ROI window' if u_boundary == roi_range[0] else '')

    if pair_range is not None:
        for u_boundary in [pair_range[0], pair_range[1]]:
            n_vals_b = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
            u_vals_b = torch.full_like(n_vals_b, float(u_boundary))
            lattice_pts_b = torch.stack([u_vals_b, n_vals_b], dim=1)
            world_pts_b = lattice.inverse_mapping(lattice_pts_b)
            world_np_b = world_pts_b.cpu().numpy()
            # Debug: print first point to see where it's actually drawn
            if u_boundary == pair_range[0]:
                print(f"      DEBUG: Pair inner boundary u={u_boundary} maps to world point {world_np_b[0]}")
                # Calculate radius from center
                center_x, center_y = 256, 256
                radius = np.sqrt((world_np_b[0,0] - center_x)**2 + (world_np_b[0,1] - center_y)**2)
                print(f"      DEBUG: This is at radius {radius:.1f} from center (should be ~210)")
            ax.plot(world_np_b[:, 0], world_np_b[:, 1], 'orange', linestyle='--', alpha=0.8, linewidth=2, label='Pair window' if u_boundary == pair_range[0] else '')

    ax.set_title('Image + Computed Seam(s)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    plt.tight_layout()

    os.makedirs('../output', exist_ok=True)
    fig.savefig(f'../output/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: ../output/{filename}")


def visualize_lattice_grid(curve_points, lattice, title, filename):
    """Visualize centerline and lattice grid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    curve_np = curve_points.cpu().numpy()
    device = curve_points.device

    # Check if lattice is cyclic
    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic

    # Panel 1: Centerline + scanlines
    ax = axes[0]

    # For cyclic curves, close the loop
    if is_cyclic:
        curve_closed = np.vstack([curve_np, curve_np[0:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], 'b-', linewidth=3, label='Centerline', zorder=10)
    else:
        ax.plot(curve_np[:, 0], curve_np[:, 1], 'b-', linewidth=3, label='Centerline', zorder=10)
    ax.scatter(curve_np[:, 0], curve_np[:, 1], c='blue', s=40, zorder=11)

    # Draw sample scanlines
    n_lines = lattice.n_lines
    sample_every = max(1, n_lines // 15)  # Show ~15 scanlines

    if hasattr(lattice, '_perp_extent'):
        extent = lattice._perp_extent
    else:
        extent = 50

    for i in range(0, n_lines, sample_every):
        origin = lattice.origins[i].cpu().numpy()
        tangent = lattice.tangents[i].cpu().numpy()

        p1 = origin - extent * tangent
        p2 = origin + extent * tangent
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.4, linewidth=1)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'{title}\nCenterline + Scanlines', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Lattice grid
    ax = axes[1]

    if hasattr(lattice, '_perp_extent'):
        lattice_width = int(2 * lattice._perp_extent)
    else:
        lattice_width = 100

    # Draw grid
    grid_spacing_n = max(1, n_lines // 20)
    grid_spacing_u = max(1, lattice_width // 20)

    # Check if lattice is cyclic
    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic
    n_max = n_lines if is_cyclic else n_lines - 1  # For cyclic, extend to show wrap-around

    for n_idx in range(0, n_lines, grid_spacing_n):
        u_vals = torch.linspace(0, lattice_width, 100, dtype=torch.float32, device=device)
        n_vals = torch.full_like(u_vals, float(n_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'r-', alpha=0.5, linewidth=0.8)

    for u_idx in range(0, lattice_width, grid_spacing_u):
        n_vals = torch.linspace(0, n_max, 100, dtype=torch.float32, device=device)
        u_vals = torch.full_like(n_vals, float(u_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'g-', alpha=0.5, linewidth=0.8)

    # Draw centerline (close loop for cyclic)
    if is_cyclic:
        curve_closed = np.vstack([curve_np, curve_np[0:1]])
        ax.plot(curve_closed[:, 0], curve_closed[:, 1], 'b-', linewidth=3, label='Centerline', zorder=10)
    else:
        ax.plot(curve_np[:, 0], curve_np[:, 1], 'b-', linewidth=3, label='Centerline', zorder=10)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Lattice Grid (Red=along, Green=across)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('../output', exist_ok=True)
    fig.savefig(f'../output/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: ../output/{filename}")


def test_sine_wave():
    """Sine wave test."""
    print("\n" + "="*70)
    print("TEST 1: Sine Wave")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define centerline
    x = torch.linspace(0, 400, 80, dtype=torch.float32, device=device)
    y = 200 + 40 * torch.sin(2 * np.pi * x / 400)
    curve_points = torch.stack([x, y], dim=1)

    # Build fine-grained lattice (no seam pairs for sine - just single seam demo)
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=40,
        perp_extent=80,  # Cover the band (50px) plus margin
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    # Visualize lattice structure
    visualize_lattice_grid(curve_points, lattice, "Sine Wave", "lattice_sine.png")

    # Compute and visualize seam pairs (shrink sine band)
    print("   Computing seam pairs (shrink sine band)...")
    test_image = create_test_image_from_curve(curve_points, 400, 400, band_width=50, device=device)
    lattice_width = int(2 * lattice._perp_extent)  # 160

    # Compute energy and resample to lattice
    energy = gradient_magnitude_energy(test_image)
    if energy.dim() == 2:
        energy = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy, lattice_width).squeeze(0)

    # Define windows: sine band in middle, background on side
    # Centerline at u=80, band extends ±25px (u=55 to u=105)
    # ROI: Sine band (u=60-100)
    # Pair: Background (u=110-140)
    roi_range = (60, 100)
    pair_range = (110, 140)

    print(f"   Lattice width: {lattice_width}")
    print(f"   ROI range (sine band): {roi_range}")
    print(f"   Pair range (background): {pair_range}")

    # Find seam pairs in lattice space
    from seam import greedy_seam_windowed
    roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction='vertical')

    print(f"   ROI seam: {roi_seam.shape}, range: [{roi_seam.min()}, {roi_seam.max()}]")
    print(f"   Pair seam: {pair_seam.shape}, range: [{pair_seam.min()}, {pair_seam.max()}]")
    visualize_seams(test_image, curve_points, lattice, roi_seam, pair_seam,
                   roi_range=roi_range, pair_range=pair_range,
                   title="Sine Wave Seam Pairs (Shrink Band)", filename="seam_sine_pairs.png")


def test_arch():
    """Arch test - semicircular arc."""
    print("\n" + "="*70)
    print("TEST 2: Arch (Semicircle)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define arch centerline (semicircle)
    center = (200, 300)
    radius = 100
    angles = torch.linspace(0, np.pi, 60, dtype=torch.float32, device=device)
    x = center[0] + radius * torch.cos(angles)
    y = center[1] - radius * torch.sin(angles)
    curve_points = torch.stack([x, y], dim=1)

    # Build lattice - cover arch + background
    # Arch is ~40px wide, extend further to include background
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=30,
        perp_extent=80,  # Cover arch (40px) + background padding (40px each side)
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    # Visualize lattice structure
    visualize_lattice_grid(curve_points, lattice, "Arch", "lattice_arch.png")

    # Compute and visualize seam pairs (grow arch)
    print("   Computing seam pairs (grow arch)...")
    test_image = create_test_image_from_curve(curve_points, 400, 400, band_width=40, device=device)
    lattice_width = int(2 * lattice._perp_extent)  # 160

    # Compute energy and resample to lattice
    energy = gradient_magnitude_energy(test_image)
    if energy.dim() == 2:
        energy = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy, lattice_width).squeeze(0)

    # Define windows for growing the arch
    # Centerline at u=80, arch extends ±20px (u=60 to u=100)
    # ROI: Outer part of arch (u=90-110, to expand arch outward)
    # Pair: Background beyond arch (u=115-140)
    roi_range = (90, 110)
    pair_range = (115, 140)

    print(f"   Lattice width: {lattice_width}")
    print(f"   ROI range (arch outer): {roi_range}")
    print(f"   Pair range (background): {pair_range}")

    # Find seam pairs in lattice space
    from seam import greedy_seam_windowed
    roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction='vertical')

    print(f"   ROI seam: {roi_seam.shape}, range: [{roi_seam.min()}, {roi_seam.max()}]")
    print(f"   Pair seam: {pair_seam.shape}, range: [{pair_seam.min()}, {pair_seam.max()}]")
    visualize_seams(test_image, curve_points, lattice, roi_seam, pair_seam,
                   roi_range=roi_range, pair_range=pair_range,
                   title="Arch Seam Pairs (Grow Arch)", filename="seam_arch_pairs.png")


def test_river():
    """River test - sinusoidal horizontal flow."""
    print("\n" + "="*70)
    print("TEST 3: River (Sinusoidal)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # River centerline
    x = torch.linspace(0, 512, 80, dtype=torch.float32, device=device)
    y = 256 + 50 * torch.sin(3 * np.pi * x / 512)
    curve_points = torch.stack([x, y], dim=1)

    # Build lattice - cover river + background
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=35,
        perp_extent=120,  # Cover river (60px wide) + background (60px each side)
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    # Visualize lattice structure
    visualize_lattice_grid(curve_points, lattice, "River", "lattice_river.png")

    # Compute and visualize seam pairs (shrink river)
    print("   Computing seam pairs (shrink river)...")
    test_image = create_test_image_from_curve(curve_points, 512, 512, band_width=60, device=device)
    lattice_width = int(2 * lattice._perp_extent)  # 240

    # Compute energy and resample to lattice
    energy = gradient_magnitude_energy(test_image)
    if energy.dim() == 2:
        energy = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy, lattice_width).squeeze(0)

    # Define windows for shrinking the river
    # Centerline at u=120, river extends ±30px (u=90 to u=150)
    # ROI: River itself (u=100-140, to shrink river)
    # Pair: Background to one side (u=160-200)
    roi_range = (100, 140)
    pair_range = (160, 200)

    print(f"   Lattice width: {lattice_width}")
    print(f"   ROI range (river): {roi_range}")
    print(f"   Pair range (background): {pair_range}")

    # Find seam pairs in lattice space
    from seam import greedy_seam_windowed
    roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction='vertical')

    print(f"   ROI seam: {roi_seam.shape}, range: [{roi_seam.min()}, {roi_seam.max()}]")
    print(f"   Pair seam: {pair_seam.shape}, range: [{pair_seam.min()}, {pair_seam.max()}]")
    visualize_seams(test_image, curve_points, lattice, roi_seam, pair_seam,
                   roi_range=roi_range, pair_range=pair_range,
                   title="River Seam Pairs (Shrink River)", filename="seam_river_pairs.png")


def test_bagel():
    """Bagel test - circular centerline around the hole."""
    print("\n" + "="*70)
    print("TEST 4: Bagel (Circle)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Bagel centerline - circle at middle radius between hole and outer edge
    center = (256, 256)
    middle_radius = 150  # Between hole and outer edge (make bigger for visibility)
    angles = torch.linspace(0, 2*np.pi, 64, dtype=torch.float32, device=device)[:-1]  # Exclude duplicate
    x = center[0] + middle_radius * torch.cos(angles)
    y = center[1] + middle_radius * torch.sin(angles)
    curve_points = torch.stack([x, y], dim=1)

    # Build lattice (CYCLIC for closed curve)
    # Centerline at radius 150, need to cover bagel + background
    # Bagel will span radius 100-200 (100px wide), add background padding
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=32,
        perp_extent=80,  # Cover from inside hole to outside background
        cyclic=True,  # Connect last scanline to first (Section 3.5)
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Cyclic: {lattice._cyclic} (closed curve)")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    # Visualize lattice structure
    visualize_lattice_grid(curve_points, lattice, "Bagel", "lattice_bagel.png")

    # Compute and visualize seam pairs (grow bagel)
    print("   Computing seam pairs (grow bagel)...")
    # Create bagel image with hole (larger for visibility)
    center = (256, 256)
    inner_radius = 100  # Hole
    outer_radius = 200  # Outer edge (centerline at radius 150)
    test_image = create_bagel_image(center, inner_radius, outer_radius, 512, 512, device=device)
    lattice_width = int(2 * lattice._perp_extent)  # 160

    # Compute energy and resample to lattice
    energy = gradient_magnitude_energy(test_image)
    if energy.dim() == 2:
        energy = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy, lattice_width).squeeze(0)

    # Define windows for growing/shrinking the bagel (keeping hole size constant)
    # Centerline at u=80 (radius 150)
    # Bagel extends from radius 100-200 → u≈30 to u≈130
    # ROI: The entire bagel donut (u=30-130, radius 100-200)
    # Pair: Background well outside bagel (u=140-160, radius 210-230)
    # This allows growing bagel outward (insert in bagel, remove from background)
    roi_range = (30, 130)
    pair_range = (140, 160)

    print(f"   Lattice width: {lattice_width}")
    print(f"   perp_extent: {lattice._perp_extent}")
    print(f"   ROI range (full bagel donut): {roi_range}")
    print(f"   Pair range (background outside): {pair_range}")

    # Debug: check what radius these map to
    # For circular lattice: radius = centerline_radius + (u - perp_extent)
    # centerline is at radius 150
    centerline_radius = 150
    roi_inner_radius = centerline_radius + (roi_range[0] - lattice._perp_extent)
    roi_outer_radius = centerline_radius + (roi_range[1] - lattice._perp_extent)
    pair_inner_radius = centerline_radius + (pair_range[0] - lattice._perp_extent)
    pair_outer_radius = centerline_radius + (pair_range[1] - lattice._perp_extent)
    print(f"   ROI maps to radius: {roi_inner_radius} to {roi_outer_radius}")
    print(f"   Pair maps to radius: {pair_inner_radius} to {pair_outer_radius}")

    # Find seam pairs in lattice space
    # NOTE: For cyclic lattices, paper uses inverted Gaussian guide on energy to ensure
    # seam starts and ends at same location (Section 4.0.1, Cyclic Greedy Seams).
    # With sesame seed texture, seams should be more interesting than just circles.
    from seam import greedy_seam_windowed
    roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction='vertical')

    print(f"   ROI seam: {roi_seam.shape}, range: [{roi_seam.min()}, {roi_seam.max()}]")
    print(f"   Pair seam: {pair_seam.shape}, range: [{pair_seam.min()}, {pair_seam.max()}]")
    visualize_seams(test_image, curve_points, lattice, roi_seam, pair_seam,
                   roi_range=roi_range, pair_range=pair_range,
                   title="Bagel Seam Pairs (Grow Bagel)", filename="seam_bagel_pairs.png")


def main():
    """Run all lattice construction and seam computation tests."""
    print("="*70)
    print("LATTICE-GUIDED SEAM COMPUTATION TESTS")
    print("Testing lattice construction and seam finding")
    print("="*70)

    test_sine_wave()
    test_arch()
    test_river()
    test_bagel()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations in ../output/:")
    print("\nLattice structure:")
    print("  - lattice_sine.png")
    print("  - lattice_arch.png")
    print("  - lattice_river.png")
    print("  - lattice_bagel.png")
    print("\nSeam pairs visualization:")
    print("  - seam_sine_pairs.png (Shrink sine band)")
    print("  - seam_arch_pairs.png (Grow arch)")
    print("  - seam_river_pairs.png (Shrink river)")
    print("  - seam_bagel_pairs.png (Grow bagel)")
    print("\nEach seam visualization shows:")
    print("  Panel 1: Original image + centerline")
    print("  Panel 2: Image + lattice grid overlay")
    print("  Panel 3: Image + computed seam(s) in world space")


if __name__ == '__main__':
    main()
