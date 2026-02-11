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


def create_arch(size=512, device='cpu'):
    """
    Create an arch test image (Figure 3 from paper).

    Semicircular arch on a plain background.
    """
    y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Arch parameters
    center_x = size / 2
    base_y = size * 0.75  # Arch base at 75% down
    radius = size * 0.35  # Arch radius
    thickness = size * 0.08  # Arch thickness

    # Distance from arch center
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - base_y)**2)

    # Create arch (semicircle above the base)
    image = torch.ones(size, size, device=device) * 0.9  # Light background

    # Arch is dark, only upper half (y < base_y)
    in_arch = (dist_from_center > radius - thickness/2) & \
              (dist_from_center < radius + thickness/2) & \
              (y < base_y)

    image[in_arch] = 0.2

    # Add some texture to arch
    texture = 0.05 * torch.sin(30 * torch.atan2(y - base_y, x - center_x))
    image[in_arch] += texture[in_arch]

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


def visualize_lattice_grid_cells(image, lattice, seam, lattice_width, color=(0, 1, 1)):
    """
    Visualize lattice as a grid of cells in world space.

    Each cell in lattice space is mapped to a quadrilateral in world space.
    Cells that are part of the seam are colored.

    Args:
        image: (H, W) grayscale image
        lattice: Lattice2D instance
        seam: (n_lines,) seam positions in lattice space
        lattice_width: width of lattice space
        color: RGB color for seam cells
    """
    H, W = image.shape
    device = image.device

    # Create RGB version
    img_rgb = torch.stack([image, image, image], dim=0)

    # Sample grid points in lattice space
    n_lines = lattice.n_lines
    u_step = max(1, lattice_width // 40)  # Sample every ~40th column
    n_step = max(1, n_lines // 40)  # Sample every ~40th row

    # Draw grid lines at sampled positions
    for n_idx in range(0, n_lines, n_step):
        u_vals = torch.arange(0, lattice_width, dtype=torch.float32, device=device)
        lattice_pts = torch.stack([
            u_vals,
            torch.full_like(u_vals, float(n_idx))
        ], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)

        # Draw line
        for i in range(len(world_pts) - 1):
            x0, y0 = world_pts[i]
            x1, y1 = world_pts[i + 1]
            # Simple line drawing
            steps = int(max(abs(x1 - x0), abs(y1 - y0)))
            if steps > 0:
                for t in torch.linspace(0, 1, steps):
                    x = x0 + t * (x1 - x0)
                    y = y0 + t * (y1 - y0)
                    xi, yi = int(round(x.item())), int(round(y.item()))
                    if 0 <= xi < W and 0 <= yi < H:
                        img_rgb[:, yi, xi] = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(1)

    # Draw seam cells
    for n_idx in range(n_lines):
        u_seam = int(seam[n_idx].item())
        # Draw a filled quad for this cell
        corners = torch.tensor([
            [u_seam, n_idx],
            [u_seam + 1, n_idx],
            [u_seam + 1, min(n_idx + 1, n_lines - 1)],
            [u_seam, min(n_idx + 1, n_lines - 1)]
        ], dtype=torch.float32, device=device)

        world_corners = lattice.inverse_mapping(corners)

        # Fill the quadrilateral (approximate)
        xs = world_corners[:, 0].cpu().numpy()
        ys = world_corners[:, 1].cpu().numpy()
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        for yi in range(max(0, y_min), min(H, y_max + 1)):
            for xi in range(max(0, x_min), min(W, x_max + 1)):
                # Check if point is inside quad (simple bounding box for now)
                img_rgb[0, yi, xi] = color[0]
                img_rgb[1, yi, xi] = color[1]
                img_rgb[2, yi, xi] = color[2]

    return img_rgb


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


def visualize_seams_on_world_energy(energy, lattice, roi_seam, pair_seam):
    """
    Overlay seam positions onto world-space energy map.

    Uses inverse_mapping to convert seam positions from lattice space to world space.

    Args:
        energy: (H, W) energy map
        lattice: Lattice2D instance
        roi_seam: (n_lines,) seam u-positions in lattice space
        pair_seam: (n_lines,) seam u-positions in lattice space
    """
    H, W = energy.shape
    device = energy.device

    # Create RGB version of energy map
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy_rgb = torch.stack([energy_norm, energy_norm, energy_norm], dim=0)

    n_lines = len(roi_seam)

    # Convert ROI seam from lattice space to world space
    # For each scanline n, seam is at position (u_seam[n], n)
    roi_lattice_pts = torch.stack([
        roi_seam,  # u coordinates
        torch.arange(n_lines, dtype=torch.float32, device=device)  # n coordinates
    ], dim=1)  # (n_lines, 2)

    roi_world_pts = lattice.inverse_mapping(roi_lattice_pts)  # (n_lines, 2) -> (x, y)

    # Convert pair seam from lattice space to world space
    pair_lattice_pts = torch.stack([
        pair_seam,
        torch.arange(n_lines, dtype=torch.float32, device=device)
    ], dim=1)

    pair_world_pts = lattice.inverse_mapping(pair_lattice_pts)  # (n_lines, 2) -> (x, y)

    # Draw seams as lines on the energy map
    # ROI seam (yellow)
    for i in range(n_lines):
        x, y = roi_world_pts[i]
        xi, yi = int(round(x.item())), int(round(y.item()))

        # Draw a small circle around the point
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx*dx + dy*dy <= 4:  # Circle of radius 2
                    yy, xx = yi + dy, xi + dx
                    if 0 <= xx < W and 0 <= yy < H:
                        energy_rgb[0, yy, xx] = 1.0  # Red
                        energy_rgb[1, yy, xx] = 1.0  # Green (yellow)
                        energy_rgb[2, yy, xx] = 0.0

    # Pair seam (green)
    for i in range(n_lines):
        x, y = pair_world_pts[i]
        xi, yi = int(round(x.item())), int(round(y.item()))

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    yy, xx = yi + dy, xi + dx
                    if 0 <= xx < W and 0 <= yy < H:
                        energy_rgb[0, yy, xx] = 0.0
                        energy_rgb[1, yy, xx] = 1.0  # Green
                        energy_rgb[2, yy, xx] = 0.0

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
    print("\n2. Creating COARSE circular lattice...")
    radius = W * 0.4
    n_lines = 32  # COARSE (not 256!)
    lattice = Lattice2D.circular(
        center=center,
        radius=radius,
        n_lines=n_lines,
        device=device
    )
    print(f"   Radius: {radius:.1f}")
    print(f"   Number of radial scanlines: {n_lines} (coarse, like paper)")

    # Define ROI and pair ranges (adjusted for coarser lattice_width=40)
    roi_range = (0, 10)       # Shrink hole (inner region)
    pair_range = (25, 35)     # Expand background (outer region)
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
    energy_with_seams = visualize_seams_on_world_energy(energy, lattice, roi_seam, pair_seam)
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
    print("DEBUGGING: River Lattice Carving with Curved Lattice")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create river
    print("1. Creating river image...")
    image = create_river(size=512, device=device)
    H, W = image.shape
    print(f"   Image size: {H}×{W}")

    # River centerline function (matches the sinusoidal curve in create_river)
    river_center_y = H / 2
    amplitude = 0.3 * H / 2  # 0.3 from create_river, scaled to pixels
    frequency = 3.0  # from create_river

    def river_centerline(x):
        """River centerline: y = center + amplitude * sin(frequency * x_norm)"""
        # x is in pixel coordinates [0, W]
        # Normalize to [-1, 1] range for sin function
        x_norm = 2 * x / W - 1
        return river_center_y + amplitude * np.sin(frequency * x_norm)

    # For river, use curved lattice from centerline points (Figure 9 approach)
    print("\n2. Creating lattice from river centerline points (Figure 9)...")
    n_samples = 100
    x_samples = torch.linspace(0, W, n_samples, device=device, dtype=torch.float32)
    y_samples = torch.tensor([river_centerline(x.item()) for x in x_samples], device=device, dtype=torch.float32)
    curve_points = torch.stack([x_samples, y_samples], dim=1)

    perp_extent = H / 3  # Extend perpendicular to river
    n_lines = 24  # COARSE lattice

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=n_lines,
        perp_extent=perp_extent,
        device=device
    )
    print(f"   Sampled {n_samples} points along river centerline")
    print(f"   Number of scanlines: {lattice.n_lines} (coarse, like paper)")
    print(f"   Perpendicular extent: ±{perp_extent:.1f} pixels from centerline")

    # Define ROI and pair ranges for seam pairs
    # ROI: the river itself (middle scanlines)
    # Pair: background regions (top and bottom scanlines)
    roi_start = int(n_lines * 0.4)  # Start at 40% of scanlines
    roi_end = int(n_lines * 0.6)    # End at 60% (middle 20%)
    pair_start = int(n_lines * 0.05)  # Top background
    pair_end = int(n_lines * 0.25)

    print(f"\n3. Seam pair regions (in lattice u-coordinates):")
    print(f"   ROI (river): scanlines [{roi_start}, {roi_end}]")
    print(f"   Pair (background): scanlines [{pair_start}, {pair_end}]")

    # Compute energy
    print("\n4. Computing energy...")
    energy = gradient_magnitude_energy(image)

    # Resample energy to lattice space
    print("5. Resampling energy to lattice space...")
    lattice_width = 40  # Coarse resolution
    energy_3d = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
    lattice_energy = lattice_energy.squeeze(0)
    print(f"   Lattice energy shape: {lattice_energy.shape}")

    # Find seams in windowed regions
    print("\n6. Finding seam pairs...")
    roi_seam = greedy_seam_windowed(lattice_energy, (roi_start, roi_end), direction='vertical')
    pair_seam = greedy_seam_windowed(lattice_energy, (pair_start, pair_end), direction='vertical')
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
    axes[0, 0].set_title('Original River Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Lattice overlay
    img_with_lattice = visualize_lattice_on_image(image, lattice, n_samples=50)
    axes[0, 1].imshow(img_with_lattice.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Curved Lattice Structure\n(Red = scanlines perpendicular to river)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Energy map with seam pairs overlayed
    print("   Creating world-space seam visualization...")
    energy_with_seams = visualize_seams_on_world_energy(energy, lattice, roi_seam, pair_seam)
    axes[0, 2].imshow(energy_with_seams.permute(1, 2, 0).cpu().numpy())
    axes[0, 2].set_title('Energy + Seam Pairs (World Space)\nYellow=ROI (river), Green=Pair (background)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2, Col 1: Lattice energy
    axes[1, 0].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    axes[1, 0].set_title(f'Energy in Lattice Space\n({lattice_energy.shape[0]} scanlines × {lattice_energy.shape[1]} samples)', fontsize=12)
    axes[1, 0].set_xlabel('u (along scanline)', fontsize=10)
    axes[1, 0].set_ylabel('n (scanline index)', fontsize=10)

    # Row 2, Col 2: Lattice energy with ROI seam
    axes[1, 1].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    scanline_indices = np.arange(len(roi_seam))
    axes[1, 1].plot(roi_seam.cpu().numpy(), scanline_indices, 'yellow', linewidth=2, label='ROI seam')
    axes[1, 1].axhline(roi_start, color='yellow', linestyle='--', alpha=0.5, label='ROI window')
    axes[1, 1].axhline(roi_end, color='yellow', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Lattice Energy + ROI Seam\n(Yellow = remove from river)', fontsize=12)
    axes[1, 1].set_xlabel('u (along scanline)', fontsize=10)
    axes[1, 1].set_ylabel('n (scanline index)', fontsize=10)
    axes[1, 1].legend(fontsize=8)

    # Row 2, Col 3: Lattice energy with both seams
    axes[1, 2].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    axes[1, 2].plot(roi_seam.cpu().numpy(), scanline_indices, 'yellow', linewidth=2, label='ROI seam (remove)')
    axes[1, 2].plot(pair_seam.cpu().numpy(), scanline_indices, 'lime', linewidth=2, label='Pair seam (insert)')
    axes[1, 2].axhline(roi_start, color='yellow', linestyle='--', alpha=0.3)
    axes[1, 2].axhline(roi_end, color='yellow', linestyle='--', alpha=0.3)
    axes[1, 2].axhline(pair_start, color='lime', linestyle='--', alpha=0.3)
    axes[1, 2].axhline(pair_end, color='lime', linestyle='--', alpha=0.3)
    axes[1, 2].set_title('Both Seam Pairs\n(Yellow=shrink river, Lime=expand background)', fontsize=12)
    axes[1, 2].set_xlabel('u (along scanline)', fontsize=10)
    axes[1, 2].set_ylabel('n (scanline index)', fontsize=10)
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('../output/debug_river_setup.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/debug_river_setup.png")

    print("\n" + "="*70)
    print("✓ Debug visualization complete!")
    print("="*70)
    print("\nCheck ../output/debug_river_setup.png")
    print("\nThis shows:")
    print("  - Original river image")
    print("  - Curved lattice structure following river path")
    print("  - Energy map with seam pairs overlayed (world space)")
    print("  - Energy in lattice space with seam pairs marked")


def debug_arch():
    """Debug arch carving (Figure 3 from paper)."""
    print("="*70)
    print("DEBUGGING: Arch (Figure 3 from Paper)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create arch
    print("1. Creating arch image...")
    image = create_arch(size=512, device=device)
    H, W = image.shape
    print(f"   Image size: {H}×{W}")

    # Arch parameters (match create_arch)
    center_x = W / 2
    base_y = H * 0.75
    radius = W * 0.35

    # Arch centerline follows a semicircle
    def arch_centerline(x):
        """Arch centerline: semicircle centered at (center_x, base_y)"""
        # For x in [center_x - radius, center_x + radius]
        # y = base_y - sqrt(radius^2 - (x - center_x)^2)
        dx = x - center_x
        if abs(dx) > radius:
            return base_y  # Outside arch
        return base_y - np.sqrt(max(0, radius**2 - dx**2))

    print("\n2. Auto-detecting ROI from energy...")
    energy_temp = gradient_magnitude_energy(image)
    # ROI is where energy is high (arch edges)
    energy_threshold = energy_temp.mean() + energy_temp.std()
    roi_mask = energy_temp > energy_threshold

    # Get bounding box of ROI
    roi_y, roi_x = torch.where(roi_mask)
    if len(roi_y) > 0:
        roi_x_min, roi_x_max = roi_x.min().item(), roi_x.max().item()
        roi_y_min, roi_y_max = roi_y.min().item(), roi_y.max().item()
        print(f"   ROI bounding box: x=[{roi_x_min:.0f}, {roi_x_max:.0f}], y=[{roi_y_min:.0f}, {roi_y_max:.0f}]")
    else:
        # Fallback to arch parameters
        roi_x_min, roi_x_max = center_x - radius, center_x + radius
        roi_y_min, roi_y_max = base_y - radius, base_y
        print(f"   Using fallback ROI from arch parameters")

    print("\n3. Creating lattice from arch centerline points (Figure 9 approach)...")
    # Sample points along the arch centerline
    x_min = roi_x_min - 20  # Small padding
    x_max = roi_x_max + 20
    n_samples = 100
    x_samples = torch.linspace(x_min, x_max, n_samples, device=device, dtype=torch.float32)
    y_samples = torch.tensor([arch_centerline(x.item()) for x in x_samples], device=device, dtype=torch.float32)
    curve_points = torch.stack([x_samples, y_samples], dim=1)

    perp_extent = (roi_y_max - roi_y_min) / 2 + 20  # Cover ROI height + padding
    n_lines = 24  # COARSE lattice like in paper

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=n_lines,
        perp_extent=perp_extent,
        device=device
    )
    print(f"   Sampled {n_samples} points along arch centerline")
    print(f"   Number of scanlines: {lattice.n_lines} (coarse, like paper)")
    print(f"   Perpendicular extent: ±{perp_extent:.1f} pixels from arch")

    # Compute energy
    print("\n4. Computing energy...")
    energy = gradient_magnitude_energy(image)

    # Resample energy to lattice space
    print("5. Resampling energy to lattice space...")
    lattice_width = 40  # Coarse resolution matching scanline count
    energy_3d = energy.unsqueeze(0)
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
    lattice_energy = lattice_energy.squeeze(0)
    print(f"   Lattice energy shape: {lattice_energy.shape}")

    # Find one seam for traditional comparison
    from src.seam import greedy_seam
    print("\n6. Finding vertical seam in lattice space...")
    seam = greedy_seam(lattice_energy, direction='vertical')
    print(f"   Seam shape: {seam.shape}")
    print(f"   Seam range: [{seam.min().item():.0f}, {seam.max().item():.0f}]")

    # Compute forward mapping for visualization
    print("\n7. Computing forward mapping for visualization...")
    from src.carving import _precompute_forward_mapping
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)

    # Create visualizations
    print("8. Creating visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1, Col 1: Original image
    axes[0, 0].imshow(image.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title('Original Arch (Figure 3)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Row 1, Col 2: Lattice overlay
    img_with_lattice = visualize_lattice_on_image(image, lattice, n_samples=40)
    axes[0, 1].imshow(img_with_lattice.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Curved Lattice Structure\n(Red = scanlines perpendicular to arch)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Row 1, Col 3: Energy map with seam overlayed
    print("   Creating world-space seam visualization...")
    # Convert seam from lattice space to world space
    seam_lattice_pts = torch.stack([
        seam.float(),
        torch.arange(n_lines, dtype=torch.float32, device=device)
    ], dim=1)
    seam_world_pts = lattice.inverse_mapping(seam_lattice_pts)

    # Create RGB version of energy
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    energy_rgb = torch.stack([energy_norm, energy_norm, energy_norm], dim=0)

    # Draw seam on energy map (cyan)
    for i in range(n_lines):
        x, y = seam_world_pts[i]
        xi, yi = int(round(x.item())), int(round(y.item()))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx*dx + dy*dy <= 4:
                    yy, xx = yi + dy, xi + dx
                    if 0 <= xx < W and 0 <= yy < H:
                        energy_rgb[0, yy, xx] = 0.0
                        energy_rgb[1, yy, xx] = 1.0  # Cyan
                        energy_rgb[2, yy, xx] = 1.0

    axes[0, 2].imshow(energy_rgb.permute(1, 2, 0).cpu().numpy())
    axes[0, 2].set_title('Energy + Seam (World Space)\nCyan = vertical seam in lattice space', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2, Col 1: Lattice energy
    axes[1, 0].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    axes[1, 0].set_title(f'Energy in Lattice Space\n({lattice_energy.shape[0]} scanlines × {lattice_energy.shape[1]} samples)', fontsize=12)
    axes[1, 0].set_xlabel('u (along scanline)', fontsize=10)
    axes[1, 0].set_ylabel('n (scanline index)', fontsize=10)

    # Row 2, Col 2: Lattice energy with seam
    axes[1, 1].imshow(lattice_energy.cpu().numpy(), cmap='hot')
    scanline_indices = np.arange(len(seam))
    axes[1, 1].plot(seam.cpu().numpy(), scanline_indices, 'cyan', linewidth=2, label='Vertical seam')
    axes[1, 1].set_title('Lattice Energy + Seam\n(Vertical seam in lattice space)', fontsize=12)
    axes[1, 1].set_xlabel('u (along scanline)', fontsize=10)
    axes[1, 1].set_ylabel('n (scanline index)', fontsize=10)
    axes[1, 1].legend(fontsize=8)

    # Row 2, Col 3: Apply traditional and lattice-guided carving
    print("   Running traditional carving for comparison...")
    from src.carving import carve_image_traditional, carve_image_lattice_guided

    n_seams_to_remove = 10  # Fewer seams for coarse lattice
    traditional_carved = carve_image_traditional(image, n_seams=n_seams_to_remove, direction='vertical')
    print(f"   Traditional carved: {traditional_carved.shape}")

    print("   Running lattice-guided carving...")
    lattice_carved = carve_image_lattice_guided(
        image, lattice, n_seams=n_seams_to_remove, direction='vertical', lattice_width=lattice_width
    )
    print(f"   Lattice-guided carved: {lattice_carved.shape}")

    # Show comparison
    axes[1, 2].imshow(traditional_carved.cpu().numpy(), cmap='gray')
    axes[1, 2].set_title(f'Traditional Carved (50 seams)\nArch distorted', fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../output/debug_arch_setup.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/debug_arch_setup.png")

    # Create a separate comparison figure showing all three
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    axes2[0].imshow(image.cpu().numpy(), cmap='gray')
    axes2[0].set_title('Original Arch', fontsize=14, fontweight='bold')
    axes2[0].axis('off')

    axes2[1].imshow(traditional_carved.cpu().numpy(), cmap='gray')
    axes2[1].set_title(f'Traditional Seam Carving ({n_seams_to_remove} seams)\n({traditional_carved.shape[1]}×{traditional_carved.shape[0]})\nArch Distorted', fontsize=14)
    axes2[1].axis('off')

    axes2[2].imshow(lattice_carved.cpu().numpy(), cmap='gray')
    axes2[2].set_title(f'Lattice-Guided Carving ({n_seams_to_remove} seams)\n({lattice_carved.shape[1]}×{lattice_carved.shape[0]})\nArch Preserved(?)', fontsize=14)
    axes2[2].axis('off')

    plt.tight_layout()
    plt.savefig('../output/arch_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/arch_comparison.png")

    print("\n" + "="*70)
    print("✓ Debug visualization complete!")
    print("="*70)
    print("\nCheck ../output/debug_arch_setup.png and arch_comparison.png")
    print("\nCompare with Figure 3 from the paper:")
    print("  - Traditional: arch should be squished horizontally")
    print("  - Lattice-guided: arch should maintain its shape")


def main():
    create_output_dir()

    print("\nWhich test would you like to debug?")
    print("1. Bagel (seam pairs)")
    print("2. River (curved lattice)")
    print("3. Arch (Figure 3)")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice == '1':
        debug_bagel()
    elif choice == '2':
        debug_river()
    elif choice == '3':
        debug_arch()
    else:
        print("Running all three...")
        debug_bagel()
        print("\n\n")
        debug_river()
        print("\n\n")
        debug_arch()


if __name__ == '__main__':
    # Just run all three
    create_output_dir()
    debug_bagel()
    print("\n\n")
    debug_river()
    print("\n\n")
    debug_arch()
