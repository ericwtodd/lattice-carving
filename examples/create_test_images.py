"""
Create and save test images for lattice-space visualization.
Run this once to generate test_bagel.png, test_arch.png, test_river.png

Uses the same image generation code from test_lattice_visualization.py
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path


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


if __name__ == "__main__":
    device = 'cpu'  # Use CPU for reproducibility
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("Creating test images...")

    # Test 1: Bagel with sesame seeds
    print("  - test_bagel.png")
    bagel = create_bagel_image(
        center=(250, 250),
        inner_radius=100,
        outer_radius=200,
        height=500,
        width=500,
        device=device
    )
    bagel_img = Image.fromarray((bagel.cpu().numpy() * 255).astype(np.uint8), mode='L')
    bagel_img.save(output_dir / "test_bagel.png")

    # Test 2: Arch (semicircle) - direct distance-based
    print("  - test_arch.png")
    height, width = 400, 500
    center_x, center_y = width // 2, height - 50
    outer_radius = 150
    inner_radius = 110

    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    dist = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)

    # Arch (only upper half - semicircle)
    arch = torch.full((height, width), 0.2, device=device)
    arch_mask = (dist >= inner_radius) & (dist <= outer_radius) & (yy < center_y)
    arch[arch_mask] = 0.8

    # Add texture
    torch.manual_seed(42)
    arch += (torch.rand(height, width, device=device) - 0.5) * 0.1
    arch = torch.clamp(arch, 0, 1)

    arch_img = Image.fromarray((arch.cpu().numpy() * 255).astype(np.uint8), mode='L')
    arch_img.save(output_dir / "test_arch.png")

    # Test 3: River (sine wave)
    print("  - test_river.png")
    height, width = 400, 500

    # Sine wave curve (like test_lattice_visualization.py)
    x_vals = torch.linspace(0, float(width), 80, dtype=torch.float32, device=device)
    y_vals = height / 2 + 40 * torch.sin(2 * np.pi * x_vals / width)
    river_points = torch.stack([x_vals, y_vals], dim=1)

    river = create_test_image_from_curve(
        curve_points=river_points,
        height=height,
        width=width,
        band_width=50,
        device=device
    )
    river_img = Image.fromarray((river.cpu().numpy() * 255).astype(np.uint8), mode='L')
    river_img.save(output_dir / "test_river.png")

    print(f"\nSaved test images to {output_dir}/")
    print("  test_bagel.png  - Bagel with sesame seeds")
    print("  test_arch.png   - Semicircular arch")
    print("  test_river.png  - Sine wave")
    print("\nThese can now be loaded for lattice-space visualization.")
