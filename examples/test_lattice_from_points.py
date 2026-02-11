"""
Simple test: Build a lattice from a curve (Figure 9 approach).

Just verify lattice construction works - no carving, no energy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.lattice import Lattice2D


def create_sine_curve(n_points=50, amplitude=50, frequency=2, device='cpu'):
    """Create a simple sine wave curve."""
    x = torch.linspace(0, 400, n_points, dtype=torch.float32, device=device)
    y = 200 + amplitude * torch.sin(frequency * np.pi * x / 400)
    return torch.stack([x, y], dim=1)


def create_arc_curve(n_points=50, center=(200, 300), radius=100, device='cpu'):
    """Create a semicircular arc."""
    angles = torch.linspace(0, np.pi, n_points, dtype=torch.float32, device=device)
    x = center[0] + radius * torch.cos(angles)
    y = center[1] - radius * torch.sin(angles)  # Flip y
    return torch.stack([x, y], dim=1)


def visualize_lattice(curve_points, lattice, title="Lattice Construction"):
    """
    Visualize the curve and the lattice built from it.

    Shows:
    - Centerline curve points (blue)
    - Lattice scanlines (red)
    - Lattice structure in world space
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Curve and scanlines
    ax = axes[0]

    # Plot centerline curve
    curve_np = curve_points.cpu().numpy()
    ax.plot(curve_np[:, 0], curve_np[:, 1], 'b-', linewidth=2, label='Centerline')
    ax.scatter(curve_np[:, 0], curve_np[:, 1], c='blue', s=20, zorder=5)

    # Plot lattice scanlines (perpendicular to curve)
    n_lines = lattice.n_lines
    device = curve_points.device

    # For each scanline, draw a line segment
    u_range = 100  # Length of scanline to draw

    for i in range(n_lines):
        # Scanline origin and tangent
        origin = lattice.origins[i].cpu().numpy()
        tangent = lattice.tangents[i].cpu().numpy()

        # Draw scanline (perpendicular segment)
        p1 = origin - u_range * tangent
        p2 = origin + u_range * tangent

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', alpha=0.5, linewidth=1)

        # Mark origin
        ax.scatter([origin[0]], [origin[1]], c='red', s=30, marker='x', zorder=5)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Image coordinates (y down)
    ax.set_title(f'{title}\nCenterline (blue) + Scanlines (red)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Lattice grid in world space
    ax = axes[1]

    # Draw grid by sampling lattice space and mapping to world space
    # lattice_width should be 2*perp_extent (full extent of scanlines)
    if hasattr(lattice, '_perp_extent'):
        lattice_width = int(2 * lattice._perp_extent)
    else:
        lattice_width = 50  # Fallback

    print(f"   Visualizing with lattice_width={lattice_width} (should be 2*perp_extent)")

    # Draw scanlines (constant n, varying u)
    for n_idx in range(0, n_lines, max(1, n_lines // 10)):  # Sample every 10th
        u_vals = torch.linspace(0, lattice_width, 50, dtype=torch.float32, device=device)
        n_vals = torch.full_like(u_vals, float(n_idx))

        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)

        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'r-', alpha=0.7, linewidth=1)

    # Draw cross-scanlines (constant u, varying n)
    for u_idx in range(0, lattice_width, max(1, lattice_width // 10)):
        n_vals = torch.linspace(0, n_lines - 1, 50, dtype=torch.float32, device=device)
        u_vals = torch.full_like(n_vals, float(u_idx))

        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)

        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'g-', alpha=0.7, linewidth=1)

    # Overlay centerline
    ax.plot(curve_np[:, 0], curve_np[:, 1], 'b-', linewidth=2, label='Centerline')

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Lattice Grid in World Space\nRed=scanlines, Green=cross-lines', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_sine_wave():
    """Test lattice construction on a sine wave."""
    print("="*70)
    print("TEST 1: Sine Wave Lattice")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create sine wave curve
    print("1. Creating sine wave curve...")
    curve_points = create_sine_curve(n_points=50, amplitude=50, frequency=2, device=device)
    print(f"   Curve points: {curve_points.shape}")

    # Build lattice
    print("2. Building lattice from curve points...")
    n_lines = 12  # Number of perpendicular scanlines
    perp_extent = 80  # Distance perpendicular to curve

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=n_lines,
        perp_extent=perp_extent,
        device=device
    )
    print(f"   Lattice scanlines: {lattice.n_lines}")
    print(f"   Perpendicular extent: ±{perp_extent} pixels")
    if hasattr(lattice, '_recommended_lattice_width'):
        print(f"   Recommended lattice_width for square cells: {lattice._recommended_lattice_width}")
        print(f"   Scanline spacing: {lattice.spacing[0].item():.2f} pixels")

    # Visualize
    print("3. Creating visualization...")
    fig = visualize_lattice(curve_points, lattice, title="Sine Wave Lattice")

    os.makedirs('../output', exist_ok=True)
    fig.savefig('../output/test_sine_lattice.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/test_sine_lattice.png")

    print("\n✓ Sine wave test complete!")


def test_arc():
    """Test lattice construction on a circular arc."""
    print("\n" + "="*70)
    print("TEST 2: Circular Arc Lattice")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create arc curve
    print("1. Creating circular arc curve...")
    curve_points = create_arc_curve(n_points=50, center=(200, 300), radius=100, device=device)
    print(f"   Curve points: {curve_points.shape}")

    # Build lattice
    print("2. Building lattice from curve points...")
    n_lines = 12
    perp_extent = 60

    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=n_lines,
        perp_extent=perp_extent,
        device=device
    )
    print(f"   Lattice scanlines: {lattice.n_lines}")
    print(f"   Perpendicular extent: ±{perp_extent} pixels")
    if hasattr(lattice, '_recommended_lattice_width'):
        print(f"   Recommended lattice_width for square cells: {lattice._recommended_lattice_width}")
        print(f"   Scanline spacing: {lattice.spacing[0].item():.2f} pixels")

    # Visualize
    print("3. Creating visualization...")
    fig = visualize_lattice(curve_points, lattice, title="Circular Arc Lattice")

    fig.savefig('../output/test_arc_lattice.png', dpi=150, bbox_inches='tight')
    print("   Saved: ../output/test_arc_lattice.png")

    print("\n✓ Arc test complete!")


def main():
    """Run all basic lattice tests."""
    print("BASIC LATTICE CONSTRUCTION TEST")
    print("Testing Figure 9 approach: build lattice from curve points\n")

    test_sine_wave()
    test_arc()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\nCheck ../output/ for visualizations:")
    print("  - test_sine_lattice.png")
    print("  - test_arc_lattice.png")
    print("\nThese show:")
    print("  Left panel: Centerline curve + perpendicular scanlines")
    print("  Right panel: Lattice grid structure in world space")


if __name__ == '__main__':
    main()
