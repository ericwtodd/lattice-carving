"""
Test lattice construction with proper visualization.

For each test case (sine, arc, bagel, river, arch), we:
1. Define centerline points
2. Build fine-grained lattice
3. Visualize lattice structure

No carving yet - just verifying lattice construction is solid.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.lattice import Lattice2D


def visualize_lattice_grid(curve_points, lattice, title, filename):
    """Visualize centerline and lattice grid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    curve_np = curve_points.cpu().numpy()
    device = curve_points.device

    # Panel 1: Centerline + scanlines
    ax = axes[0]
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

    for n_idx in range(0, n_lines, grid_spacing_n):
        u_vals = torch.linspace(0, lattice_width, 100, dtype=torch.float32, device=device)
        n_vals = torch.full_like(u_vals, float(n_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'r-', alpha=0.5, linewidth=0.8)

    for u_idx in range(0, lattice_width, grid_spacing_u):
        n_vals = torch.linspace(0, n_lines - 1, 100, dtype=torch.float32, device=device)
        u_vals = torch.full_like(n_vals, float(u_idx))
        lattice_pts = torch.stack([u_vals, n_vals], dim=1)
        world_pts = lattice.inverse_mapping(lattice_pts)
        world_np = world_pts.cpu().numpy()
        ax.plot(world_np[:, 0], world_np[:, 1], 'g-', alpha=0.5, linewidth=0.8)

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

    # Build fine-grained lattice
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=40,  # Fine-grained (was 12)
        perp_extent=60,
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    visualize_lattice_grid(curve_points, lattice, "Sine Wave", "lattice_sine.png")


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

    # Build lattice
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=30,
        perp_extent=50,
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    visualize_lattice_grid(curve_points, lattice, "Arch", "lattice_arch.png")


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

    # Build lattice
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=35,
        perp_extent=80,
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    visualize_lattice_grid(curve_points, lattice, "River", "lattice_river.png")


def test_bagel():
    """Bagel test - circular centerline around the hole."""
    print("\n" + "="*70)
    print("TEST 4: Bagel (Circle)")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Bagel centerline - circle at middle radius between hole and outer edge
    center = (256, 256)
    middle_radius = 80  # Between hole and outer edge
    angles = torch.linspace(0, 2*np.pi, 64, dtype=torch.float32, device=device)[:-1]  # Exclude duplicate
    x = center[0] + middle_radius * torch.cos(angles)
    y = center[1] + middle_radius * torch.sin(angles)
    curve_points = torch.stack([x, y], dim=1)

    # Build lattice (CYCLIC for closed curve)
    lattice = Lattice2D.from_curve_points(
        curve_points=curve_points,
        n_lines=32,
        perp_extent=60,  # Cover hole to outer edge
        cyclic=True,  # Connect last scanline to first (Section 3.5)
        device=device
    )

    print(f"   Scanlines: {lattice.n_lines}")
    print(f"   Cyclic: {lattice._cyclic} (closed curve)")
    print(f"   Recommended lattice_width: {lattice._recommended_lattice_width}")

    visualize_lattice_grid(curve_points, lattice, "Bagel", "lattice_bagel.png")


def main():
    """Run all lattice visualization tests."""
    print("="*70)
    print("LATTICE CONSTRUCTION TESTS")
    print("Verifying lattice structure before adding carving")
    print("="*70)

    test_sine_wave()
    test_arch()
    test_river()
    test_bagel()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETE!")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - lattice_sine.png")
    print("  - lattice_arch.png")
    print("  - lattice_river.png")
    print("  - lattice_bagel.png")
    print("\nEach shows:")
    print("  Left: Centerline + sample scanlines")
    print("  Right: Full lattice grid structure")


if __name__ == '__main__':
    main()
