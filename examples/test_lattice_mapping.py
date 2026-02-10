"""
Test and visualize lattice mapping functions.

This script validates that the forward and inverse mappings work correctly
for both rectangular and circular lattices.
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice2D


def test_rectangular_lattice():
    """Test rectangular lattice mapping (should match identity for grid points)."""
    print("Testing rectangular lattice...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = 10, 15

    # Create rectangular lattice
    lattice = Lattice2D.rectangular(H, W, device=device)

    # Test grid points
    grid_points = []
    for y in range(H):
        for x in range(W):
            grid_points.append([x, y])

    world_points = torch.tensor(grid_points, dtype=torch.float32, device=device)

    # Forward mapping
    lattice_points = lattice.forward_mapping(world_points)

    # Inverse mapping
    reconstructed = lattice.inverse_mapping(lattice_points)

    # Check reconstruction error
    error = torch.mean(torch.abs(world_points - reconstructed))
    print(f"  Reconstruction error: {error.item():.6f}")

    # Check lattice space properties
    print(f"  Lattice points u range: [{lattice_points[:, 0].min():.2f}, {lattice_points[:, 0].max():.2f}]")
    print(f"  Lattice points n range: [{lattice_points[:, 1].min():.2f}, {lattice_points[:, 1].max():.2f}]")

    return error.item() < 0.1  # Should be very small


def test_circular_lattice():
    """Test circular lattice mapping."""
    print("\nTesting circular lattice...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    center = (5.0, 5.0)
    radius = 3.0
    n_lines = 16

    # Create circular lattice
    lattice = Lattice2D.circular(center, radius, n_lines, device=device)

    # Test points on a circle
    angles = torch.linspace(0, 2 * np.pi, 20, device=device)[:-1]
    r = radius * 0.5  # Test at half radius

    world_points = torch.stack([
        center[0] + r * torch.cos(angles),
        center[1] + r * torch.sin(angles)
    ], dim=1)

    # Forward mapping
    lattice_points = lattice.forward_mapping(world_points)

    # Inverse mapping
    reconstructed = lattice.inverse_mapping(lattice_points)

    # Check reconstruction error
    error = torch.mean(torch.abs(world_points - reconstructed))
    print(f"  Reconstruction error: {error.item():.6f}")

    return error.item() < 1.0  # Allow more error due to discretization


def visualize_lattice_structure():
    """Visualize different lattice structures."""
    print("\nVisualizing lattice structures...")

    device = 'cpu'
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Rectangular lattice
    ax = axes[0]
    H, W = 8, 12
    lattice = Lattice2D.rectangular(H, W, device=device)

    # Draw scanlines
    for n in range(lattice.n_lines):
        origin = lattice.origins[n].numpy()
        tangent = lattice.tangents[n].numpy()

        # Draw scanline
        x_start = origin[0]
        y_start = origin[1]
        x_end = origin[0] + W * tangent[0]
        y_end = origin[1] + W * tangent[1]

        ax.plot([x_start, x_end], [y_start, y_end], 'b-', alpha=0.5, linewidth=0.5)

    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-1, H + 1)
    ax.set_aspect('equal')
    ax.set_title('Rectangular Lattice')
    ax.grid(True, alpha=0.3)

    # 2. Circular lattice
    ax = axes[1]
    center = (6.0, 6.0)
    radius = 4.0
    n_lines = 24
    lattice = Lattice2D.circular(center, radius, n_lines, device=device)

    # Draw scanlines (radial lines)
    for n in range(lattice.n_lines):
        origin = lattice.origins[n].numpy()
        tangent = lattice.tangents[n].numpy()

        # Draw radial line
        x_start = origin[0]
        y_start = origin[1]
        x_end = origin[0] + radius * tangent[0]
        y_end = origin[1] + radius * tangent[1]

        ax.plot([x_start, x_end], [y_start, y_end], 'r-', alpha=0.5, linewidth=0.5)

    # Draw circles at different radii
    for r_frac in [0.25, 0.5, 0.75, 1.0]:
        circle = plt.Circle(center, radius * r_frac, fill=False,
                           color='blue', alpha=0.3, linewidth=0.5)
        ax.add_patch(circle)

    ax.set_xlim(center[0] - radius - 1, center[0] + radius + 1)
    ax.set_ylim(center[1] - radius - 1, center[1] + radius + 1)
    ax.set_aspect('equal')
    ax.set_title('Circular Lattice')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/lattice_structures.png', dpi=150)
    print("  Saved: ../output/lattice_structures.png")


def visualize_mapping():
    """Visualize the mapping from world space to lattice index space."""
    print("\nVisualizing world → lattice mapping...")

    device = 'cpu'
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Rectangular lattice
    H, W = 20, 30
    lattice = Lattice2D.rectangular(H, W, device=device)

    # Create grid of world points
    y_coords = torch.linspace(0, H - 1, H, device=device)
    x_coords = torch.linspace(0, W - 1, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    world_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Forward mapping
    lattice_points = lattice.forward_mapping(world_points)

    # Plot world space
    ax = axes[0, 0]
    ax.scatter(world_points[:, 0].numpy(), world_points[:, 1].numpy(),
              c='blue', s=1, alpha=0.5)
    ax.set_title('World Space (Rectangular)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot lattice space
    ax = axes[0, 1]
    ax.scatter(lattice_points[:, 0].numpy(), lattice_points[:, 1].numpy(),
              c='red', s=1, alpha=0.5)
    ax.set_title('Lattice Index Space (Rectangular)')
    ax.set_xlabel('u')
    ax.set_ylabel('n')
    ax.grid(True, alpha=0.3)

    # Circular lattice
    center = (15.0, 10.0)
    radius = 8.0
    n_lines = 32
    lattice = Lattice2D.circular(center, radius, n_lines, device=device)

    # Create points in circular region
    world_points_list = []
    for theta in torch.linspace(0, 2 * np.pi, 40, device=device)[:-1]:
        for r in torch.linspace(0, radius, 20, device=device):
            x = center[0] + r * torch.cos(theta)
            y = center[1] + r * torch.sin(theta)
            world_points_list.append([x, y])

    world_points = torch.tensor(world_points_list, dtype=torch.float32, device=device)

    # Forward mapping
    lattice_points = lattice.forward_mapping(world_points)

    # Plot world space
    ax = axes[1, 0]
    ax.scatter(world_points[:, 0].numpy(), world_points[:, 1].numpy(),
              c='blue', s=1, alpha=0.5)
    ax.set_title('World Space (Circular)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Plot lattice space
    ax = axes[1, 1]
    ax.scatter(lattice_points[:, 0].numpy(), lattice_points[:, 1].numpy(),
              c='red', s=1, alpha=0.5)
    ax.set_title('Lattice Index Space (Circular)')
    ax.set_xlabel('u (radial distance)')
    ax.set_ylabel('n (angle index)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../output/lattice_mapping.png', dpi=150)
    print("  Saved: ../output/lattice_mapping.png")


def main():
    print("=" * 60)
    print("Testing Lattice Mapping Functions")
    print("=" * 60)

    # Run tests
    rect_pass = test_rectangular_lattice()
    circ_pass = test_circular_lattice()

    # Visualizations
    visualize_lattice_structure()
    visualize_mapping()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Rectangular lattice: {'PASS' if rect_pass else 'FAIL'}")
    print(f"  Circular lattice: {'PASS' if circ_pass else 'FAIL'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
