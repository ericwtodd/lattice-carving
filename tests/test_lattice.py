"""
Tests for lattice construction, mappings, and resampling.

Organized into:
  1. Lattice construction basics
  2. Mapping correctness (exact values, not just round-trips)
  3. Resampling correctness (geometric meaning, not just shapes)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import pytest
from src.lattice import Lattice2D
from src.energy import gradient_magnitude_energy
from src.seam import greedy_seam

from conftest import make_gradient_image, make_ring_image


# ---------------------------------------------------------------------------
# 1. Lattice construction
# ---------------------------------------------------------------------------

class TestLatticeConstruction:
    def test_rectangular_tangents_horizontal_normals_vertical(self):
        lat = Lattice2D.rectangular(5, 8)
        for n in range(5):
            assert torch.allclose(lat.tangents[n], torch.tensor([1.0, 0.0]))
            assert torch.allclose(lat.normals[n], torch.tensor([0.0, 1.0]))

    def test_rectangular_origins_are_rows(self):
        lat = Lattice2D.rectangular(4, 10)
        for n in range(4):
            assert lat.origins[n, 0].item() == 0.0   # x = 0
            assert lat.origins[n, 1].item() == float(n)  # y = row index

    def test_circular_tangents_are_unit(self):
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=36)
        norms = torch.norm(lat.tangents, dim=1)
        assert torch.allclose(norms, torch.ones(36), atol=1e-6)

    def test_circular_tangents_match_angles(self):
        n_lines = 8
        lat = Lattice2D.circular((0.0, 0.0), 10.0, n_lines=n_lines)
        for i in range(n_lines):
            angle = 2 * math.pi * i / n_lines
            expected = torch.tensor([math.cos(angle), math.sin(angle)])
            assert torch.allclose(lat.tangents[i], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Mapping correctness — exact known values
# ---------------------------------------------------------------------------

class TestRectangularMappingExact:
    """Rectangular lattice is the identity mapping: (u, n) == (x, y)."""

    def test_inverse_mapping_is_identity(self):
        lat = Lattice2D.rectangular(10, 10)
        pts = torch.tensor([[3.0, 7.0], [0.0, 0.0], [9.0, 9.0], [4.5, 2.5]])
        world = lat.inverse_mapping(pts)
        assert torch.allclose(world, pts, atol=1e-5)

    def test_forward_mapping_is_identity(self):
        lat = Lattice2D.rectangular(10, 10)
        pts = torch.tensor([[3.0, 7.0], [0.0, 0.0], [9.0, 9.0]])
        lattice = lat.forward_mapping(pts)
        assert torch.allclose(lattice[:, 0], pts[:, 0], atol=1e-5)
        assert torch.allclose(lattice[:, 1], pts[:, 1], atol=0.01)

    def test_roundtrip_is_exact(self):
        """forward(inverse(p)) == p for integer-coordinate lattice points."""
        lat = Lattice2D.rectangular(10, 10)
        pts = torch.tensor([[5.0, 3.0], [0.0, 0.0], [9.0, 8.0]])
        recovered = lat.forward_mapping(lat.inverse_mapping(pts))
        assert torch.allclose(recovered[:, 0], pts[:, 0], atol=1e-5)
        assert torch.allclose(recovered[:, 1], pts[:, 1], atol=0.01)


class TestCircularMappingExact:
    """Circular lattice: u = radial distance, n = angle index."""

    def test_center_maps_to_zero_u(self):
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=36)
        pts = torch.tensor([[0.0, 0.0], [0.0, 9.0], [0.0, 18.0]])
        world = lat.inverse_mapping(pts)
        assert torch.allclose(world, torch.tensor([[cx, cy]] * 3), atol=1e-4)

    def test_known_radial_points(self):
        """Scanline 0 points along +x, scanline at 90deg along +y, etc."""
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=72)

        cases = [
            (10.0, 0.0, cx + 10.0, cy),
            (10.0, 18.0, cx, cy + 10.0),
            (10.0, 36.0, cx - 10.0, cy),
            (10.0, 54.0, cx, cy - 10.0),
        ]
        for u, n, ex, ey in cases:
            world = lat.inverse_mapping(torch.tensor([[u, n]]))
            assert abs(world[0, 0].item() - ex) < 0.1, f"u={u},n={n}: x={world[0,0]:.2f} expected {ex}"
            assert abs(world[0, 1].item() - ey) < 0.1, f"u={u},n={n}: y={world[0,1]:.2f} expected {ey}"

    def test_forward_recovers_correct_scanline(self):
        """Points on specific radial lines should map back to the right scanline."""
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=72)

        pts = torch.tensor([[60.0, 50.0]])
        result = lat.forward_mapping(pts)
        assert abs(result[0, 0].item() - 10.0) < 0.5
        assert abs(result[0, 1].item() - 0.0) < 1.0

        pts = torch.tensor([[50.0, 60.0]])
        result = lat.forward_mapping(pts)
        assert abs(result[0, 0].item() - 10.0) < 0.5
        assert abs(result[0, 1].item() - 18.0) < 1.0

    def test_roundtrip_tight_tolerance(self):
        """inverse -> forward should recover u and n within angular discretization."""
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=72)
        pts = torch.tensor([
            [20.0, 0.0],
            [30.0, 18.0],
            [15.0, 36.0],
            [10.0, 54.0],
        ])
        world = lat.inverse_mapping(pts)
        recovered = lat.forward_mapping(world)
        assert torch.allclose(recovered[:, 0], pts[:, 0], atol=0.5)
        assert torch.allclose(recovered[:, 1], pts[:, 1], atol=1.0)


# ---------------------------------------------------------------------------
# 3. Resampling correctness
# ---------------------------------------------------------------------------

class TestResampleToLatticeSpace:
    def test_rectangular_is_identity(self):
        """Rectangular lattice resampling should reproduce the image exactly."""
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)
        resampled = lat.resample_to_lattice_space(image, lattice_width=W)
        assert torch.allclose(resampled, image, atol=0.01)

    def test_circular_unrolls_ring_to_vertical_stripe(self):
        """A ring in world space should become a vertical stripe in lattice space."""
        H, W = 64, 64
        cx, cy = 32.0, 32.0
        ring_radius = 15.0
        ring_width = 4.0

        image = make_ring_image(H, W, cx, cy,
                                ring_radius - ring_width / 2,
                                ring_radius + ring_width / 2)

        lat = Lattice2D.circular((cx, cy), 30.0, n_lines=72)
        lattice_img = lat.resample_to_lattice_space(image, lattice_width=30)

        ring_col = lattice_img[0, :, int(ring_radius)]
        assert ring_col.mean() > 0.5, f"Ring column mean={ring_col.mean():.3f}, expected >0.5"

        bg_col = lattice_img[0, :, 5]
        assert bg_col.mean() < 0.2, f"Background column mean={bg_col.mean():.3f}, expected <0.2"

        assert ring_col.std() < 0.3, f"Ring column std={ring_col.std():.3f}, expected <0.3"

    def test_rectangular_roundtrip_preserves_image(self):
        """Resample to lattice space and back should approximate original."""
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)
        lattice_img = lat.resample_to_lattice_space(image, lattice_width=W)
        recovered = lat.resample_from_lattice_space(lattice_img, H, W)
        assert torch.allclose(recovered, image, atol=0.05)

    def test_grayscale_preserves_dims(self):
        image = make_gradient_image(16, 24, channels=0)
        lat = Lattice2D.rectangular(16, 24)
        resampled = lat.resample_to_lattice_space(image, lattice_width=24)
        assert resampled.dim() == 2
        assert resampled.shape == (16, 24)


# ---------------------------------------------------------------------------
# 4. Lattice-guided seam equivalence
# ---------------------------------------------------------------------------

class TestLatticeGuidedSeamEquivalence:
    def test_rectangular_lattice_produces_same_seams_as_traditional(self):
        """Since rectangular lattice = identity, carving in lattice space
        should find the exact same seam as traditional carving."""
        H, W = 20, 30
        image = make_gradient_image(H, W, channels=0)
        lat = Lattice2D.rectangular(H, W)

        lattice_img = lat.resample_to_lattice_space(image, lattice_width=W)

        energy_lattice = gradient_magnitude_energy(lattice_img)
        energy_direct = gradient_magnitude_energy(image)

        seam_lattice = greedy_seam(energy_lattice, direction='vertical')
        seam_direct = greedy_seam(energy_direct, direction='vertical')

        assert torch.equal(seam_lattice, seam_direct), \
            f"Lattice seam {seam_lattice.tolist()} != direct seam {seam_direct.tolist()}"
