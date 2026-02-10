"""
Tests for lattice mappings, resampling, and end-to-end carving.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.lattice import Lattice2D
from src.energy import gradient_magnitude_energy
from src.seam import greedy_seam, remove_seam
from src.carving import carve_image_traditional, carve_image_lattice_guided


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gradient_image(H, W, channels=3):
    """Create a horizontal gradient image (dark left, bright right)."""
    grad = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    if channels > 0:
        return grad.unsqueeze(0).expand(channels, H, W).clone()
    return grad


def make_stripe_image(H, W, stripe_width=10, channels=3):
    """Create an image with vertical stripes of alternating intensity."""
    img = torch.zeros(H, W)
    for i in range(0, W, stripe_width * 2):
        img[:, i:i + stripe_width] = 1.0
    if channels > 0:
        return img.unsqueeze(0).expand(channels, H, W).clone()
    return img


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------

class TestLatticeConstruction:
    def test_rectangular_shapes(self):
        lat = Lattice2D.rectangular(10, 20)
        assert lat.n_lines == 10
        assert lat.origins.shape == (10, 2)
        assert lat.tangents.shape == (10, 2)
        assert lat.normals.shape == (10, 2)

    def test_rectangular_tangents_are_horizontal(self):
        lat = Lattice2D.rectangular(5, 5)
        # tangent should be (1, 0) for every scanline
        expected = torch.zeros(5, 2)
        expected[:, 0] = 1.0
        assert torch.allclose(lat.tangents, expected)

    def test_rectangular_normals_are_vertical(self):
        lat = Lattice2D.rectangular(5, 5)
        # normal should be (0, 1) for every scanline
        expected = torch.zeros(5, 2)
        expected[:, 1] = 1.0
        assert torch.allclose(lat.normals, expected)

    def test_circular_shapes(self):
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=36)
        assert lat.n_lines == 36
        assert lat.origins.shape == (36, 2)

    def test_circular_tangents_are_unit(self):
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=36)
        norms = torch.norm(lat.tangents, dim=1)
        assert torch.allclose(norms, torch.ones(36), atol=1e-5)


# ---------------------------------------------------------------------------
# Mapping round-trips
# ---------------------------------------------------------------------------

class TestRectangularMapping:
    """For a rectangular lattice, forward(inverse(p)) ≈ p and vice-versa."""

    def test_inverse_then_forward_roundtrip(self):
        H, W = 20, 30
        lat = Lattice2D.rectangular(H, W)

        # Sample some lattice-space points: (u, n)
        lattice_pts = torch.tensor([
            [0.0, 0.0],
            [5.0, 3.0],
            [15.0, 10.0],
            [29.0, 19.0],
        ])

        world_pts = lat.inverse_mapping(lattice_pts)
        recovered = lat.forward_mapping(world_pts)

        # u should be recovered exactly; n may have small fractional error
        assert torch.allclose(recovered[:, 0], lattice_pts[:, 0], atol=0.5)
        assert torch.allclose(recovered[:, 1], lattice_pts[:, 1], atol=0.5)

    def test_forward_then_inverse_roundtrip(self):
        H, W = 20, 30
        lat = Lattice2D.rectangular(H, W)

        # Points in world space (x, y)
        world_pts = torch.tensor([
            [0.0, 0.0],
            [10.0, 5.0],
            [29.0, 19.0],
        ])

        lattice_pts = lat.forward_mapping(world_pts)
        recovered = lat.inverse_mapping(lattice_pts)

        assert torch.allclose(recovered, world_pts, atol=0.5)

    def test_rectangular_mapping_is_identity(self):
        """For a rectangular lattice, (u, n) should map to (x, y) = (u, n)."""
        H, W = 10, 10
        lat = Lattice2D.rectangular(H, W)

        lattice_pts = torch.tensor([
            [3.0, 7.0],
            [0.0, 0.0],
            [9.0, 9.0],
        ])

        world_pts = lat.inverse_mapping(lattice_pts)
        # For rectangular: world (x, y) should equal lattice (u, n)
        assert torch.allclose(world_pts, lattice_pts, atol=1e-5)


class TestCircularMapping:
    def test_inverse_maps_center_correctly(self):
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=36)

        # u=0 on any scanline should map to the center
        lattice_pts = torch.tensor([[0.0, 0.0], [0.0, 18.0]])
        world_pts = lat.inverse_mapping(lattice_pts)

        assert torch.allclose(world_pts[:, 0], torch.tensor([cx, cx]), atol=1e-4)
        assert torch.allclose(world_pts[:, 1], torch.tensor([cy, cy]), atol=1e-4)

    def test_inverse_maps_radial_distance(self):
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=36)

        # u=10 on scanline 0 (angle=0, tangent=(1,0)) should map to (cx+10, cy)
        lattice_pts = torch.tensor([[10.0, 0.0]])
        world_pts = lat.inverse_mapping(lattice_pts)

        assert torch.allclose(world_pts, torch.tensor([[cx + 10.0, cy]]), atol=1e-4)

    def test_inverse_forward_roundtrip(self):
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=72)

        # Points at various radii and angles (u, n)
        lattice_pts = torch.tensor([
            [20.0, 0.0],
            [30.0, 18.0],
            [10.0, 36.0],
        ])

        world_pts = lat.inverse_mapping(lattice_pts)
        recovered = lat.forward_mapping(world_pts)

        assert torch.allclose(recovered[:, 0], lattice_pts[:, 0], atol=1.0)
        assert torch.allclose(recovered[:, 1], lattice_pts[:, 1], atol=1.0)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

class TestResampleToLatticeSpace:
    def test_rectangular_identity(self):
        """Resampling with a rectangular lattice should approximate the original image."""
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)

        resampled = lat.resample_to_lattice_space(image, lattice_width=W)

        # Should be very close to the original image
        assert resampled.shape == image.shape
        assert torch.allclose(resampled, image, atol=0.05)

    def test_output_shape(self):
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)

        resampled = lat.resample_to_lattice_space(image, lattice_width=32)
        assert resampled.shape == (3, H, 32)

    def test_grayscale_input(self):
        H, W = 16, 24
        image = make_gradient_image(H, W, channels=0)
        lat = Lattice2D.rectangular(H, W)

        resampled = lat.resample_to_lattice_space(image, lattice_width=W)
        assert resampled.dim() == 2
        assert resampled.shape == (H, W)


class TestResampleFromLatticeSpace:
    def test_rectangular_roundtrip(self):
        """Resample to lattice space and back should approximate the original."""
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)

        lattice_img = lat.resample_to_lattice_space(image, lattice_width=W)
        recovered = lat.resample_from_lattice_space(lattice_img, H, W)

        assert recovered.shape == image.shape
        # Allow some error from double interpolation
        assert torch.allclose(recovered, image, atol=0.1)

    def test_output_shape(self):
        H, W = 16, 24
        lat = Lattice2D.rectangular(H, W)
        lattice_img = torch.rand(3, H, W)

        out = lat.resample_from_lattice_space(lattice_img, 32, 48)
        assert out.shape == (3, 32, 48)


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_gradient_energy_shape(self):
        image = make_gradient_image(16, 24)
        energy = gradient_magnitude_energy(image)
        assert energy.shape == (16, 24)

    def test_flat_image_low_interior_energy(self):
        """Interior of a flat image should have zero energy (borders have Sobel edge artifacts)."""
        image = torch.ones(3, 16, 24) * 0.5
        energy = gradient_magnitude_energy(image)
        # Interior pixels (away from border) should be zero
        assert energy[1:-1, 1:-1].max() < 1e-5

    def test_stripe_image_has_edges(self):
        image = make_stripe_image(16, 40, stripe_width=10)
        energy = gradient_magnitude_energy(image)
        assert energy.max() > 0.1


# ---------------------------------------------------------------------------
# Seam computation
# ---------------------------------------------------------------------------

class TestSeam:
    def test_greedy_seam_shape(self):
        energy = torch.rand(16, 24)
        seam = greedy_seam(energy, direction='vertical')
        assert seam.shape == (16,)
        assert seam.min() >= 0
        assert seam.max() < 24

    def test_greedy_seam_horizontal_shape(self):
        energy = torch.rand(16, 24)
        seam = greedy_seam(energy, direction='horizontal')
        assert seam.shape == (24,)
        assert seam.min() >= 0
        assert seam.max() < 16

    def test_seam_continuity(self):
        """Adjacent seam indices should differ by at most 1."""
        energy = torch.rand(32, 32)
        seam = greedy_seam(energy, direction='vertical')
        diffs = torch.abs(seam[1:] - seam[:-1])
        assert diffs.max() <= 1

    def test_remove_seam_vertical(self):
        image = torch.rand(3, 16, 24)
        seam = greedy_seam(gradient_magnitude_energy(image), direction='vertical')
        carved = remove_seam(image, seam, direction='vertical')
        assert carved.shape == (3, 16, 23)

    def test_remove_seam_horizontal(self):
        image = torch.rand(3, 16, 24)
        seam = greedy_seam(gradient_magnitude_energy(image), direction='horizontal')
        carved = remove_seam(image, seam, direction='horizontal')
        assert carved.shape == (3, 15, 24)


# ---------------------------------------------------------------------------
# End-to-end carving
# ---------------------------------------------------------------------------

class TestTraditionalCarving:
    def test_reduces_width(self):
        image = make_stripe_image(16, 40, stripe_width=10)
        carved = carve_image_traditional(image, n_seams=5, direction='vertical')
        assert carved.shape == (3, 16, 35)

    def test_reduces_height(self):
        image = make_stripe_image(40, 16, stripe_width=10)
        carved = carve_image_traditional(image, n_seams=5, direction='horizontal')
        assert carved.shape == (3, 35, 16)

    def test_preserves_dtype(self):
        image = make_stripe_image(16, 40)
        carved = carve_image_traditional(image, n_seams=2)
        assert carved.dtype == image.dtype


class TestLatticeGuidedCarving:
    def test_rectangular_lattice_output_shape(self):
        """Lattice-guided carving with rectangular lattice returns same H x W."""
        H, W = 16, 40
        image = make_stripe_image(H, W, stripe_width=10)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        # Output shape is always the original H x W (resampled back)
        assert carved.shape == (3, H, W)

    def test_circular_lattice_output_shape(self):
        """Lattice-guided carving with circular lattice returns original H x W."""
        H, W = 64, 64
        image = torch.rand(3, H, W)
        lat = Lattice2D.circular((32.0, 32.0), 30.0, n_lines=36)

        carved = carve_image_lattice_guided(image, lat, n_seams=3, lattice_width=40)
        assert carved.shape == (3, H, W)

    def test_rectangular_lattice_actually_changes_image(self):
        """Carving should actually modify the image content."""
        H, W = 16, 40
        image = make_stripe_image(H, W, stripe_width=10)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        # The carved image should not be identical to the input
        assert not torch.allclose(carved, image, atol=1e-3)
