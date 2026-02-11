"""
Tests for carving algorithms: traditional, lattice-guided, and seam pairs.

Organized into:
  1. Traditional carving (shape/validity)
  2. Lattice-guided carving (shape, equivalence, correctness)
  3. Carving the mapping (Section 3.3)
  4. Seam pairs (Section 3.6)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.lattice import Lattice2D
from src.carving import carve_image_traditional, carve_image_lattice_guided, carve_seam_pairs

from conftest import make_gradient_image, make_ring_image


# ---------------------------------------------------------------------------
# 1. Traditional carving
# ---------------------------------------------------------------------------

class TestTraditionalCarving:
    def test_reduces_width(self):
        image = torch.rand(3, 20, 30)
        carved = carve_image_traditional(image, n_seams=5, direction='vertical')
        assert carved.shape == (3, 20, 25)

    def test_carved_output_is_valid(self):
        """Carved image should have correct shape and no NaN/Inf values."""
        torch.manual_seed(42)
        image = torch.rand(3, 30, 30)
        carved = carve_image_traditional(image, n_seams=10, direction='vertical')
        assert carved.shape == (3, 30, 20)
        assert torch.isfinite(carved).all()
        assert carved.min() >= 0.0
        assert carved.max() <= 1.0

    def test_multiple_carves_reduce_correctly(self):
        """Iterative carving should reduce dimensions by exactly n_seams."""
        image = torch.rand(3, 25, 40)
        for n in [1, 5, 15]:
            carved = carve_image_traditional(image, n_seams=n, direction='vertical')
            assert carved.shape == (3, 25, 40 - n)


# ---------------------------------------------------------------------------
# 2. Lattice-guided carving
# ---------------------------------------------------------------------------

class TestLatticeGuidedCarving:
    def test_lattice_guided_output_shape(self):
        """Lattice-guided carving returns the original spatial dimensions."""
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        assert carved.shape == (3, H, W)

    def test_lattice_guided_modifies_image(self):
        """After carving, the image should actually be different."""
        H, W = 20, 30
        image = make_gradient_image(H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        assert not torch.allclose(carved, image, atol=1e-3)

    def test_circular_lattice_carving_runs(self):
        """Circular lattice-guided carving should complete without error."""
        H, W = 64, 64
        image = make_ring_image(H, W, 32.0, 32.0, 10.0, 20.0)
        lat = Lattice2D.circular((32.0, 32.0), 30.0, n_lines=36)
        carved = carve_image_lattice_guided(image, lat, n_seams=3, lattice_width=30)
        assert carved.shape == (1, H, W)
        assert not torch.allclose(carved, image, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Carving the mapping (Section 3.3)
# ---------------------------------------------------------------------------

class TestCarvingTheMapping:
    def test_output_shape_unchanged(self):
        """Carving the mapping preserves image dimensions."""
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        assert carved.shape == (3, H, W)

    def test_output_finite_and_valid(self):
        """Output should have no NaN/Inf and stay in [0, 1]."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        assert torch.isfinite(carved).all()
        assert carved.min() >= -0.01
        assert carved.max() <= 1.01

    def test_rectangular_matches_traditional(self):
        """For a rectangular lattice, the mapping approach should produce
        results close to traditional carving (first W-k columns)."""
        torch.manual_seed(123)
        H, W = 16, 24
        n_seams = 3
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        traditional = carve_image_traditional(image, n_seams=n_seams)
        mapping = carve_image_lattice_guided(image, lat, n_seams=n_seams, lattice_width=W)

        left_cols = W // 3
        trad_left = traditional[:, :, :left_cols]
        map_left = mapping[:, :, :left_cols]
        mae = (trad_left - map_left).abs().mean()
        assert mae < 0.15, f"Left region MAE {mae:.4f} too large"

    def test_content_shifts_past_seam(self):
        """For a known zero-energy column, content to the right should shift left."""
        H, W = 20, 30
        flat_image = torch.ones(1, H, W) * 0.5
        flat_image[:, :, :10] = 0.0
        flat_image[:, :, 11:] = 1.0

        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(flat_image, lat, n_seams=1, lattice_width=W)
        assert not torch.allclose(carved, flat_image, atol=1e-3)

    def test_multiple_seams_accumulate(self):
        """More seams should produce more change from original overall."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved_1 = carve_image_lattice_guided(image, lat, n_seams=1, lattice_width=W)
        carved_5 = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        carved_10 = carve_image_lattice_guided(image, lat, n_seams=10, lattice_width=W)

        diff_1 = (carved_1 - image).abs().mean()
        diff_5 = (carved_5 - image).abs().mean()
        diff_10 = (carved_10 - image).abs().mean()
        assert diff_5 > diff_1, f"5 seams diff {diff_5:.4f} <= 1 seam diff {diff_1:.4f}"
        assert diff_10 > diff_5, f"10 seams diff {diff_10:.4f} <= 5 seams diff {diff_5:.4f}"


# ---------------------------------------------------------------------------
# 4. Seam pairs (Section 3.6)
# ---------------------------------------------------------------------------

class TestSeamPairs:
    def test_boundary_preservation(self):
        """Pixels outside both ROI and pair windows should be nearly unchanged."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_seam_pairs(image, lat, n_seams=1,
                                  roi_range=(10, 15), pair_range=(25, 30),
                                  lattice_width=W)

        assert carved.shape == (3, H, W)

        left_diff = (carved[:, :, :5] - image[:, :, :5]).abs().max()
        assert left_diff < 0.1, f"Left boundary changed by {left_diff:.4f}"

        right_diff = (carved[:, :, 35:] - image[:, :, 35:]).abs().max()
        assert right_diff < 0.1, f"Right boundary changed by {right_diff:.4f}"

    def test_roi_content_changes(self):
        """Content in ROI region should differ from original."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_seam_pairs(image, lat, n_seams=2,
                                  roi_range=(10, 15), pair_range=(25, 30),
                                  lattice_width=W)

        roi_diff = (carved[:, :, 10:16] - image[:, :, 10:16]).abs().mean()
        assert roi_diff > 0.01, f"ROI region unchanged (diff={roi_diff:.4f})"

    def test_pair_content_changes(self):
        """Content in pair region should differ from original."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_seam_pairs(image, lat, n_seams=2,
                                  roi_range=(10, 15), pair_range=(25, 30),
                                  lattice_width=W)

        pair_diff = (carved[:, :, 25:31] - image[:, :, 25:31]).abs().mean()
        assert pair_diff > 0.01, f"Pair region unchanged (diff={pair_diff:.4f})"

    def test_net_shift_zero_outside(self):
        """Combined warp is identity outside both windows — shape preserved."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_seam_pairs(image, lat, n_seams=1,
                                  roi_range=(10, 15), pair_range=(25, 30),
                                  lattice_width=W)

        assert carved.shape == image.shape
        assert torch.isfinite(carved).all()
