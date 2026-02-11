"""
Tests for carving algorithms: traditional, lattice-guided, and seam pairs.

Organized into:
  1. Traditional carving (shape/validity)
  2. Carving the mapping — rectangular equivalence
  3. Seam pairs — boundary preservation

TODO (Task 7): Replace commented-out placeholder tests with real correctness
tests: circle preservation, no-blur validation, exact shift verification,
cumulative shift values, boundary fixedness with exact pixel checks.
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
# 1. Traditional carving — these are meaningful (verify exact dimension math)
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
# 2. Carving the mapping — one meaningful comparison test
# ---------------------------------------------------------------------------

class TestCarvingTheMapping:
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


# ---------------------------------------------------------------------------
# 3. Seam pairs — boundary preservation is meaningful
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


# ---------------------------------------------------------------------------
# Commented out: placeholder tests that only check shapes or "something changed".
# Will be replaced with real correctness tests in Task 7:
#   - Circle preservation (ring + circular lattice stays circular after N seams)
#   - No-blur (Laplacian variance of correct approach >= naive)
#   - Exact content shift direction (pixels at col K+ shift left by 1)
#   - Cumulative shift values (after N seams, shifts are integers in [0, N])
#   - Boundary fixedness (exact pixel equality outside lattice region)
# ---------------------------------------------------------------------------

# class TestLatticeGuidedCarving:
#     def test_lattice_guided_output_shape(self):
#     def test_lattice_guided_modifies_image(self):
#     def test_circular_lattice_carving_runs(self):

# class TestCarvingTheMapping (remaining weak tests):
#     def test_output_shape_unchanged(self):
#     def test_output_finite_and_valid(self):
#     def test_content_shifts_past_seam(self):
#     def test_multiple_seams_accumulate(self):

# class TestSeamPairs (remaining weak tests):
#     def test_roi_content_changes(self):
#     def test_pair_content_changes(self):
#     def test_net_shift_zero_outside(self):
