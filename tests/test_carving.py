"""
Tests for carving algorithms: traditional, lattice-guided, and seam pairs.

Focused on observable correctness: identity, seam placement, pixel preservation.
Visual validation is in examples/reproduce_figures.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.lattice import Lattice2D
from src.carving import (
    carve_image_traditional, carve_image_lattice_guided, carve_seam_pairs,
)
from src.energy import gradient_magnitude_energy

from conftest import make_gradient_image


# ---------------------------------------------------------------------------
# 1. Rectangular lattice-guided == traditional carving
# ---------------------------------------------------------------------------

class TestRectangularMatchesTraditional:
    def test_rectangular_lattice_guided_is_finite_and_bounded(self):
        """Rectangular lattice-guided carving must produce finite output in [0,1].

        Note: traditional carving shrinks image width (removes seams), while
        lattice-guided warps the image (keeps original size). They are not directly
        comparable by pixel value — seam equivalence is tested in test_lattice.py.
        """
        torch.manual_seed(42)
        H, W = 20, 30
        image = make_gradient_image(H, W)
        lattice = Lattice2D.rectangular(H, W)

        result = carve_image_lattice_guided(
            image, lattice, n_seams=5, lattice_width=W, method='greedy'
        )

        assert result.shape == (3, H, W), f"Shape changed: {result.shape}"
        assert torch.isfinite(result).all(), "Output contains NaN or Inf"
        assert result.min() >= 0.0, f"Output below 0: {result.min():.4f}"
        assert result.max() <= 1.0, f"Output above 1: {result.max():.4f}"

    def test_traditional_reduces_width(self):
        """Traditional carving must reduce image width by n_seams."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = make_gradient_image(H, W)
        result = carve_image_traditional(image, n_seams=5, direction='vertical')
        assert result.shape == (3, H, W - 5), f"Wrong shape: {result.shape}"


# ---------------------------------------------------------------------------
# 2. Zero seams is identity
# ---------------------------------------------------------------------------

class TestZeroSeamsIsIdentity:
    def test_zero_seams_traditional(self):
        torch.manual_seed(0)
        image = torch.rand(3, 16, 20)
        result = carve_image_traditional(image, n_seams=0, direction='vertical')
        assert torch.equal(result, image)

    def test_zero_seams_lattice_guided(self):
        """n_seams=0 should return the original image without any modification."""
        torch.manual_seed(0)
        H, W = 16, 20
        image = torch.rand(3, H, W)
        lattice = Lattice2D.rectangular(H, W)
        result = carve_image_lattice_guided(image, lattice, n_seams=0, lattice_width=W)
        max_diff = (result - image).abs().max()
        assert max_diff < 1e-4, f"Zero seams changed image by {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 3. One seam removes the lowest-energy column
# ---------------------------------------------------------------------------

class TestOneSeamRemovesLowestEnergyColumn:
    def test_one_seam_removes_zero_energy_column(self):
        """Traditional carving with one zero-energy column should remove that column."""
        H, W = 10, 15
        image = torch.rand(1, H, W)
        # Make column 7 all zeros (zero gradient energy)
        image[:, :, 7] = 0.0

        # Standard traditional carving — DP finds globally optimal seam
        carved = carve_image_traditional(image, n_seams=1, direction='vertical')

        # The zero-energy column should be gone: output width = W-1
        assert carved.shape == (1, H, W - 1), f"Wrong shape: {carved.shape}"

        # The carved image should contain none of column 7's zeros surrounded by nonzero
        # Since all neighboring columns are random (nonzero), the seam must pass through col 7
        # Verify: no all-zero column in the output
        col_means = carved[0].mean(dim=0)  # (W-1,)
        assert col_means.min() > 0.01, "Zero column was not removed"


# ---------------------------------------------------------------------------
# 4. Resample round-trip accuracy (tight tolerance)
# ---------------------------------------------------------------------------

class TestResampleRoundtripRectangular:
    def test_resample_roundtrip_tight_tolerance(self):
        """Resample to lattice space and back should be near-exact for rectangular lattice."""
        H, W = 16, 24
        image = make_gradient_image(H, W)
        lattice = Lattice2D.rectangular(H, W)
        roundtrip = lattice.resample_from_lattice_space(
            lattice.resample_to_lattice_space(image, W), H, W
        )
        max_diff = (image - roundtrip).abs().max()
        assert max_diff < 1e-3, (
            f"Round-trip error {max_diff:.6f} exceeds 1e-3. "
            f"Bilinear interpolation should be near-lossless for rectangular lattice."
        )


# ---------------------------------------------------------------------------
# 5. ROI pixels outside bounds are unchanged
# ---------------------------------------------------------------------------

class TestROIOutsidePixelsUnchanged:
    def test_roi_outside_pixels_unchanged(self):
        """Pixels clearly outside the ROI u-bounds should be exactly the original."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lattice = Lattice2D.rectangular(H, W)

        # ROI: u in [12, 28]. Pixels at x=0..9 and x=32..39 are clearly outside.
        carved = carve_image_lattice_guided(
            image, lattice, n_seams=3, lattice_width=W,
            roi_bounds=(12.0, 28.0)
        )

        left_diff = (carved[:, :, :10] - image[:, :, :10]).abs().max()
        right_diff = (carved[:, :, 33:] - image[:, :, 33:]).abs().max()
        assert left_diff == 0.0, f"Left of ROI changed by {left_diff:.6f}"
        assert right_diff == 0.0, f"Right of ROI changed by {right_diff:.6f}"


# ---------------------------------------------------------------------------
# 6. Seam pairs preserve image dimensions
# ---------------------------------------------------------------------------

class TestSeamPairsPreservesDimensions:
    def test_seam_pairs_never_change_shape(self):
        """carve_seam_pairs must always return the same shape as input."""
        torch.manual_seed(42)
        for H, W in [(10, 30), (20, 50), (15, 40)]:
            image = torch.rand(3, H, W)
            lattice = Lattice2D.rectangular(H, W)
            carved = carve_seam_pairs(
                image, lattice, n_seams=1,
                roi_range=(W // 4, W // 3),
                pair_range=(2 * W // 3, 3 * W // 4),
                lattice_width=W
            )
            assert carved.shape == (3, H, W), (
                f"Shape changed for H={H}, W={W}: got {carved.shape}"
            )
