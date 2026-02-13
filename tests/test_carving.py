"""
Tests for carving algorithms: traditional, lattice-guided, and seam pairs.

Focused on observable behavior, not implementation details.
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

from conftest import make_gradient_image, make_ring_image


class TestTraditionalCarving:
    def test_reduces_width(self):
        image = torch.rand(3, 20, 30)
        carved = carve_image_traditional(image, n_seams=5, direction='vertical')
        assert carved.shape == (3, 20, 25)

    def test_carved_output_is_valid(self):
        torch.manual_seed(42)
        image = torch.rand(3, 30, 30)
        carved = carve_image_traditional(image, n_seams=10, direction='vertical')
        assert carved.shape == (3, 30, 20)
        assert torch.isfinite(carved).all()
        assert carved.min() >= 0.0
        assert carved.max() <= 1.0


class TestCarvingTheMapping:
    def test_zero_seams_returns_original(self):
        """With 0 seams, output should match input (identity roundtrip)."""
        torch.manual_seed(42)
        H, W = 16, 20
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        result = carve_image_lattice_guided(image, lat, n_seams=0, lattice_width=W)
        max_diff = (result - image).abs().max()
        assert max_diff < 1e-4, f"Zero seams changed image by {max_diff:.6f}"

    def test_output_finite_and_valid(self):
        """Output should be finite and match input shape."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=3, lattice_width=W)
        assert carved.shape == (3, H, W)
        assert torch.isfinite(carved).all()

    def test_roi_pixels_outside_unchanged(self):
        """Pixels outside ROI bounds should be exactly unchanged."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(
            image, lat, n_seams=3, lattice_width=W,
            roi_bounds=(10.0, 30.0)
        )
        assert torch.equal(carved[:, :, :10], image[:, :, :10])
        assert torch.equal(carved[:, :, 31:], image[:, :, 31:])


class TestSeamPairs:
    def test_boundary_preservation(self):
        """Pixels far outside both windows should be nearly unchanged."""
        torch.manual_seed(42)
        H, W = 20, 50
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_seam_pairs(image, lat, n_seams=2,
                                  roi_range=(10, 15), pair_range=(30, 35),
                                  lattice_width=W)
        assert carved.shape == (3, H, W)
        left_diff = (carved[:, :, :5] - image[:, :, :5]).abs().max()
        assert left_diff < 0.05, f"Left boundary changed by {left_diff:.4f}"
        right_diff = (carved[:, :, 45:] - image[:, :, 45:]).abs().max()
        assert right_diff < 0.05, f"Right boundary changed by {right_diff:.4f}"

    def test_dimensions_preserved(self):
        """Seam pairs should never change image dimensions."""
        torch.manual_seed(42)
        for H, W in [(10, 30), (20, 50), (15, 40)]:
            image = torch.rand(3, H, W)
            lat = Lattice2D.rectangular(H, W)
            carved = carve_seam_pairs(image, lat, n_seams=1,
                                      roi_range=(5, 10),
                                      pair_range=(W - 10, W - 5),
                                      lattice_width=W)
            assert carved.shape == (3, H, W), f"Shape changed: {carved.shape}"

    def test_roi_content_changes(self):
        """Content inside the ROI window should be modified."""
        torch.manual_seed(42)
        H, W = 20, 50
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_seam_pairs(image, lat, n_seams=2,
                                  roi_range=(10, 15), pair_range=(30, 35),
                                  lattice_width=W)
        roi_diff = (carved[:, :, 10:16] - image[:, :, 10:16]).abs().max()
        assert roi_diff > 0.01, "ROI content unchanged"
