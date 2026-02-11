"""Tests for lattice smoothing (Section 3.4.2, Figure 9)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.lattice import Lattice2D


class TestOverlapDetection:
    def test_rectangular_has_no_overlaps(self):
        """A rectangular lattice should never have overlapping scanlines."""
        lat = Lattice2D.rectangular(20, 30)
        assert not lat._check_overlapping_scanlines()

    def test_zigzag_has_overlaps(self):
        """A jagged zigzag curve should produce overlapping scanlines."""
        # Create a curve that zigzags sharply, causing scanlines to cross
        n_lines = 10
        origins = torch.zeros(n_lines, 2)
        tangents = torch.zeros(n_lines, 2)

        for i in range(n_lines):
            origins[i, 0] = float(i) * 5.0  # x increases
            # Sharp zigzag in y: 0, 20, 0, 20, ...
            origins[i, 1] = 20.0 if i % 2 == 1 else 0.0
            tangents[i] = torch.tensor([0.0, 1.0])  # All point up

        spacing = torch.ones(n_lines) * 5.0
        lat = Lattice2D(origins, tangents, spacing)
        # With scanlines all pointing up but origins zigzagging wildly,
        # adjacent scanlines should overlap
        assert lat._check_overlapping_scanlines()


class TestSmoothing:
    def test_smooth_reduces_overlaps(self):
        """Smoothing a zigzag lattice should reduce or eliminate overlaps."""
        n_lines = 20
        origins = torch.zeros(n_lines, 2)
        tangents = torch.zeros(n_lines, 2)

        for i in range(n_lines):
            origins[i, 0] = float(i) * 5.0
            # Moderate zigzag
            origins[i, 1] = 8.0 * ((-1) ** i)
            tangents[i] = torch.tensor([0.0, 1.0])

        spacing = torch.ones(n_lines) * 5.0
        lat = Lattice2D(origins, tangents, spacing)

        had_overlaps = lat._check_overlapping_scanlines()
        assert had_overlaps, "Test setup should have overlaps"

        lat.smooth(max_iterations=50)
        assert not lat._check_overlapping_scanlines(), \
            "Smoothing should eliminate overlaps for moderate zigzag"

    def test_smooth_preserves_endpoints(self):
        """Smoothing should not move the first and last origins."""
        n_lines = 10
        origins = torch.zeros(n_lines, 2)
        for i in range(n_lines):
            origins[i, 0] = float(i) * 5.0
            origins[i, 1] = 5.0 * ((-1) ** i)
        tangents = torch.zeros(n_lines, 2)
        tangents[:, 0] = 0.0
        tangents[:, 1] = 1.0
        spacing = torch.ones(n_lines) * 5.0

        lat = Lattice2D(origins, tangents, spacing)
        first_origin = lat.origins[0].clone()
        last_origin = lat.origins[-1].clone()

        lat.smooth(max_iterations=20)

        assert torch.allclose(lat.origins[0], first_origin), \
            "First origin should not move"
        assert torch.allclose(lat.origins[-1], last_origin), \
            "Last origin should not move"

    def test_smooth_approximates_original(self):
        """Smoothed curve should still be near the original curve."""
        n_lines = 20
        origins = torch.zeros(n_lines, 2)
        for i in range(n_lines):
            origins[i, 0] = float(i) * 5.0
            origins[i, 1] = 3.0 * ((-1) ** i)  # Small zigzag
        tangents = torch.zeros(n_lines, 2)
        tangents[:, 1] = 1.0
        spacing = torch.ones(n_lines) * 5.0

        lat = Lattice2D(origins, tangents, spacing)
        original_origins = lat.origins.clone()

        lat.smooth(max_iterations=50)

        # Max deviation should be bounded (smoothing shouldn't move points far)
        max_dev = (lat.origins - original_origins).norm(dim=1).max()
        assert max_dev < 10.0, f"Max deviation {max_dev:.2f} too large"

    def test_smooth_no_op_on_straight_line(self):
        """Smoothing a perfectly straight lattice should be a no-op."""
        lat = Lattice2D.rectangular(20, 30)
        original_origins = lat.origins.clone()

        lat.smooth(max_iterations=10)

        assert torch.allclose(lat.origins, original_origins, atol=1e-6), \
            "Rectangular lattice should not change after smoothing"

    def test_smooth_tangents_remain_unit(self):
        """After smoothing, tangent vectors should still be unit length."""
        n_lines = 15
        origins = torch.zeros(n_lines, 2)
        for i in range(n_lines):
            origins[i, 0] = float(i) * 5.0
            origins[i, 1] = 4.0 * ((-1) ** i)
        tangents = torch.zeros(n_lines, 2)
        tangents[:, 1] = 1.0
        spacing = torch.ones(n_lines) * 5.0

        lat = Lattice2D(origins, tangents, spacing)
        lat.smooth(max_iterations=20)

        norms = torch.norm(lat.tangents, dim=1)
        assert torch.allclose(norms, torch.ones(n_lines), atol=1e-4), \
            f"Tangent norms after smoothing: {norms.tolist()}"
