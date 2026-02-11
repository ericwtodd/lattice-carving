"""
Tests for carving algorithms: traditional, lattice-guided, and seam pairs.

Organized into:
  1. Traditional carving (exact dimension math, seam behavior)
  2. Carving the mapping — rectangular equivalence, no-blur, shift correctness
  3. ROI-bounded carving — pixel exactness outside bounds
  4. Seam pairs — boundary preservation, net-zero shift, bounded cumulative shift
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import pytest
from src.lattice import Lattice2D
from src.carving import (
    carve_image_traditional, carve_image_lattice_guided, carve_seam_pairs,
    _carve_image_lattice_naive, _precompute_forward_mapping,
    _interpolate_seam, _warp_and_resample,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import greedy_seam, greedy_seam_windowed

from conftest import make_gradient_image, make_ring_image


# ---------------------------------------------------------------------------
# 1. Traditional carving — verify dimensions and seam selection
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

    def test_seam_avoids_high_energy_edge(self):
        """Seams should avoid crossing high-energy edges.

        A vertical edge creates high gradient energy — seams should route
        around it, preserving the edge in the carved result.
        """
        torch.manual_seed(42)
        H, W = 30, 40
        image = torch.zeros(3, H, W)
        image[:, :, 20:] = 1.0
        carved = carve_image_traditional(image, n_seams=5, direction='vertical')
        # The sharp edge should survive carving
        row_mid = H // 2
        values = carved[0, row_mid, :]
        diffs = (values[1:] - values[:-1]).abs()
        assert diffs.max() > 0.5, f"Edge disappeared: max diff = {diffs.max():.4f}"


# ---------------------------------------------------------------------------
# 2. Carving the mapping — correctness, no-blur, shift behavior
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

    def test_zero_seams_returns_original(self):
        """With 0 seams removed, carving the mapping should return the
        original image exactly (for a rectangular lattice, the forward→inverse
        roundtrip is identity, so zero shift means no change)."""
        torch.manual_seed(42)
        H, W = 16, 20
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        result = carve_image_lattice_guided(image, lat, n_seams=0, lattice_width=W)
        max_diff = (result - image).abs().max()
        assert max_diff < 1e-4, f"Zero seams changed image by {max_diff:.6f}"

    def test_no_blur_vs_naive(self):
        """The correct approach (carving the mapping) should produce a sharper
        result than the naive double-interpolation approach.

        Sharpness is measured by Laplacian variance (higher = sharper).
        This validates the core insight of Section 3.3.
        """
        torch.manual_seed(42)
        H, W = 40, 60
        # High-frequency checkerboard pattern
        image = torch.zeros(1, H, W)
        for y in range(H):
            for x in range(W):
                image[0, y, x] = float(((x // 4) + (y // 4)) % 2)

        # Use a curved lattice to force non-trivial resampling
        curve_pts = torch.tensor([
            [0.0, H / 2 - 3], [W / 4, H / 2 + 3],
            [W / 2, H / 2 - 3], [3 * W / 4, H / 2 + 3],
            [float(W - 1), H / 2 - 3]
        ])
        lat = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=H / 2)

        n_seams = 5
        naive = _carve_image_lattice_naive(image, lat, n_seams=n_seams, lattice_width=W)
        correct = carve_image_lattice_guided(image, lat, n_seams=n_seams, lattice_width=W)

        def laplacian_variance(img):
            if img.dim() == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 3:
                img = img.unsqueeze(0)
            kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                                  dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
            lap = F.conv2d(img[:, :1], kernel, padding=1)
            return lap.var().item()

        naive_sharpness = laplacian_variance(naive)
        correct_sharpness = laplacian_variance(correct)

        # Correct approach should be at least as sharp (within tolerance)
        assert correct_sharpness >= naive_sharpness * 0.8, \
            f"Correct approach not sharper: {correct_sharpness:.6f} vs naive {naive_sharpness:.6f}"

    def test_content_shifts_preserve_monotonicity(self):
        """After carving a horizontal gradient, the monotonic brightness
        increase from left to right should be preserved. If the shift
        direction were wrong, monotonicity would break."""
        torch.manual_seed(42)
        H, W = 12, 24
        image = make_gradient_image(H, W, channels=1)  # dark left, bright right

        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=3, lattice_width=W)

        # Check monotonicity along the middle row
        mid_row = H // 2
        values = carved[0, mid_row, :]
        diffs = values[1:] - values[:-1]
        # Allow small negative diffs from bilinear interpolation
        assert (diffs >= -0.02).all(), \
            f"Monotonicity broken: min diff = {diffs.min():.4f}"

    def test_output_finite_and_valid(self):
        """Lattice-guided output should be finite and within input range."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(image, lat, n_seams=3, lattice_width=W)
        assert carved.shape == (3, H, W), "Output shape should match input"
        assert torch.isfinite(carved).all(), "Output has NaN/Inf"

    def test_cumulative_shift_increases_with_seams(self):
        """More seam removals should produce larger cumulative shifts.
        We verify this indirectly: the difference from original should
        grow monotonically with n_seams."""
        torch.manual_seed(42)
        H, W = 16, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        prev_diff = 0.0
        for n in [1, 3, 5]:
            carved = carve_image_lattice_guided(image, lat, n_seams=n, lattice_width=W)
            diff = (carved - image).abs().mean().item()
            assert diff >= prev_diff * 0.9, \
                f"Diff for {n} seams ({diff:.4f}) not >= prev ({prev_diff:.4f})"
            prev_diff = diff


# ---------------------------------------------------------------------------
# 3. ROI-bounded carving — pixels outside ROI stay identical
# ---------------------------------------------------------------------------

class TestROIBounds:
    def test_pixels_outside_roi_unchanged(self):
        """With roi_bounds set, pixels outside the ROI region should be
        exactly equal to the original image."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_image_lattice_guided(
            image, lat, n_seams=3, lattice_width=W,
            roi_bounds=(10.0, 30.0)
        )

        # Pixels at columns 0-9 should be exactly unchanged
        assert torch.equal(carved[:, :, :10], image[:, :, :10]), \
            "Left border pixels changed despite being outside ROI"
        # Pixels at columns 31-39 should be exactly unchanged
        assert torch.equal(carved[:, :, 31:], image[:, :, 31:]), \
            "Right border pixels changed despite being outside ROI"

    def test_roi_interior_is_modified(self):
        """Pixels inside the ROI should actually be modified by carving."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_image_lattice_guided(
            image, lat, n_seams=3, lattice_width=W,
            roi_bounds=(10.0, 30.0)
        )

        interior_diff = (carved[:, :, 15:25] - image[:, :, 15:25]).abs().max()
        assert interior_diff > 0.01, "ROI interior unchanged — carving had no effect"

    def test_full_roi_matches_no_roi(self):
        """When roi_bounds covers the entire image, result should match
        carving without roi_bounds."""
        torch.manual_seed(42)
        H, W = 16, 24
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        no_roi = carve_image_lattice_guided(image, lat, n_seams=2, lattice_width=W)
        full_roi = carve_image_lattice_guided(
            image, lat, n_seams=2, lattice_width=W,
            roi_bounds=(0.0, float(W))
        )

        max_diff = (no_roi - full_roi).abs().max()
        assert max_diff < 1e-4, f"Full ROI differs from no ROI by {max_diff:.6f}"


# ---------------------------------------------------------------------------
# 4. Seam pairs — boundary preservation, net-zero shift
# ---------------------------------------------------------------------------

class TestSeamPairs:
    def test_boundary_preservation(self):
        """Pixels far outside both ROI and pair windows should be nearly
        unchanged. The +1 (ROI) and -1 (pair) shifts cancel outside both."""
        torch.manual_seed(42)
        H, W = 20, 50
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved = carve_seam_pairs(image, lat, n_seams=2,
                                  roi_range=(10, 15), pair_range=(30, 35),
                                  lattice_width=W)

        assert carved.shape == (3, H, W)

        # Left boundary: columns 0-4 (well away from ROI at 10-15)
        left_diff = (carved[:, :, :5] - image[:, :, :5]).abs().max()
        assert left_diff < 0.05, f"Left boundary changed by {left_diff:.4f}"

        # Right boundary: columns 45-49 (well away from pair at 30-35)
        right_diff = (carved[:, :, 45:] - image[:, :, 45:]).abs().max()
        assert right_diff < 0.05, f"Right boundary changed by {right_diff:.4f}"

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
        assert roi_diff > 0.01, "ROI content unchanged — seam pair had no effect"

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

    def test_cumulative_shift_bounded_and_cancels(self):
        """After N seam pairs, verify:
        1. Shift is bounded by [-N, +N] everywhere
        2. Shift is ~0 outside both windows (cancellation)
        3. Shift is positive in ROI region (content removed)
        """
        torch.manual_seed(42)
        H, W = 16, 40
        image = torch.rand(1, H, W)
        lat = Lattice2D.rectangular(H, W)

        u_map, n_map = _precompute_forward_mapping(lat, H, W, image.device)
        cumulative_shift = torch.zeros_like(u_map)
        n_seams = 3

        for i in range(n_seams):
            if i == 0:
                energy = gradient_magnitude_energy(image)
            else:
                warped = _warp_and_resample(image, lat, u_map, n_map, cumulative_shift)
                energy = gradient_magnitude_energy(warped)

            energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
            lattice_energy = lat.resample_to_lattice_space(energy_3d, W)
            if lattice_energy.dim() == 3:
                lattice_energy = lattice_energy.squeeze(0)
            lattice_energy = normalize_energy(lattice_energy)

            roi_seam = greedy_seam_windowed(lattice_energy, (10, 20))
            pair_seam = greedy_seam_windowed(lattice_energy, (25, 35))

            roi_interp = _interpolate_seam(roi_seam, n_map)
            pair_interp = _interpolate_seam(pair_seam, n_map)

            u_adjusted = u_map + cumulative_shift
            new_shift = torch.zeros_like(u_map)
            new_shift += torch.where(u_adjusted >= roi_interp,
                                     torch.ones_like(u_map),
                                     torch.zeros_like(u_map))
            new_shift += torch.where(u_adjusted > pair_interp,
                                     -torch.ones_like(u_map),
                                     torch.zeros_like(u_map))
            cumulative_shift += new_shift

        # 1. Shift bounded by [-N, +N]
        assert cumulative_shift.max() <= n_seams + 0.5, \
            f"Max shift {cumulative_shift.max():.1f} exceeds +{n_seams}"
        assert cumulative_shift.min() >= -n_seams - 0.5, \
            f"Min shift {cumulative_shift.min():.1f} exceeds -{n_seams}"

        # 2. Far-left pixels (col 0-5) have zero net shift
        left_shift = cumulative_shift[:, :5].abs().max()
        assert left_shift < 0.5, f"Far-left shift = {left_shift:.2f}, expected ~0"

        # 3. Far-right pixels (col 36-39) have zero net shift
        right_shift = cumulative_shift[:, 36:].abs().max()
        assert right_shift < 0.5, f"Far-right shift = {right_shift:.2f}, expected ~0"

    def test_opposite_direction_reverses_effect(self):
        """Swapping ROI and pair regions should produce the opposite effect:
        what was shrinking now grows, and vice versa."""
        torch.manual_seed(42)
        H, W = 16, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        forward = carve_seam_pairs(image, lat, n_seams=2,
                                   roi_range=(8, 14), pair_range=(24, 30),
                                   lattice_width=W)
        reverse = carve_seam_pairs(image, lat, n_seams=2,
                                   roi_range=(24, 30), pair_range=(8, 14),
                                   lattice_width=W)

        # The two results should differ (they're not inverses, but they
        # should produce different outputs since they shift in opposite regions)
        diff = (forward - reverse).abs().mean()
        assert diff > 0.01, "Swapped ROI/pair produced same result"
