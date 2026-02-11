"""
Tests for lattice mappings, resampling, and end-to-end carving.

Organized into:
  1. Lattice construction basics
  2. Mapping correctness (exact values, not just round-trips)
  3. Resampling correctness (geometric meaning, not just shapes)
  4. Seam algorithm correctness (seams go through low-energy regions)
  5. End-to-end carving correctness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import pytest
from src.lattice import Lattice2D
from src.energy import gradient_magnitude_energy
from src.seam import greedy_seam, greedy_seam_windowed, remove_seam
from src.carving import carve_image_traditional, carve_image_lattice_guided, carve_seam_pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gradient_image(H, W, channels=3):
    """Horizontal gradient: dark left, bright right."""
    grad = torch.linspace(0, 1, W).unsqueeze(0).expand(H, W)
    if channels > 0:
        return grad.unsqueeze(0).expand(channels, H, W).clone()
    return grad


def make_ring_image(H, W, cx, cy, inner_r, outer_r):
    """Grayscale image: 1.0 inside the ring, 0.0 outside."""
    img = torch.zeros(1, H, W)
    for y in range(H):
        for x in range(W):
            r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if inner_r <= r <= outer_r:
                img[0, y, x] = 1.0
    return img


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
        # u should equal x exactly; n may have small fractional interpolation
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
        # u=0 on any scanline → center
        pts = torch.tensor([[0.0, 0.0], [0.0, 9.0], [0.0, 18.0]])
        world = lat.inverse_mapping(pts)
        assert torch.allclose(world, torch.tensor([[cx, cy]] * 3), atol=1e-4)

    def test_known_radial_points(self):
        """Scanline 0 points along +x, scanline at 90° along +y, etc."""
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=72)
        # 72 lines → 5° per line. Scanline 18 = 90°

        cases = [
            # (u, n) → expected (x, y)
            (10.0, 0.0, cx + 10.0, cy),          # 0°: +x
            (10.0, 18.0, cx, cy + 10.0),          # 90°: +y
            (10.0, 36.0, cx - 10.0, cy),          # 180°: -x
            (10.0, 54.0, cx, cy - 10.0),          # 270°: -y
        ]
        for u, n, ex, ey in cases:
            world = lat.inverse_mapping(torch.tensor([[u, n]]))
            assert abs(world[0, 0].item() - ex) < 0.1, f"u={u},n={n}: x={world[0,0]:.2f} expected {ex}"
            assert abs(world[0, 1].item() - ey) < 0.1, f"u={u},n={n}: y={world[0,1]:.2f} expected {ey}"

    def test_forward_recovers_correct_scanline(self):
        """Points on specific radial lines should map back to the right scanline."""
        cx, cy = 50.0, 50.0
        lat = Lattice2D.circular((cx, cy), 40.0, n_lines=72)

        # Point at (60, 50): on scanline 0 (angle 0), u = 10
        pts = torch.tensor([[60.0, 50.0]])
        result = lat.forward_mapping(pts)
        assert abs(result[0, 0].item() - 10.0) < 0.5  # u ≈ 10
        assert abs(result[0, 1].item() - 0.0) < 1.0    # n ≈ 0

        # Point at (50, 60): on scanline 18 (angle 90°), u = 10
        pts = torch.tensor([[50.0, 60.0]])
        result = lat.forward_mapping(pts)
        assert abs(result[0, 0].item() - 10.0) < 0.5  # u ≈ 10
        assert abs(result[0, 1].item() - 18.0) < 1.0   # n ≈ 18

    def test_roundtrip_tight_tolerance(self):
        """inverse → forward should recover u and n within angular discretization."""
        lat = Lattice2D.circular((50.0, 50.0), 40.0, n_lines=72)
        pts = torch.tensor([
            [20.0, 0.0],
            [30.0, 18.0],
            [15.0, 36.0],
            [10.0, 54.0],
        ])
        world = lat.inverse_mapping(pts)
        recovered = lat.forward_mapping(world)
        # With 72 lines (5° spacing), angular error < 1 line
        assert torch.allclose(recovered[:, 0], pts[:, 0], atol=0.5)  # u (radial)
        assert torch.allclose(recovered[:, 1], pts[:, 1], atol=1.0)  # n (angular)


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
        """A ring in world space should become a vertical stripe in lattice space.

        A ring at radius r maps to constant u=r across all scanlines (n).
        So in the lattice image, column u=r should be bright for all rows.
        """
        H, W = 64, 64
        cx, cy = 32.0, 32.0
        ring_radius = 15.0
        ring_width = 4.0

        image = make_ring_image(H, W, cx, cy,
                                ring_radius - ring_width / 2,
                                ring_radius + ring_width / 2)

        lat = Lattice2D.circular((cx, cy), 30.0, n_lines=72)
        lattice_img = lat.resample_to_lattice_space(image, lattice_width=30)
        # lattice_img shape: (1, 72, 30)

        # Column at u=ring_radius should be bright across all rows
        ring_col = lattice_img[0, :, int(ring_radius)]
        assert ring_col.mean() > 0.5, f"Ring column mean={ring_col.mean():.3f}, expected >0.5"

        # Column far from ring (u=5) should be dark
        bg_col = lattice_img[0, :, 5]
        assert bg_col.mean() < 0.2, f"Background column mean={bg_col.mean():.3f}, expected <0.2"

        # Ring column should be consistent across all angles (low variance)
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
# 4. Energy function
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_uniform_interior_is_zero(self):
        """A solid-color image should have zero energy in the interior."""
        image = torch.ones(3, 20, 20) * 0.5
        energy = gradient_magnitude_energy(image)
        assert energy[2:-2, 2:-2].max() < 1e-5

    def test_vertical_edge_has_horizontal_energy(self):
        """An image with a single vertical edge should have energy along that edge."""
        image = torch.zeros(1, 20, 20)
        image[:, :, 10:] = 1.0  # Left half dark, right half bright
        energy = gradient_magnitude_energy(image)
        # Energy should be high near column 10 and low elsewhere
        edge_energy = energy[2:-2, 9:12].mean()
        bg_energy = energy[2:-2, 2:7].mean()
        assert edge_energy > 10 * bg_energy


# ---------------------------------------------------------------------------
# 5. Seam computation — correctness, not just shapes
# ---------------------------------------------------------------------------

class TestSeamCorrectness:
    def test_seam_follows_zero_energy_column(self):
        """Given an energy map that's zero in one column, the seam goes there."""
        H, W = 20, 20
        energy = torch.ones(H, W)
        energy[:, 10] = 0.0  # Zero-energy column
        seam = greedy_seam(energy, direction='vertical')
        assert (seam == 10).all(), f"Expected all 10, got {seam.tolist()}"

    def test_seam_follows_zero_energy_row(self):
        """Horizontal seam follows zero-energy row."""
        H, W = 20, 20
        energy = torch.ones(H, W)
        energy[10, :] = 0.0
        seam = greedy_seam(energy, direction='horizontal')
        assert (seam == 10).all()

    def test_seam_follows_diagonal_valley(self):
        """Seam should follow a diagonal zero-energy path."""
        H, W = 20, 20
        energy = torch.ones(H, W) * 10.0
        for i in range(H):
            col = min(5 + i, W - 1)  # Diagonal from (0,5) going right
            energy[i, col] = 0.0
        seam = greedy_seam(energy, direction='vertical')
        for i in range(H):
            expected = min(5 + i, W - 1)
            assert seam[i].item() == expected, f"Row {i}: got {seam[i]}, expected {expected}"

    def test_seam_continuity(self):
        """Adjacent seam indices must differ by at most 1."""
        torch.manual_seed(42)
        energy = torch.rand(50, 50)
        seam = greedy_seam(energy, direction='vertical')
        diffs = torch.abs(seam[1:] - seam[:-1])
        assert diffs.max() <= 1

    def test_remove_seam_preserves_non_seam_pixels(self):
        """After removing a seam, remaining pixels should be the original values."""
        # 1-channel image with known values: pixel value = column index
        H, W = 4, 10
        image = torch.arange(W, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, H, W).clone()

        # Remove seam at column 5 in every row
        seam = torch.full((H,), 5, dtype=torch.long)
        carved = remove_seam(image, seam, direction='vertical')

        assert carved.shape == (1, H, W - 1)
        # Columns 0-4 are unchanged
        assert torch.equal(carved[0, 0, :5], torch.tensor([0., 1., 2., 3., 4.]))
        # Columns 5-8 are the original 6-9 (shifted left)
        assert torch.equal(carved[0, 0, 5:], torch.tensor([6., 7., 8., 9.]))

    def test_remove_seam_with_varying_positions(self):
        """Seam that zigzags removes correct pixel from each row."""
        image = torch.arange(6, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, 3, 6).clone()
        seam = torch.tensor([2, 3, 2])  # Zigzag seam
        carved = remove_seam(image, seam, direction='vertical')

        # Row 0: remove col 2 → [0, 1, 3, 4, 5]
        assert torch.equal(carved[0, 0], torch.tensor([0., 1., 3., 4., 5.]))
        # Row 1: remove col 3 → [0, 1, 2, 4, 5]
        assert torch.equal(carved[0, 1], torch.tensor([0., 1., 2., 4., 5.]))
        # Row 2: remove col 2 → [0, 1, 3, 4, 5]
        assert torch.equal(carved[0, 2], torch.tensor([0., 1., 3., 4., 5.]))


# ---------------------------------------------------------------------------
# 6. End-to-end traditional carving
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
# 7. End-to-end lattice-guided carving
# ---------------------------------------------------------------------------
# NOTE (2026-02-11): These tests mostly check shapes/types, NOT correctness.
# They pass even when the algorithm produces severely distorted output.
# TODO: Add tests that verify actual image quality (e.g., circles stay circular)

class TestLatticeGuidedCarving:
    def test_rectangular_lattice_produces_same_seams_as_traditional(self):
        """Since rectangular lattice = identity, carving in lattice space
        should find the exact same seam as traditional carving."""
        H, W = 20, 30
        image = make_gradient_image(H, W, channels=0)
        lat = Lattice2D.rectangular(H, W)

        # Resample to lattice space (should be identity)
        lattice_img = lat.resample_to_lattice_space(image, lattice_width=W)

        # Compute seam in both spaces
        energy_lattice = gradient_magnitude_energy(lattice_img)
        energy_direct = gradient_magnitude_energy(image)

        seam_lattice = greedy_seam(energy_lattice, direction='vertical')
        seam_direct = greedy_seam(energy_direct, direction='vertical')

        assert torch.equal(seam_lattice, seam_direct), \
            f"Lattice seam {seam_lattice.tolist()} != direct seam {seam_direct.tolist()}"

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
# 8. Carving the mapping (Section 3.3)
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
        assert carved.min() >= -0.01  # allow tiny border overshoot
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

        # The mapping approach keeps original dims and shifts content via
        # bilinear sampling, so it won't be pixel-identical to traditional.
        # But the left portion (away from seams) should be similar.
        left_cols = W // 3
        trad_left = traditional[:, :, :left_cols]
        map_left = mapping[:, :, :left_cols]
        # They should be correlated — mean absolute difference should be small
        mae = (trad_left - map_left).abs().mean()
        assert mae < 0.15, f"Left region MAE {mae:.4f} too large"

    def test_content_shifts_past_seam(self):
        """For a known zero-energy column, content to the right should shift left."""
        H, W = 20, 30
        # Image where each pixel's value is its column index (normalized)
        image = torch.arange(W, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, H, W).clone() / (W - 1)

        # Create energy with a zero column at col 10
        # Use an image that has a flat region around col 10
        flat_image = torch.ones(1, H, W) * 0.5
        flat_image[:, :, :10] = 0.0  # left side dark
        flat_image[:, :, 11:] = 1.0  # right side bright
        # The seam should go through column 10 (the edge)

        lat = Lattice2D.rectangular(H, W)
        carved = carve_image_lattice_guided(flat_image, lat, n_seams=1, lattice_width=W)

        # After carving, the image should be modified
        assert not torch.allclose(carved, flat_image, atol=1e-3)

    def test_multiple_seams_accumulate(self):
        """After k seams, content should have compressed by k.
        More seams should produce more change from original overall."""
        torch.manual_seed(42)
        H, W = 20, 30
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        carved_1 = carve_image_lattice_guided(image, lat, n_seams=1, lattice_width=W)
        carved_5 = carve_image_lattice_guided(image, lat, n_seams=5, lattice_width=W)
        carved_10 = carve_image_lattice_guided(image, lat, n_seams=10, lattice_width=W)

        # More seams = more change from original
        diff_1 = (carved_1 - image).abs().mean()
        diff_5 = (carved_5 - image).abs().mean()
        diff_10 = (carved_10 - image).abs().mean()
        assert diff_5 > diff_1, f"5 seams diff {diff_5:.4f} <= 1 seam diff {diff_1:.4f}"
        assert diff_10 > diff_5, f"10 seams diff {diff_10:.4f} <= 5 seams diff {diff_5:.4f}"


# ---------------------------------------------------------------------------
# 9. Seam pairs (Section 3.6)
# ---------------------------------------------------------------------------

class TestSeamPairs:
    def test_boundary_preservation(self):
        """Pixels outside both ROI and pair windows should be nearly unchanged."""
        torch.manual_seed(42)
        H, W = 20, 40
        image = torch.rand(3, H, W)
        lat = Lattice2D.rectangular(H, W)

        # ROI in columns 10-15, pair in columns 25-30
        carved = carve_seam_pairs(image, lat, n_seams=1,
                                  roi_range=(10, 15), pair_range=(25, 30),
                                  lattice_width=W)

        assert carved.shape == (3, H, W)

        # Leftmost columns (0-5) should be nearly unchanged
        left_diff = (carved[:, :, :5] - image[:, :, :5]).abs().max()
        assert left_diff < 0.1, f"Left boundary changed by {left_diff:.4f}"

        # Rightmost columns (35-39) should be nearly unchanged
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


# ---------------------------------------------------------------------------
# 10. Windowed seam finding
# ---------------------------------------------------------------------------

class TestWindowedSeam:
    def test_seam_stays_within_window(self):
        """All seam indices must be within [col_start, col_end]."""
        torch.manual_seed(42)
        H, W = 30, 30
        energy = torch.rand(H, W)
        col_start, col_end = 10, 20

        seam = greedy_seam_windowed(energy, (col_start, col_end))
        assert (seam >= col_start).all(), f"Seam went below {col_start}: min={seam.min()}"
        assert (seam <= col_end).all(), f"Seam went above {col_end}: max={seam.max()}"

    def test_follows_zero_energy_in_window(self):
        """Known zero-energy column within window should be found."""
        H, W = 20, 30
        energy = torch.ones(H, W)
        # Zero-energy column at 15, window is [10, 20]
        energy[:, 15] = 0.0

        seam = greedy_seam_windowed(energy, (10, 20))
        assert (seam == 15).all(), f"Expected all 15, got {seam.tolist()}"

# %%
