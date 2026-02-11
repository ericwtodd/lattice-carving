"""Tests for energy functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.energy import gradient_magnitude_energy, normalize_energy, forward_energy


class TestGradientMagnitudeEnergy:
    def test_uniform_interior_is_zero(self):
        """A solid-color image should have zero energy in the interior."""
        image = torch.ones(3, 20, 20) * 0.5
        energy = gradient_magnitude_energy(image)
        assert energy[2:-2, 2:-2].max() < 1e-5

    def test_vertical_edge_has_horizontal_energy(self):
        """An image with a single vertical edge should have energy along that edge."""
        image = torch.zeros(1, 20, 20)
        image[:, :, 10:] = 1.0
        energy = gradient_magnitude_energy(image)
        edge_energy = energy[2:-2, 9:12].mean()
        bg_energy = energy[2:-2, 2:7].mean()
        assert edge_energy > 10 * bg_energy

    def test_horizontal_edge_has_vertical_energy(self):
        """An image with a horizontal edge should have energy along that edge."""
        image = torch.zeros(1, 20, 20)
        image[:, 10:, :] = 1.0
        energy = gradient_magnitude_energy(image)
        edge_energy = energy[9:12, 2:-2].mean()
        bg_energy = energy[2:7, 2:-2].mean()
        assert edge_energy > 10 * bg_energy

    def test_output_shape_matches_input(self):
        """Energy map should have same spatial dimensions as input."""
        image = torch.rand(3, 32, 48)
        energy = gradient_magnitude_energy(image)
        assert energy.shape == (32, 48)

    def test_grayscale_input(self):
        """Should work with 2D grayscale input."""
        image = torch.rand(20, 20)
        energy = gradient_magnitude_energy(image)
        assert energy.shape == (20, 20)

    def test_energy_nonnegative(self):
        """Energy should always be non-negative (L1 norm of gradients)."""
        torch.manual_seed(42)
        image = torch.rand(3, 30, 30)
        energy = gradient_magnitude_energy(image)
        assert (energy >= 0).all()


class TestNormalizeEnergy:
    def test_output_range(self):
        """Normalized energy should be in [0, 1]."""
        torch.manual_seed(42)
        energy = torch.rand(20, 30) * 100 + 5  # arbitrary range
        normed = normalize_energy(energy)
        assert normed.min() >= -1e-6
        assert normed.max() <= 1.0 + 1e-6

    def test_min_is_zero_max_is_one(self):
        """After normalization, min should be ~0 and max should be ~1."""
        energy = torch.tensor([[1.0, 5.0], [3.0, 10.0]])
        normed = normalize_energy(energy)
        assert abs(normed.min().item()) < 1e-6
        assert abs(normed.max().item() - 1.0) < 1e-6

    def test_preserves_relative_ordering(self):
        """Normalization is monotonic — relative ordering unchanged."""
        energy = torch.tensor([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]])
        normed = normalize_energy(energy)
        # Flatten and check: if a > b in original, a > b in normalized
        flat_e = energy.flatten()
        flat_n = normed.flatten()
        for i in range(len(flat_e)):
            for j in range(len(flat_e)):
                if flat_e[i] > flat_e[j]:
                    assert flat_n[i] > flat_n[j]

    def test_uniform_energy_produces_zeros(self):
        """Uniform energy should normalize to all zeros (or near-zero)."""
        energy = torch.ones(10, 10) * 5.0
        normed = normalize_energy(energy)
        assert normed.max() < 1e-6


class TestForwardEnergy:
    def test_uniform_image_low_energy(self):
        """A uniform image should have very low forward energy everywhere."""
        image = torch.ones(1, 20, 20) * 0.5
        fe = forward_energy(image)
        # Interior should be near zero (no edges to introduce)
        assert fe[2:-2, 2:-2].max() < 1e-4

    def test_vertical_edge_penalizes_crossing(self):
        """Forward energy should be high where a seam would cross an edge."""
        image = torch.zeros(1, 20, 20)
        image[:, :, 10:] = 1.0
        fe = forward_energy(image)
        # Energy near the edge should be higher than far from it
        edge_energy = fe[-1, 9:12].mean()  # bottom row near edge
        bg_energy = fe[-1, 2:5].mean()     # bottom row away from edge
        assert edge_energy > bg_energy

    def test_output_shape(self):
        """Forward energy should have same spatial dims as input."""
        image = torch.rand(3, 30, 40)
        fe = forward_energy(image)
        assert fe.shape == (30, 40)

    def test_cumulative_increases_downward(self):
        """Forward energy is cumulative DP — values should generally increase
        from top to bottom for non-trivial images."""
        torch.manual_seed(42)
        image = torch.rand(3, 20, 20)
        fe = forward_energy(image)
        # Average of last row should be >= average of first row
        assert fe[-1].mean() >= fe[0].mean()
