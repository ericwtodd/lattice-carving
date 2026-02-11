"""Tests for energy functions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.energy import gradient_magnitude_energy


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
