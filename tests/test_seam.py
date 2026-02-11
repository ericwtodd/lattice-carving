"""Tests for seam computation algorithms."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.seam import greedy_seam, greedy_seam_windowed, multi_greedy_seam, remove_seam


class TestGreedySeam:
    def test_seam_follows_zero_energy_column(self):
        """Given an energy map that's zero in one column, the seam goes there."""
        H, W = 20, 20
        energy = torch.ones(H, W)
        energy[:, 10] = 0.0
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
            col = min(5 + i, W - 1)
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

    def test_seam_length_matches_height(self):
        """Vertical seam length should equal image height."""
        energy = torch.rand(30, 40)
        seam = greedy_seam(energy, direction='vertical')
        assert seam.shape == (30,)

    def test_horizontal_seam_length_matches_width(self):
        """Horizontal seam length should equal image width."""
        energy = torch.rand(30, 40)
        seam = greedy_seam(energy, direction='horizontal')
        assert seam.shape == (40,)


class TestRemoveSeam:
    def test_preserves_non_seam_pixels(self):
        """After removing a seam, remaining pixels should be the original values."""
        H, W = 4, 10
        image = torch.arange(W, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, H, W).clone()
        seam = torch.full((H,), 5, dtype=torch.long)
        carved = remove_seam(image, seam, direction='vertical')

        assert carved.shape == (1, H, W - 1)
        assert torch.equal(carved[0, 0, :5], torch.tensor([0., 1., 2., 3., 4.]))
        assert torch.equal(carved[0, 0, 5:], torch.tensor([6., 7., 8., 9.]))

    def test_with_varying_positions(self):
        """Seam that zigzags removes correct pixel from each row."""
        image = torch.arange(6, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, 3, 6).clone()
        seam = torch.tensor([2, 3, 2])
        carved = remove_seam(image, seam, direction='vertical')

        assert torch.equal(carved[0, 0], torch.tensor([0., 1., 3., 4., 5.]))
        assert torch.equal(carved[0, 1], torch.tensor([0., 1., 2., 4., 5.]))
        assert torch.equal(carved[0, 2], torch.tensor([0., 1., 3., 4., 5.]))

    def test_reduces_width_by_one(self):
        """Vertical seam removal should reduce width by 1."""
        image = torch.rand(3, 20, 30)
        seam = torch.zeros(20, dtype=torch.long)
        carved = remove_seam(image, seam, direction='vertical')
        assert carved.shape == (3, 20, 29)

    def test_reduces_height_by_one(self):
        """Horizontal seam removal should reduce height by 1."""
        image = torch.rand(3, 20, 30)
        seam = torch.zeros(30, dtype=torch.long)
        carved = remove_seam(image, seam, direction='horizontal')
        assert carved.shape == (3, 19, 30)


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
        energy[:, 15] = 0.0
        seam = greedy_seam_windowed(energy, (10, 20))
        assert (seam == 15).all(), f"Expected all 15, got {seam.tolist()}"


class TestMultiGreedySeam:
    def test_returns_sorted_by_energy(self):
        """First seam should have lowest total energy."""
        torch.manual_seed(42)
        energy = torch.rand(20, 20)
        seams = multi_greedy_seam(energy, n_seams=5)
        assert len(seams) == 5

        # Compute total energies
        energies = []
        for seam in seams:
            total = sum(energy[i, seam[i]].item() for i in range(20))
            energies.append(total)
        # Should be sorted ascending
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1] + 1e-6
