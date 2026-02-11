"""Shared test fixtures for lattice-carving test suite."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from src.lattice import Lattice2D


@pytest.fixture
def rect_lattice():
    """Standard 20x30 rectangular lattice."""
    return Lattice2D.rectangular(20, 30)


@pytest.fixture
def circular_lattice():
    """72-line circular lattice centered at (50, 50)."""
    return Lattice2D.circular((50.0, 50.0), 40.0, n_lines=72)


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
