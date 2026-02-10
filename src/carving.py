"""
High-level carving functions that orchestrate the lattice-guided workflow.
"""

import torch
from typing import Optional
from .lattice import Lattice2D
from .energy import gradient_magnitude_energy
from .seam import greedy_seam, multi_greedy_seam, remove_seam


def carve_image_traditional(image: torch.Tensor, n_seams: int,
                           direction: str = 'vertical') -> torch.Tensor:
    """
    Traditional rectangular seam carving.

    Args:
        image: Image tensor (C, H, W) or (H, W)
        n_seams: Number of seams to remove
        direction: 'vertical' or 'horizontal'

    Returns:
        Carved image
    """
    carved = image.clone()

    for i in range(n_seams):
        # Compute energy
        energy = gradient_magnitude_energy(carved)

        # Find seam
        seam = greedy_seam(energy, direction=direction)

        # Remove seam
        carved = remove_seam(carved, seam, direction=direction)

    return carved


def carve_image_lattice_guided(image: torch.Tensor, lattice: Lattice2D,
                               n_seams: int, direction: str = 'vertical',
                               lattice_width: Optional[int] = None) -> torch.Tensor:
    """
    Lattice-guided seam carving.

    This implements the generalized carving workflow:
    1. Resample image to lattice index space (creates rectified image)
    2. Compute seams in lattice space
    3. Remove seams in lattice space
    4. Resample back to world space

    Args:
        image: Image tensor (C, H, W) or (H, W) in world space
        lattice: Lattice2D structure defining the coordinate mapping
        n_seams: Number of seams to remove
        direction: 'vertical' or 'horizontal'
        lattice_width: Width of lattice space (if None, uses image width)

    Returns:
        Carved image in world space
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape
    device = image.device

    # Default lattice width to image width
    if lattice_width is None:
        lattice_width = W

    # Step 1: Resample to lattice space
    lattice_image = lattice.resample_to_lattice_space(image, lattice_width)

    # Step 2-3: Perform seam carving in lattice space
    for i in range(n_seams):
        # Compute energy in lattice space
        energy = gradient_magnitude_energy(lattice_image)

        # Find seam in lattice space
        seam = greedy_seam(energy, direction=direction)

        # Remove seam in lattice space
        lattice_image = remove_seam(lattice_image, seam, direction=direction)

    # Step 4: Resample back to world space
    carved_image = lattice.resample_from_lattice_space(lattice_image, H, W)

    if squeeze_output:
        carved_image = carved_image.squeeze(0)

    return carved_image


def carve_with_comparison(image: torch.Tensor, lattice: Lattice2D,
                         n_seams: int, direction: str = 'vertical',
                         lattice_width: Optional[int] = None):
    """
    Perform both traditional and lattice-guided carving for comparison.

    Args:
        image: Image tensor (C, H, W) or (H, W)
        lattice: Lattice2D structure for lattice-guided carving
        n_seams: Number of seams to remove
        direction: 'vertical' or 'horizontal'
        lattice_width: Width of lattice space (if None, uses image width)

    Returns:
        Tuple of (traditional_carved, lattice_carved)
    """
    # Traditional carving
    traditional_carved = carve_image_traditional(image, n_seams, direction)

    # Lattice-guided carving
    lattice_carved = carve_image_lattice_guided(
        image, lattice, n_seams, direction, lattice_width
    )

    return traditional_carved, lattice_carved
