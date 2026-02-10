"""
Lattice structure and mapping functions.

A lattice is defined by a sequence of parallel planes with:
- Origin points o_n for each plane n
- Basis vectors u_n, v_n for each plane
- Spacing Δx between consecutive planes

The lattice provides bidirectional mapping between:
- World space V (actual image/volume coordinates)
- Lattice index space L (regular grid for seam carving)
"""

import torch
import numpy as np
from typing import Tuple, Optional


class Lattice2D:
    """
    2D lattice for image carving.

    For 2D images, we use a sequence of "scanlines" instead of planes.
    Each scanline is defined by:
    - An origin point o_n
    - A tangent vector u_n (along the line)
    - Spacing to the next scanline

    For a standard rectangular image, this would be horizontal lines.
    For curved carving (e.g., circular), lines can curve.
    """

    def __init__(
        self,
        origins: torch.Tensor,  # Shape: (n_lines, 2)
        tangents: torch.Tensor,  # Shape: (n_lines, 2)
        spacing: torch.Tensor,   # Shape: (n_lines,)
    ):
        """
        Initialize a 2D lattice.

        Args:
            origins: Origin points for each scanline (n_lines, 2)
            tangents: Unit tangent vectors for each scanline (n_lines, 2)
            spacing: Distance to next scanline (n_lines,)
        """
        self.origins = origins
        self.tangents = tangents
        self.spacing = spacing
        self.n_lines = origins.shape[0]

        # Compute normal vectors (perpendicular to tangents)
        self.normals = torch.stack([
            -tangents[:, 1],
            tangents[:, 0]
        ], dim=1)

    @classmethod
    def rectangular(cls, height: int, width: int, device='cpu'):
        """
        Create a rectangular lattice for standard seam carving.

        Args:
            height: Image height
            width: Image width
            device: torch device

        Returns:
            Rectangular lattice with horizontal scanlines
        """
        origins = torch.zeros(height, 2, device=device)
        origins[:, 1] = torch.arange(height, device=device)  # y-coordinates

        tangents = torch.zeros(height, 2, device=device)
        tangents[:, 0] = 1.0  # Horizontal direction

        spacing = torch.ones(height, device=device)

        return cls(origins, tangents, spacing)

    @classmethod
    def circular(cls, center: Tuple[float, float], radius: float,
                 n_lines: int, device='cpu'):
        """
        Create a circular lattice for carving circular objects.

        Args:
            center: Center of circle (x, y)
            radius: Radius of circle
            n_lines: Number of radial lines
            device: torch device

        Returns:
            Circular lattice with radial scanlines
        """
        angles = torch.linspace(0, 2 * np.pi, n_lines + 1, device=device)[:-1]

        # Origins at the center
        origins = torch.tensor(center, device=device).unsqueeze(0).repeat(n_lines, 1)

        # Tangents point radially outward
        tangents = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1)

        # Uniform angular spacing
        spacing = torch.ones(n_lines, device=device) * (2 * np.pi / n_lines)

        return cls(origins, tangents, spacing)

    def forward_mapping(self, world_points: torch.Tensor) -> torch.Tensor:
        """
        Map world space points to lattice index space.

        f: V → L

        Args:
            world_points: Points in world space (N, 2)

        Returns:
            Points in lattice index space (N, 2)
        """
        raise NotImplementedError("Forward mapping needs implementation")

    def inverse_mapping(self, lattice_points: torch.Tensor) -> torch.Tensor:
        """
        Map lattice index space points to world space.

        g: L → V

        Args:
            lattice_points: Points in lattice index space (N, 2)

        Returns:
            Points in world space (N, 2)
        """
        raise NotImplementedError("Inverse mapping needs implementation")
