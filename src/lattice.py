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

        For 2D, we map (x, y) in world space to (u, n) in lattice space where:
        - u: position along the scanline
        - n: which scanline (can be fractional for interpolation)

        Based on Section 3.2.1 of the paper (adapted for 2D).

        Args:
            world_points: Points in world space (N, 2)

        Returns:
            Points in lattice index space (N, 2) with (u, n) coordinates
        """
        N = world_points.shape[0]
        device = world_points.device

        lattice_points = torch.zeros(N, 2, device=device)

        for i in range(N):
            p_w = world_points[i]  # (x, y)

            # Find closest scanline by checking distance to each
            min_dist = float('inf')
            best_n = 0

            for n in range(self.n_lines):
                # Vector from scanline origin to point
                v = p_w - self.origins[n]

                # Distance along normal direction
                dist = torch.abs(torch.dot(v, self.normals[n]))

                if dist < min_dist:
                    min_dist = dist
                    best_n = n

            # Now compute position along the tangent and fractional scanline index
            # For scanline n, project point onto it
            n = best_n
            v = p_w - self.origins[n]

            # Position along tangent (u coordinate)
            u = torch.dot(v, self.tangents[n])

            # Fractional scanline position
            # Check neighboring scanlines for interpolation
            if n < self.n_lines - 1:
                dist_to_n = torch.abs(torch.dot(v, self.normals[n]))
                v_next = p_w - self.origins[n + 1]
                dist_to_next = torch.abs(torch.dot(v_next, self.normals[n + 1]))

                # Linear interpolation between scanlines
                total_dist = dist_to_n + dist_to_next
                if total_dist > 1e-6:
                    frac = dist_to_n / total_dist
                    n_frac = n + frac
                else:
                    n_frac = float(n)
            else:
                n_frac = float(n)

            lattice_points[i, 0] = u
            lattice_points[i, 1] = n_frac

        return lattice_points

    def inverse_mapping(self, lattice_points: torch.Tensor) -> torch.Tensor:
        """
        Map lattice index space points to world space.

        g: L → V

        For 2D, we map (u, n) in lattice space to (x, y) in world space where:
        - u: position along the scanline
        - n: which scanline (can be fractional for interpolation)

        Args:
            lattice_points: Points in lattice index space (N, 2) with (u, n)

        Returns:
            Points in world space (N, 2) with (x, y)
        """
        N = lattice_points.shape[0]
        device = lattice_points.device

        world_points = torch.zeros(N, 2, device=device)

        for i in range(N):
            u, n_frac = lattice_points[i]

            # Get integer scanline index and fractional part
            n = int(torch.floor(n_frac).item())
            frac = (n_frac - n).item()

            # Clamp to valid range
            n = max(0, min(n, self.n_lines - 1))

            # Get position on scanline n
            p_n = self.origins[n] + u * self.tangents[n]

            # If fractional, interpolate with next scanline
            if frac > 1e-6 and n < self.n_lines - 1:
                p_next = self.origins[n + 1] + u * self.tangents[n + 1]
                world_points[i] = (1 - frac) * p_n + frac * p_next
            else:
                world_points[i] = p_n

        return world_points
