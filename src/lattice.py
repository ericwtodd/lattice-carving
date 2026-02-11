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
import torch.nn.functional as F
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
        origins[:, 1] = torch.arange(height, device=device, dtype=torch.float32)

        tangents = torch.zeros(height, 2, device=device)
        tangents[:, 0] = 1.0  # Horizontal direction

        spacing = torch.ones(height, device=device)

        return cls(origins, tangents, spacing)

    @classmethod
    def from_curve_points(cls, curve_points: torch.Tensor, n_lines: int,
                         perp_extent: float, device='cpu'):
        """
        Create a lattice from a list of centerline points (Figure 9 approach).

        This is the paper's approach: take a sequence of (x, y) points defining
        a curve, compute tangents, build perpendicular scanlines.

        Args:
            curve_points: (N, 2) tensor of (x, y) points defining centerline
            n_lines: Number of scanlines perpendicular to curve
            perp_extent: Distance to extend perpendicular to curve (both sides)
            device: torch device

        Returns:
            Lattice with scanlines perpendicular to the curve
        """
        N = curve_points.shape[0]

        # Compute tangents by finite differences
        tangents = torch.zeros_like(curve_points)
        tangents[0] = curve_points[1] - curve_points[0]  # Forward diff at start
        tangents[-1] = curve_points[-1] - curve_points[-2]  # Backward diff at end
        tangents[1:-1] = (curve_points[2:] - curve_points[:-2]) / 2  # Central diff

        # Normalize tangents
        tangent_norms = torch.sqrt((tangents ** 2).sum(dim=1, keepdim=True))
        tangents = tangents / (tangent_norms + 1e-8)

        # Compute normals (perpendicular, pointing "left" of curve)
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)

        # Sample n_lines points along the curve
        if n_lines > N:
            # Interpolate to get more points
            indices = torch.linspace(0, N - 1, n_lines, device=device)
            indices_floor = torch.floor(indices).long()
            indices_ceil = torch.ceil(indices).long().clamp(max=N - 1)
            frac = indices - indices_floor.float()

            centerline_pts = (1 - frac).unsqueeze(1) * curve_points[indices_floor] + \
                            frac.unsqueeze(1) * curve_points[indices_ceil]
            scanline_tangents = (1 - frac).unsqueeze(1) * normals[indices_floor] + \
                               frac.unsqueeze(1) * normals[indices_ceil]
        else:
            # Subsample the curve
            indices = torch.linspace(0, N - 1, n_lines, device=device).long()
            centerline_pts = curve_points[indices]
            scanline_tangents = normals[indices]

        # Normalize tangents again after interpolation
        norms = torch.sqrt((scanline_tangents ** 2).sum(dim=1, keepdim=True))
        scanline_tangents = scanline_tangents / (norms + 1e-8)

        # IMPORTANT: Offset origins so centerline is at u=perp_extent (middle of scanline)
        # Instead of origin being on centerline (which makes u=0 to 2*perp_extent asymmetric),
        # we shift origin back by perp_extent so:
        #   u=0: one side (centerline - perp_extent * normal)
        #   u=perp_extent: centerline (middle)
        #   u=2*perp_extent: other side (centerline + perp_extent * normal)
        origins = centerline_pts - perp_extent * scanline_tangents

        # Spacing between scanlines (arc length approximation)
        spacing = torch.ones(n_lines, device=device) * (2 * perp_extent / n_lines)

        return cls(origins, scanline_tangents, spacing)

    @classmethod
    def from_horizontal_curve(cls, y_fn, x_range: Tuple[float, float],
                              n_lines: int, perp_extent: float, device='cpu'):
        """
        Create a lattice following a horizontally-flowing curved feature (like a river).

        The curve is defined by y = y_fn(x), and scanlines run perpendicular to
        the curve (roughly vertical). This creates a coordinate system where:
        - u ≈ x (position along the horizontal direction)
        - n = perpendicular distance from the curve

        Args:
            y_fn: Function x -> y defining the centerline curve
            x_range: (x_min, x_max) horizontal extent
            n_lines: Number of parallel scanlines (perpendicular spacing)
            perp_extent: Distance to extend perpendicular to curve (both sides)
            device: torch device

        Returns:
            Lattice with scanlines perpendicular to the curve
        """
        x_min, x_max = x_range

        # Create vertical scanlines at regular x positions
        # Each scanline is perpendicular to the local curve direction
        x_positions = torch.linspace(x_min, x_max, n_lines, device=device)

        origins = torch.zeros(n_lines, 2, device=device)
        tangents = torch.zeros(n_lines, 2, device=device)
        spacing = torch.ones(n_lines, device=device) * ((x_max - x_min) / n_lines)

        # For each x position, compute the curve y and the perpendicular direction
        for i, x in enumerate(x_positions):
            y = y_fn(x.item())

            # Compute derivative numerically for tangent direction
            dx = 0.1
            y_left = y_fn(max(x.item() - dx, x_min))
            y_right = y_fn(min(x.item() + dx, x_max))
            dy_dx = (y_right - y_left) / (2 * dx)

            # Tangent to curve: (1, dy/dx)
            curve_tangent = torch.tensor([1.0, dy_dx], device=device)
            curve_tangent = curve_tangent / torch.sqrt((curve_tangent ** 2).sum())

            # Normal (perpendicular): rotate tangent by 90 degrees
            normal = torch.tensor([-curve_tangent[1], curve_tangent[0]], device=device)

            # Origin is on the centerline
            origins[i] = torch.tensor([x, y], device=device)

            # Tangent for scanline points perpendicular to curve (along normal)
            tangents[i] = normal

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
        origins = torch.tensor(center, device=device, dtype=torch.float32).unsqueeze(0).repeat(n_lines, 1)

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
        Map world space points to lattice index space (vectorized).

        f: V -> L

        For 2D, we map (x, y) in world space to (u, n) in lattice space where:
        - u: position along the scanline
        - n: which scanline (can be fractional for interpolation)

        Args:
            world_points: Points in world space (N, 2)

        Returns:
            Points in lattice index space (N, 2) with (u, n) coordinates
        """
        N = world_points.shape[0]
        device = world_points.device

        # Compute vector from each scanline origin to each point
        # world_points: (N, 2), origins: (n_lines, 2)
        # diff: (N, n_lines, 2)
        diff = world_points.unsqueeze(1) - self.origins.unsqueeze(0)

        # Distance along normal direction for each scanline
        # normals: (n_lines, 2) -> (1, n_lines, 2)
        normal_dist = torch.abs((diff * self.normals.unsqueeze(0)).sum(dim=2))  # (N, n_lines)

        # Tangent projection (u) for each scanline — used to break ties.
        # When two scanlines (e.g. a radial line and its opposite) have the
        # same normal distance, the correct one is the one giving u >= 0.
        tangent_proj = (diff * self.tangents.unsqueeze(0)).sum(dim=2)  # (N, n_lines)
        penalty = torch.where(tangent_proj < 0,
                              torch.tensor(1e10, device=device),
                              torch.zeros(1, device=device))
        effective_dist = normal_dist + penalty

        # Find closest scanline for each point
        best_n = torch.argmin(effective_dist, dim=1)  # (N,)

        # Gather the diff vectors for the best scanlines
        batch_idx = torch.arange(N, device=device)
        best_diff = diff[batch_idx, best_n]  # (N, 2)
        best_tangent = self.tangents[best_n]  # (N, 2)

        # u = projection along tangent
        u = (best_diff * best_tangent).sum(dim=1)  # (N,)

        # Compute fractional scanline position by interpolating with neighbor
        best_normal_dist = normal_dist[batch_idx, best_n]  # (N,)

        next_n = torch.clamp(best_n + 1, 0, self.n_lines - 1)
        next_normal_dist = normal_dist[batch_idx, next_n]  # (N,)

        total_dist = best_normal_dist + next_normal_dist
        frac = torch.where(
            total_dist > 1e-6,
            best_normal_dist / total_dist,
            torch.zeros_like(total_dist)
        )

        # At the last scanline, no fractional part
        at_last = best_n >= self.n_lines - 1
        frac = torch.where(at_last, torch.zeros_like(frac), frac)

        n_frac = best_n.float() + frac

        return torch.stack([u, n_frac], dim=1)

    def inverse_mapping(self, lattice_points: torch.Tensor) -> torch.Tensor:
        """
        Map lattice index space points to world space (vectorized).

        g: L -> V

        For 2D, we map (u, n) in lattice space to (x, y) in world space.

        Args:
            lattice_points: Points in lattice index space (N, 2) with (u, n)

        Returns:
            Points in world space (N, 2) with (x, y)
        """
        u = lattice_points[:, 0]       # (N,)
        n_frac = lattice_points[:, 1]  # (N,)

        n = torch.floor(n_frac).long()
        frac = n_frac - n.float()

        # Clamp to valid range
        n = torch.clamp(n, 0, self.n_lines - 1)
        n_next = torch.clamp(n + 1, 0, self.n_lines - 1)

        # Gather origins and tangents for scanline n and n+1
        o_n = self.origins[n]          # (N, 2)
        t_n = self.tangents[n]         # (N, 2)
        o_next = self.origins[n_next]  # (N, 2)
        t_next = self.tangents[n_next] # (N, 2)

        # p = origin + u * tangent
        p_n = o_n + u.unsqueeze(1) * t_n
        p_next = o_next + u.unsqueeze(1) * t_next

        # Linearly interpolate between scanlines
        frac = frac.unsqueeze(1)  # (N, 1)
        world_points = (1.0 - frac) * p_n + frac * p_next

        return world_points

    def resample_to_lattice_space(self, image: torch.Tensor,
                                   lattice_width: int) -> torch.Tensor:
        """
        Resample an image from world space to lattice index space.

        Creates a "rectified" image where:
        - Rows correspond to scanlines (n)
        - Columns correspond to positions along the scanline (u)

        Uses inverse_mapping to find world-space sample locations, then
        bilinear sampling via grid_sample.

        Args:
            image: Image tensor (C, H, W) or (H, W)
            lattice_width: Number of samples along each scanline (u dimension)

        Returns:
            Rectified image in lattice space (C, n_lines, lattice_width)
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = image.shape
        device = image.device

        # Build a regular grid in lattice space: rows = scanlines, cols = u positions
        n_coords = torch.arange(self.n_lines, dtype=torch.float32, device=device)
        u_coords = torch.arange(lattice_width, dtype=torch.float32, device=device)

        # 'ij' indexing: first arg varies along dim-0 (rows), second along dim-1 (cols)
        n_grid, u_grid = torch.meshgrid(n_coords, u_coords, indexing='ij')
        # Both have shape (n_lines, lattice_width)

        # Flatten to (N, 2) with columns (u, n) for inverse_mapping
        lattice_pts = torch.stack([u_grid.reshape(-1), n_grid.reshape(-1)], dim=1)

        # Map every lattice pixel to its world-space (x, y)
        world_pts = self.inverse_mapping(lattice_pts)  # (N, 2)

        # Reshape back to the grid layout
        x = world_pts[:, 0].reshape(self.n_lines, lattice_width)
        y = world_pts[:, 1].reshape(self.n_lines, lattice_width)

        # Normalise to [-1, 1] for grid_sample (x indexes W, y indexes H)
        x_norm = 2.0 * x / (W - 1) - 1.0
        y_norm = 2.0 * y / (H - 1) - 1.0

        # grid_sample expects (N, H_out, W_out, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

        lattice_image = F.grid_sample(
            image.unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0)  # (C, n_lines, lattice_width)

        if squeeze_output:
            lattice_image = lattice_image.squeeze(0)

        return lattice_image

    def resample_from_lattice_space(self, lattice_image: torch.Tensor,
                                     output_height: int, output_width: int) -> torch.Tensor:
        """
        Resample an image from lattice index space back to world space.

        Uses forward_mapping to find each output pixel's lattice coordinates,
        then bilinear sampling via grid_sample.

        Args:
            lattice_image: Image in lattice space (C, n_lines, lattice_width)
            output_height: Desired output height in world space
            output_width: Desired output width in world space

        Returns:
            Image in world space (C, output_height, output_width)
        """
        if lattice_image.dim() == 2:
            lattice_image = lattice_image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        C, n_lines, lattice_width = lattice_image.shape
        device = lattice_image.device

        # Build a grid covering the output world space
        y_coords = torch.arange(output_height, dtype=torch.float32, device=device)
        x_coords = torch.arange(output_width, dtype=torch.float32, device=device)

        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Both have shape (output_height, output_width)

        world_pts = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

        # Map to lattice coordinates (u, n)
        lattice_pts = self.forward_mapping(world_pts)

        u = lattice_pts[:, 0].reshape(output_height, output_width)
        n = lattice_pts[:, 1].reshape(output_height, output_width)

        # Normalise to [-1, 1] for grid_sample
        # u indexes lattice_width (W dimension), n indexes n_lines (H dimension)
        u_norm = 2.0 * u / (lattice_width - 1) - 1.0
        n_norm = 2.0 * n / (n_lines - 1) - 1.0

        grid = torch.stack([u_norm, n_norm], dim=-1).unsqueeze(0)

        world_image = F.grid_sample(
            lattice_image.unsqueeze(0), grid,
            mode='bilinear', padding_mode='border', align_corners=True
        ).squeeze(0)

        if squeeze_output:
            world_image = world_image.squeeze(0)

        return world_image
