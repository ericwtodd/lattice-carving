"""
High-level carving functions that orchestrate the lattice-guided workflow.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .lattice import Lattice2D
from .energy import gradient_magnitude_energy
from .seam import greedy_seam, greedy_seam_windowed, remove_seam


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


def _carve_image_lattice_naive(image: torch.Tensor, lattice: Lattice2D,
                                n_seams: int, direction: str = 'vertical',
                                lattice_width: Optional[int] = None) -> torch.Tensor:
    """
    Naive lattice-guided seam carving (double interpolation).

    This implements the naive approach that resamples pixel data twice:
    V->L then L->V, causing blurring. Kept for comparison tests.

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

    # Default lattice width to image width
    if lattice_width is None:
        lattice_width = W

    # Step 1: Resample to lattice space
    lattice_image = lattice.resample_to_lattice_space(image, lattice_width)

    # Step 2-3: Perform seam carving in lattice space
    for i in range(n_seams):
        energy = gradient_magnitude_energy(lattice_image)
        seam = greedy_seam(energy, direction=direction)
        lattice_image = remove_seam(lattice_image, seam, direction=direction)

    # Step 4: Resample back to world space
    carved_image = lattice.resample_from_lattice_space(lattice_image, H, W)

    if squeeze_output:
        carved_image = carved_image.squeeze(0)

    return carved_image


def _precompute_forward_mapping(lattice: Lattice2D, H: int, W: int,
                                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute (u_map, n_map) for all world pixels via lattice.forward_mapping.

    Returns:
        u_map: (H, W) — u coordinate for each pixel
        n_map: (H, W) — fractional n coordinate for each pixel
    """
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    world_pts = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)

    lattice_pts = lattice.forward_mapping(world_pts)  # (H*W, 2)
    u_map = lattice_pts[:, 0].reshape(H, W)
    n_map = lattice_pts[:, 1].reshape(H, W)
    return u_map, n_map


def _interpolate_seam(seam: torch.Tensor, n_map: torch.Tensor) -> torch.Tensor:
    """Interpolate integer seam positions at fractional n values.

    Args:
        seam: Integer seam positions per scanline (n_lines,)
        n_map: Fractional scanline indices for each pixel (H, W)

    Returns:
        Interpolated seam position for each pixel (H, W)
    """
    n_lines = seam.shape[0]
    seam_float = seam.float()

    n_floor = torch.floor(n_map).long().clamp(0, n_lines - 1)
    n_ceil = (n_floor + 1).clamp(0, n_lines - 1)
    n_frac = n_map - torch.floor(n_map)

    # Clamp n_frac for pixels at the last scanline
    n_frac = n_frac.clamp(0.0, 1.0)

    seam_at_floor = seam_float[n_floor]  # (H, W)
    seam_at_ceil = seam_float[n_ceil]    # (H, W)
    seam_interp = (1.0 - n_frac) * seam_at_floor + n_frac * seam_at_ceil

    return seam_interp


def _warp_and_resample(image: torch.Tensor, lattice: Lattice2D,
                        u_map: torch.Tensor, n_map: torch.Tensor,
                        u_shift: torch.Tensor) -> torch.Tensor:
    """Apply u-shift and resample from a copy of the current image.

    Args:
        image: Current image (C, H, W)
        lattice: Lattice2D structure
        u_map: (H, W) u coordinates for each pixel
        n_map: (H, W) n coordinates for each pixel
        u_shift: (H, W) shift to apply to u coordinates

    Returns:
        Warped image (C, H, W)
    """
    C, H, W = image.shape
    device = image.device

    u_shifted = u_map + u_shift

    # Compute world-space coordinates for shifted lattice positions
    lattice_pts_shifted = torch.stack([
        u_shifted.reshape(-1),
        n_map.reshape(-1)
    ], dim=1)
    world_pts_star = lattice.inverse_mapping(lattice_pts_shifted)  # (H*W, 2)

    x_star = world_pts_star[:, 0].reshape(H, W)
    y_star = world_pts_star[:, 1].reshape(H, W)

    # Normalize to [-1, 1] for grid_sample
    x_norm = 2.0 * x_star / (W - 1) - 1.0
    y_norm = 2.0 * y_star / (H - 1) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    warped = F.grid_sample(
        image.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)  # (C, H, W)

    return warped


def carve_image_lattice_guided(image: torch.Tensor, lattice: Lattice2D,
                                n_seams: int, direction: str = 'vertical',
                                lattice_width: Optional[int] = None) -> torch.Tensor:
    """
    Lattice-guided seam carving using the "carving the mapping" approach.

    Instead of resampling pixel data through the lattice (which causes
    blurring from double interpolation), this method:
    1. Resamples only *energy* from V to L (interpolating energy is fine)
    2. Finds seam in L
    3. Shifts u-coordinates past the seam by +1
    4. Resamples pixel data only once: V*(p_w) = V_copy(g(u_shifted, n))

    See Section 3.3, Fig. 8 of Flynn et al. 2021.

    Args:
        image: Image tensor (C, H, W) or (H, W) in world space
        lattice: Lattice2D structure defining the coordinate mapping
        n_seams: Number of seams to remove
        direction: 'vertical' or 'horizontal'
        lattice_width: Width of lattice space (if None, uses image width)

    Returns:
        Carved image in world space (same shape as input)
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape
    device = image.device

    if lattice_width is None:
        lattice_width = W

    # Precompute forward mapping once (lattice geometry is fixed)
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)

    original_image = image.clone()  # Keep original for sampling
    cumulative_shift = torch.zeros_like(u_map)  # Track total shift across all iterations

    for i in range(n_seams):
        # Step 1: Compute energy from current warped state
        if i == 0:
            energy = gradient_magnitude_energy(original_image)
        else:
            current_warped = _warp_and_resample(original_image, lattice, u_map, n_map, cumulative_shift)
            energy = gradient_magnitude_energy(current_warped)

        # Step 2: Resample energy to lattice space
        if energy.dim() == 2:
            energy_3d = energy.unsqueeze(0)
        else:
            energy_3d = energy
        lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)

        # Step 3: Find seam in lattice space
        seam = greedy_seam(lattice_energy, direction=direction)

        # Step 4: Interpolate seam at each pixel's fractional n
        seam_interp = _interpolate_seam(seam, n_map)

        # Step 5: Update cumulative shift: +1 for pixels past the seam
        # Use u_map + cumulative_shift to account for previous seam removals
        u_adjusted = u_map + cumulative_shift
        new_shift = torch.where(u_adjusted >= seam_interp,
                                torch.ones_like(u_map),
                                torch.zeros_like(u_map))
        cumulative_shift = cumulative_shift + new_shift

    # Step 6: Final warp - sample from ORIGINAL image using cumulative shift
    current = _warp_and_resample(original_image, lattice, u_map, n_map, cumulative_shift)

    if squeeze_output:
        current = current.squeeze(0)

    return current


def carve_seam_pairs(image: torch.Tensor, lattice: Lattice2D,
                     n_seams: int, roi_range: Tuple[int, int],
                     pair_range: Tuple[int, int],
                     direction: str = 'vertical',
                     lattice_width: Optional[int] = None) -> torch.Tensor:
    """
    Seam pair carving for local region resizing without changing global boundaries.

    Two non-overlapping windows in lattice u-coordinates:
    - ROI (region of interest): the region to shrink (seam removed)
    - Pair: compensating region to expand (seam inserted)

    The +1 and -1 shifts cancel at the global boundary, preserving image
    dimensions and edge content.

    See Section 3.6 of Flynn et al. 2021.

    Args:
        image: Image tensor (C, H, W) or (H, W) in world space
        lattice: Lattice2D structure defining the coordinate mapping
        n_seams: Number of seam pairs to process
        roi_range: (u_start, u_end) — lattice u-coordinate range for ROI
        pair_range: (u_start, u_end) — lattice u-coordinate range for pair region
        direction: 'vertical' or 'horizontal'
        lattice_width: Width of lattice space (if None, uses image width)

    Returns:
        Carved image in world space (same shape as input)
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape
    device = image.device

    if lattice_width is None:
        lattice_width = W

    # Precompute forward mapping once
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)

    original_image = image.clone()  # Keep original for sampling
    cumulative_shift = torch.zeros_like(u_map)  # Track total shift across all iterations

    for i in range(n_seams):
        # Step 1: Compute energy from current warped state
        if i == 0:
            energy = gradient_magnitude_energy(original_image)
        else:
            current_warped = _warp_and_resample(original_image, lattice, u_map, n_map, cumulative_shift)
            energy = gradient_magnitude_energy(current_warped)

        # Step 2: Resample energy to lattice space
        if energy.dim() == 2:
            energy_3d = energy.unsqueeze(0)
        else:
            energy_3d = energy
        lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)

        # Step 3: Find seams in windowed regions
        roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction=direction)
        pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction=direction)

        # Step 4: Interpolate both seams at each pixel's fractional n
        roi_seam_interp = _interpolate_seam(roi_seam, n_map)
        pair_seam_interp = _interpolate_seam(pair_seam, n_map)

        # Step 5: Combined shift (account for cumulative shifts)
        # +1 for u >= roi_seam (ROI removal — compress)
        # -1 for u > pair_seam (pair insertion — expand)
        u_adjusted = u_map + cumulative_shift
        new_shift = torch.zeros_like(u_map)
        new_shift = new_shift + torch.where(u_adjusted >= roi_seam_interp,
                                             torch.ones_like(u_map),
                                             torch.zeros_like(u_map))
        new_shift = new_shift + torch.where(u_adjusted > pair_seam_interp,
                                             -torch.ones_like(u_map),
                                             torch.zeros_like(u_map))
        cumulative_shift = cumulative_shift + new_shift

    # Step 6: Final warp - sample from ORIGINAL image using cumulative shift
    current = _warp_and_resample(original_image, lattice, u_map, n_map, cumulative_shift)

    if squeeze_output:
        current = current.squeeze(0)

    return current


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
