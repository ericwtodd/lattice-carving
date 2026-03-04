"""
High-level carving functions that orchestrate the lattice-guided workflow.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .lattice import Lattice2D
from .energy import gradient_magnitude_energy, normalize_energy
from .seam import (greedy_seam, greedy_seam_windowed, remove_seam,
                    dp_seam, dp_seam_windowed, dp_seam_cyclic)


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
        energy = normalize_energy(gradient_magnitude_energy(carved))
        seam = greedy_seam(energy, direction=direction)
        carved = remove_seam(carved, seam, direction=direction)

    return carved


def _carve_image_lattice_naive(image: torch.Tensor, lattice: Lattice2D,
                                n_seams: int, direction: str = 'vertical',
                                lattice_width: Optional[int] = None) -> torch.Tensor:
    """
    Naive lattice-guided seam carving (double interpolation).
    Kept for comparison tests.
    """
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape

    if lattice_width is None:
        lattice_width = W

    lattice_image = lattice.resample_to_lattice_space(image, lattice_width)

    for i in range(n_seams):
        energy = gradient_magnitude_energy(lattice_image)
        seam = greedy_seam(energy, direction=direction)
        lattice_image = remove_seam(lattice_image, seam, direction=direction)

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


def _compute_valid_mask(lattice: Lattice2D, u_map: torch.Tensor,
                        n_map: torch.Tensor, H: int, W: int,
                        device: torch.device,
                        threshold: float = 3.0,
                        lattice_width: Optional[int] = None) -> torch.Tensor:
    """Compute a boolean mask of pixels that lie inside the lattice region.

    Uses a roundtrip test: forward -> inverse -> compare with original position.
    Pixels inside the lattice region roundtrip accurately; distant pixels don't.

    Optionally also requires u and n coordinates to be within lattice bounds.
    This is important for curved lattices where the forward mapping can send
    distant pixels to far-away u values (e.g. across an S-bend), producing
    ambiguous mappings that cause slicing/smearing artifacts during carving.

    Args:
        lattice: Lattice2D structure
        u_map: (H, W) u coordinates from forward mapping
        n_map: (H, W) n coordinates from forward mapping
        H, W: image dimensions
        device: torch device
        threshold: max roundtrip error in pixels to be considered valid
        lattice_width: if given, also require 0 <= u <= lattice_width and
            0 <= n <= n_lines-1 for a pixel to be valid

    Returns:
        valid_mask: (H, W) boolean tensor
    """
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    world_pts = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)
    lattice_pts = torch.stack([u_map.reshape(-1), n_map.reshape(-1)], dim=1)
    roundtrip_pts = lattice.inverse_mapping(lattice_pts)
    roundtrip_err = torch.sqrt(((world_pts - roundtrip_pts)**2).sum(dim=1)).reshape(H, W)
    valid = roundtrip_err < threshold

    if lattice_width is not None:
        valid = valid & (u_map >= 0) & (u_map <= lattice_width)
        valid = valid & (n_map >= 0) & (n_map <= lattice.n_lines - 1)

    return valid


def _interpolate_seam(seam: torch.Tensor, n_map: torch.Tensor,
                      cyclic: bool = False) -> torch.Tensor:
    """Interpolate integer seam positions at fractional n values.

    Args:
        seam: Integer seam positions per scanline (n_lines,)
        n_map: Fractional scanline indices for each pixel (H, W)
        cyclic: If True, wrap n_ceil around for cyclic lattices

    Returns:
        Interpolated seam position for each pixel (H, W)
    """
    n_lines = seam.shape[0]
    seam_float = seam.float()

    n_floor = torch.floor(n_map).long().clamp(0, n_lines - 1)
    if cyclic:
        n_ceil = (n_floor + 1) % n_lines
    else:
        n_ceil = (n_floor + 1).clamp(0, n_lines - 1)
    n_frac = n_map - torch.floor(n_map)
    n_frac = n_frac.clamp(0.0, 1.0)

    seam_at_floor = seam_float[n_floor]
    seam_at_ceil = seam_float[n_ceil]
    seam_interp = (1.0 - n_frac) * seam_at_floor + n_frac * seam_at_ceil

    return seam_interp


def _warp_and_resample(image: torch.Tensor, lattice: Lattice2D,
                        u_map: torch.Tensor, n_map: torch.Tensor,
                        u_shift: torch.Tensor) -> torch.Tensor:
    """Apply u-shift via modified inverse mapping g* and resample from image.

    Implements paper Eq. 4-5: for each world pixel p_w with lattice coords
    (u, n) = f(p_w), compute p*_w = g*(u, n) where g* maps (u + shift, n)
    back to world space. The output pixel gets image(p*_w).

    Args:
        image: Source image V_c to sample from (C, H, W)
        lattice: Lattice2D structure
        u_map: (H, W) u coordinates for each pixel (from forward mapping)
        n_map: (H, W) fractional n coordinates for each pixel
        u_shift: (H, W) shift to apply to u (the "carved" part of g*)

    Returns:
        Warped image V* (C, H, W)
    """
    C, H, W = image.shape
    device = image.device

    u_shifted = u_map + u_shift

    lattice_pts_shifted = torch.stack([
        u_shifted.reshape(-1),
        n_map.reshape(-1)
    ], dim=1)
    world_pts_star = lattice.inverse_mapping(lattice_pts_shifted)

    x_star = world_pts_star[:, 0].reshape(H, W)
    y_star = world_pts_star[:, 1].reshape(H, W)

    x_norm = 2.0 * x_star / (W - 1) - 1.0
    y_norm = 2.0 * y_star / (H - 1) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)

    warped = F.grid_sample(
        image.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)

    return warped


def _shift_to_warp_grid(lattice: Lattice2D, u_map: torch.Tensor,
                         n_map: torch.Tensor, combined_shift: torch.Tensor,
                         H: int, W: int) -> torch.Tensor:
    """Convert a u-shift map to a normalized sampling grid for grid_sample.

    Applies the shift to u_map, maps back to world space via inverse_mapping,
    and normalises to [-1, 1] for use with F.grid_sample.

    Returns:
        grid: (1, H, W, 2) normalised sampling grid
    """
    u_shifted = u_map + combined_shift
    lattice_pts = torch.stack([u_shifted.reshape(-1), n_map.reshape(-1)], dim=1)
    world_pts = lattice.inverse_mapping(lattice_pts)
    x_star = world_pts[:, 0].reshape(H, W)
    y_star = world_pts[:, 1].reshape(H, W)
    x_norm = 2.0 * x_star / (W - 1) - 1.0
    y_norm = 2.0 * y_star / (H - 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # (1, H, W, 2)


def _sample_src_map(src_map: torch.Tensor, grid: torch.Tensor,
                    valid_mask: torch.Tensor,
                    x_grid: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
    """Compose src_map with a new warp grid.

    For each output pixel, finds where in the ORIGINAL image it should
    sample from after applying the new warp. Pixels outside the valid mask
    keep identity source coordinates (→ original image unchanged).

    Args:
        src_map:    (2, H, W) coordinate map: channel 0 = src_x, channel 1 = src_y
        grid:       (1, H, W, 2) normalised sampling grid from _shift_to_warp_grid
        valid_mask: (H, W) bool — where to apply the warp
        x_grid:     (H, W) original x pixel indices (identity x-coords)
        y_grid:     (H, W) original y pixel indices (identity y-coords)

    Returns:
        Updated src_map (2, H, W)
    """
    new_src = F.grid_sample(
        src_map.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)
    # Outside valid region: source stays at original pixel (identity)
    new_src[0] = torch.where(valid_mask, new_src[0], x_grid)
    new_src[1] = torch.where(valid_mask, new_src[1], y_grid)
    return new_src


def _src_map_to_image(image: torch.Tensor, src_map: torch.Tensor,
                      H: int, W: int) -> torch.Tensor:
    """Sample image at positions given by src_map (single grid_sample call).

    Args:
        image:   Original image (C, H, W)
        src_map: (2, H, W) coordinate map in pixel coords

    Returns:
        Resampled image (C, H, W)
    """
    x_norm = 2.0 * src_map[0] / (W - 1) - 1.0
    y_norm = 2.0 * src_map[1] / (H - 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)


def carve_image_lattice_guided(image: torch.Tensor, lattice: Lattice2D,
                                n_seams: int, direction: str = 'vertical',
                                lattice_width: Optional[int] = None,
                                roi_bounds: Optional[Tuple[float, float]] = None,
                                n_candidates: int = 1,
                                method: str = 'dp') -> torch.Tensor:
    """
    Lattice-guided seam carving via composed coordinate mapping (Section 3.3, Eq. 5).

    Instead of warping pixel values N times (which compounds blur), we accumulate
    a 2-channel source-coordinate map across all seam iterations and apply a
    single final grid_sample to the original image. This matches the C++ approach
    of composing mappings before sampling.

    Each iteration:
      1. Derive current image from src_map (one grid_sample — for energy only)
      2. Compute energy, resample to lattice space, find seam → combined_shift
      3. Compute warp grid from (u_map + shift) via inverse_mapping
      4. Compose: src_map ← sample(src_map, warp_grid)

    Final output: grid_sample(original_image, src_map_final) — one interpolation.
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

    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, device,
                                     lattice_width=lattice_width)
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)

    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic

    # Identity source coordinate map: src_map[0]=x, src_map[1]=y
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    src_map = torch.stack([x_grid, y_grid], dim=0)  # (2, H, W)

    for i in range(n_seams):
        # Current image for energy (sampled from original via accumulated map)
        current_image = _src_map_to_image(image, src_map, H, W)

        energy = gradient_magnitude_energy(current_image)
        if energy.dim() == 2:
            energy_3d = energy.unsqueeze(0)
        else:
            energy_3d = energy
        lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)
        lattice_energy = normalize_energy(lattice_energy)

        if method == 'dp':
            seam = dp_seam(lattice_energy, direction=direction)
        else:
            seam = greedy_seam(lattice_energy, direction=direction,
                               n_candidates=n_candidates)

        seam_interp = _interpolate_seam(seam, n_map, cyclic=is_cyclic)

        shift = torch.where(u_map >= seam_interp,
                            torch.ones_like(u_map),
                            torch.zeros_like(u_map))
        shift = torch.where(valid_mask, shift, torch.zeros_like(shift))

        # Compose warp into source coordinate map
        grid = _shift_to_warp_grid(lattice, u_map, n_map, shift, H, W)
        src_map = _sample_src_map(src_map, grid, valid_mask, x_grid, y_grid)

    # Single final interpolation from original image
    result = _src_map_to_image(image, src_map, H, W)
    result = torch.where(valid_mask_3d, result, image)

    if roi_bounds is not None:
        u_min, u_max = roi_bounds
        roi_valid = (u_map >= u_min) & (u_map <= u_max)
        roi_valid = roi_valid & (n_map >= 0) & (n_map <= lattice.n_lines - 1)
        roi_valid_3d = roi_valid.unsqueeze(0).expand_as(result)
        result = torch.where(roi_valid_3d, result, image)

    if squeeze_output:
        result = result.squeeze(0)

    return result


def carve_seam_pairs(image: torch.Tensor, lattice: Lattice2D,
                     n_seams: int, roi_range: Tuple[int, int],
                     pair_range: Tuple[int, int],
                     direction: str = 'vertical',
                     lattice_width: Optional[int] = None,
                     n_candidates: int = 1,
                     method: str = 'dp',
                     mode: str = 'shrink') -> torch.Tensor:
    """
    Seam pair carving via composed coordinate mapping (Section 3.3 + 3.6).

    Accumulates seam shifts into a 2-channel source-coordinate map and applies
    a single final grid_sample to the original image, matching the C++ approach
    of composing all mappings before sampling. Eliminates compounding blur from
    N iterative bilinear resamples.

    Each iteration:
      1. Derive current image from src_map (one grid_sample — for energy only)
      2. Find ROI and pair seams in windowed lattice-space energy
      3. Build combined shift (shrink ROI + expand pair)
      4. Compute warp grid via inverse_mapping and compose into src_map

    Final output: grid_sample(original_image, src_map_final) — one interpolation.

    Args:
        image: Image tensor (C, H, W) or (H, W) in world space
        lattice: Lattice2D structure defining the coordinate mapping
        n_seams: Number of seam pairs to process
        roi_range: (u_start, u_end) — lattice u-coordinate range for ROI
        pair_range: (u_start, u_end) — lattice u-coordinate range for pair region
        direction: 'vertical' or 'horizontal'
        lattice_width: Width of lattice space (if None, uses image width)
        n_candidates: Number of multi-greedy starting points (1 = single greedy)
        method: 'dp' (optimal, default) or 'greedy' (fast, may wander)
        mode: 'shrink' (default) to compress ROI / expand pair,
              'grow' to expand ROI / compress pair

    Returns:
        Carved image in world space (same shape as input)
    """
    if mode not in ('shrink', 'grow'):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'shrink' or 'grow'.")

    roi_sign = 1.0 if mode == 'shrink' else -1.0

    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape
    device = image.device

    if lattice_width is None:
        lattice_width = W

    u_map, n_map = _precompute_forward_mapping(lattice, H, W, device)
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, device,
                                     lattice_width=lattice_width)
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)

    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic

    # Identity source coordinate map
    y_coords = torch.arange(H, dtype=torch.float32, device=device)
    x_coords = torch.arange(W, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    src_map = torch.stack([x_grid, y_grid], dim=0)  # (2, H, W)

    for i in range(n_seams):
        # Current image for energy (sampled from original via accumulated map)
        current_image = _src_map_to_image(image, src_map, H, W)

        energy = gradient_magnitude_energy(current_image)
        if energy.dim() == 2:
            energy_3d = energy.unsqueeze(0)
        else:
            energy_3d = energy
        lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_width)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)
        lattice_energy = normalize_energy(lattice_energy)

        # Find seams in windowed regions
        if method == 'dp' and is_cyclic:
            roi_seam = dp_seam_cyclic(lattice_energy, roi_range, direction=direction)
            pair_seam = dp_seam_cyclic(lattice_energy, pair_range, direction=direction)
        elif method == 'dp':
            roi_seam = dp_seam_windowed(lattice_energy, roi_range, direction=direction)
            pair_seam = dp_seam_windowed(lattice_energy, pair_range, direction=direction)
        else:
            roi_seam = greedy_seam_windowed(lattice_energy, roi_range, direction=direction,
                                            n_candidates=n_candidates)
            pair_seam = greedy_seam_windowed(lattice_energy, pair_range, direction=direction,
                                             n_candidates=n_candidates)

        roi_seam_interp = _interpolate_seam(roi_seam, n_map, cyclic=is_cyclic)
        pair_seam_interp = _interpolate_seam(pair_seam, n_map, cyclic=is_cyclic)

        # Combined shift: shrink ROI + expand pair
        combined_shift = torch.zeros_like(u_map)
        combined_shift = combined_shift + torch.where(
            u_map >= roi_seam_interp,
            torch.full_like(u_map, roi_sign),
            torch.zeros_like(u_map))
        combined_shift = combined_shift + torch.where(
            u_map > pair_seam_interp,
            torch.full_like(u_map, -roi_sign),
            torch.zeros_like(u_map))
        combined_shift = torch.where(valid_mask, combined_shift,
                                     torch.zeros_like(combined_shift))

        # Compose warp into source coordinate map
        grid = _shift_to_warp_grid(lattice, u_map, n_map, combined_shift, H, W)
        src_map = _sample_src_map(src_map, grid, valid_mask, x_grid, y_grid)

    # Single final interpolation from original image
    result = _src_map_to_image(image, src_map, H, W)
    result = torch.where(valid_mask_3d, result, image)

    if squeeze_output:
        result = result.squeeze(0)

    return result


def carve_with_comparison(image: torch.Tensor, lattice: Lattice2D,
                         n_seams: int, direction: str = 'vertical',
                         lattice_width: Optional[int] = None):
    """
    Perform both traditional and lattice-guided carving for comparison.
    """
    traditional_carved = carve_image_traditional(image, n_seams, direction)
    lattice_carved = carve_image_lattice_guided(
        image, lattice, n_seams, direction, lattice_width
    )
    return traditional_carved, lattice_carved
