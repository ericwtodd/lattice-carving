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
                        threshold: float = 3.0) -> torch.Tensor:
    """Compute a boolean mask of pixels that lie inside the lattice region.

    Uses a roundtrip test: forward -> inverse -> compare with original position.
    Pixels inside the lattice region roundtrip accurately; distant pixels don't.

    Args:
        lattice: Lattice2D structure
        u_map: (H, W) u coordinates from forward mapping
        n_map: (H, W) n coordinates from forward mapping
        H, W: image dimensions
        device: torch device
        threshold: max roundtrip error in pixels to be considered valid

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
    return roundtrip_err < threshold


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


def carve_image_lattice_guided(image: torch.Tensor, lattice: Lattice2D,
                                n_seams: int, direction: str = 'vertical',
                                lattice_width: Optional[int] = None,
                                roi_bounds: Optional[Tuple[float, float]] = None,
                                n_candidates: int = 1,
                                method: str = 'dp') -> torch.Tensor:
    """
    Lattice-guided seam carving via iterative warping (Section 3.3, Fig. 8).

    Each iteration: compute energy on the current image, find a seam in lattice
    space, then warp the current image by one seam step. This correctly handles
    the composition of multiple g* mappings (the paper's approach).
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
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, device)
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)

    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic
    original_image = image.clone()
    current_image = image.clone()

    for i in range(n_seams):
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

        # Single-step shift: compare original lattice coords, not adjusted
        single_shift = torch.where(u_map >= seam_interp,
                                   torch.ones_like(u_map),
                                   torch.zeros_like(u_map))
        single_shift = torch.where(valid_mask, single_shift,
                                   torch.zeros_like(single_shift))

        # Warp from current image (not original) — paper's iterative approach
        warped = _warp_and_resample(current_image, lattice, u_map, n_map,
                                    single_shift)
        current_image = torch.where(valid_mask_3d, warped, current_image)

    # ROI masking — pixels outside lattice bounds stay unchanged
    current_image = torch.where(valid_mask_3d, current_image, original_image)

    if roi_bounds is not None:
        u_min, u_max = roi_bounds
        roi_valid = (u_map >= u_min) & (u_map <= u_max)
        roi_valid = roi_valid & (n_map >= 0) & (n_map <= lattice.n_lines - 1)
        roi_valid_3d = roi_valid.unsqueeze(0).expand_as(current_image)
        current_image = torch.where(roi_valid_3d, current_image, original_image)

    if squeeze_output:
        current_image = current_image.squeeze(0)

    return current_image


def carve_seam_pairs(image: torch.Tensor, lattice: Lattice2D,
                     n_seams: int, roi_range: Tuple[int, int],
                     pair_range: Tuple[int, int],
                     direction: str = 'vertical',
                     lattice_width: Optional[int] = None,
                     n_candidates: int = 1,
                     method: str = 'dp',
                     mode: str = 'shrink') -> torch.Tensor:
    """
    Seam pair carving for local region resizing without changing global boundaries.

    Uses iterative warping: each iteration reads from the current image state
    and applies a single-step combined shift (ROI + pair), correctly handling
    the composition of multiple g* mappings. See Section 3.6 of Flynn et al. 2021.

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
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, device)
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)

    is_cyclic = hasattr(lattice, '_cyclic') and lattice._cyclic
    original_image = image.clone()
    current_image = image.clone()

    for i in range(n_seams):
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

        # Single-step combined shift using original lattice coords
        combined_shift = torch.zeros_like(u_map)
        combined_shift = combined_shift + torch.where(
            u_map >= roi_seam_interp,
            torch.full_like(u_map, roi_sign),
            torch.zeros_like(u_map))
        combined_shift = combined_shift + torch.where(
            u_map > pair_seam_interp,
            torch.full_like(u_map, -roi_sign),
            torch.zeros_like(u_map))
        # Zero out shift for pixels outside the lattice region
        combined_shift = torch.where(valid_mask, combined_shift,
                                     torch.zeros_like(combined_shift))

        # Warp from current image (iterative, not cumulative from original)
        warped = _warp_and_resample(current_image, lattice, u_map, n_map,
                                    combined_shift)
        current_image = torch.where(valid_mask_3d, warped, current_image)

    # Pixels outside the lattice stay unchanged from original
    current_image = torch.where(valid_mask_3d, current_image, original_image)

    if squeeze_output:
        current_image = current_image.squeeze(0)

    return current_image


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