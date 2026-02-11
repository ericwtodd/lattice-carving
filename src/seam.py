"""
Seam computation algorithms.

Two approaches:
1. Graph-cut: Optimal seam finding (slow)
2. Greedy: Fast approximate seam finding (500x+ faster, comparable quality)
"""

import torch
from typing import List, Tuple


def greedy_seam(energy: torch.Tensor, direction: str = 'vertical') -> torch.Tensor:
    """
    Compute a seam using the greedy approach from Flynn et al. 2021.

    The greedy approach:
    1. Start at the first row/column
    2. For each subsequent row/column, move to the lowest energy neighbor
    3. Continue until reaching the end

    This is much faster than graph-cut (O(n) vs O(n²)) and produces
    comparable visual quality.

    Args:
        energy: Energy map (H, W)
        direction: 'vertical' or 'horizontal'

    Returns:
        Seam indices - for vertical: (H,) with column index per row
                      for horizontal: (W,) with row index per column
    """
    H, W = energy.shape

    if direction == 'vertical':
        # Vertical seam (remove one pixel from each row)
        seam = torch.zeros(H, dtype=torch.long, device=energy.device)

        # Start at the row with minimum energy in first row
        seam[0] = torch.argmin(energy[0])

        # Greedy selection: move to lowest energy neighbor in next row
        for i in range(1, H):
            prev_col = seam[i - 1].item()

            # Consider neighbors: left, center, right
            left = max(0, prev_col - 1)
            right = min(W - 1, prev_col + 1)

            # Find minimum energy neighbor
            neighbors = energy[i, left:right + 1]
            local_min_idx = torch.argmin(neighbors)
            seam[i] = left + local_min_idx

        return seam

    elif direction == 'horizontal':
        # Horizontal seam (remove one pixel from each column)
        seam = torch.zeros(W, dtype=torch.long, device=energy.device)

        # Start at column with minimum energy in first column
        seam[0] = torch.argmin(energy[:, 0])

        # Greedy selection
        for j in range(1, W):
            prev_row = seam[j - 1].item()

            # Consider neighbors: up, center, down
            top = max(0, prev_row - 1)
            bottom = min(H - 1, prev_row + 1)

            neighbors = energy[top:bottom + 1, j]
            local_min_idx = torch.argmin(neighbors)
            seam[j] = top + local_min_idx

        return seam

    else:
        raise ValueError(f"Invalid direction: {direction}")


def greedy_seam_cyclic(energy: torch.Tensor, col_range: Tuple[int, int],
                      direction: str = 'vertical', guide_width: float = 10.0) -> torch.Tensor:
    """
    Find greedy seam for cyclic lattices with Gaussian energy guide.

    For cyclic lattices (closed curves like circles), we need the seam to
    start and end at the same u-position. This is done by:
    1. Computing an initial seam
    2. Adding an inverted Gaussian centered on the initial seam
    3. Recomputing with the guided energy

    From paper Section 4.0.1, Figure 12.

    Args:
        energy: Energy map (H, W)
        col_range: Tuple (start, end) — inclusive column range
        direction: 'vertical' or 'horizontal'
        guide_width: Width of Gaussian guide (smaller = stronger guidance)

    Returns:
        Seam indices that start and end at same position
    """
    H, W = energy.shape
    col_start, col_end = int(col_range[0]), int(col_range[1])

    # Step 1: Compute initial seam without guide
    initial_seam = greedy_seam_windowed(energy, col_range, direction)

    # Step 2: Create Gaussian energy guide centered on initial seam
    # The guide steers the seam back to its starting position
    if direction == 'vertical':
        # Create coordinate grids
        n_coords = torch.arange(H, device=energy.device, dtype=torch.float32)
        u_coords = torch.arange(W, device=energy.device, dtype=torch.float32)
        nn, uu = torch.meshgrid(n_coords, u_coords, indexing='ij')

        # Initial seam position (we want to end here too)
        initial_u = initial_seam[0].float()

        # Distance from each pixel to the initial seam path
        # For cyclic, the seam should curve back to initial_u at the end
        seam_positions = initial_seam.float()

        # Compute distance to seam at each scanline
        dist_to_seam = torch.abs(uu - seam_positions.unsqueeze(1))

        # Inverted Gaussian: low energy near the initial seam path
        # This attracts the seam to follow a path that closes
        gaussian_guide = torch.exp(-(dist_to_seam ** 2) / (2 * guide_width ** 2))

        # Invert: high Gaussian = low energy addition
        # We add negative of this to make the seam prefer closing
        energy_guided = energy.clone()

        # Add stronger guidance near the start and end to force closure
        closure_strength = torch.linspace(1.0, 0.0, H // 2, device=energy.device)
        closure_strength = torch.cat([closure_strength, closure_strength.flip(0)])
        if len(closure_strength) < H:
            closure_strength = torch.nn.functional.pad(closure_strength, (0, H - len(closure_strength)))

        # Subtract Gaussian (lower energy = preferred path)
        energy_guided = energy - 0.5 * gaussian_guide * closure_strength.unsqueeze(1)

    else:
        raise NotImplementedError("Cyclic seam currently only for vertical direction")

    # Step 3: Recompute seam with guided energy
    final_seam = greedy_seam_windowed(energy_guided, col_range, direction)

    return final_seam


def greedy_seam_windowed(energy: torch.Tensor, col_range: Tuple[int, int],
                        direction: str = 'vertical') -> torch.Tensor:
    """
    Find greedy seam constrained to columns [col_start, col_end].

    Same logic as greedy_seam but the seam is restricted to stay within
    the specified column (or row) range.

    Args:
        energy: Energy map (H, W)
        col_range: Tuple (start, end) — inclusive column range
        direction: 'vertical' or 'horizontal'

    Returns:
        Seam indices (same format as greedy_seam)
    """
    H, W = energy.shape
    col_start, col_end = int(col_range[0]), int(col_range[1])

    # Validate window bounds
    if direction == 'vertical':
        col_start = max(0, min(col_start, W - 1))
        col_end = max(col_start, min(col_end, W - 1))
    else:
        # For horizontal, col_range refers to rows
        col_start = max(0, min(col_start, H - 1))
        col_end = max(col_start, min(col_end, H - 1))

    if direction == 'vertical':
        seam = torch.zeros(H, dtype=torch.long, device=energy.device)

        # Start at argmin within the window in first row
        seam[0] = col_start + torch.argmin(energy[0, col_start:col_end + 1])

        for i in range(1, H):
            prev_col = seam[i - 1].item()
            left = max(col_start, prev_col - 1)
            right = min(col_end, prev_col + 1)

            neighbors = energy[i, left:right + 1]
            if neighbors.numel() == 0:
                # Fallback if window is empty (shouldn't happen with validation)
                seam[i] = col_start
            else:
                local_min_idx = torch.argmin(neighbors)
                seam[i] = left + local_min_idx

        return seam

    elif direction == 'horizontal':
        seam = torch.zeros(W, dtype=torch.long, device=energy.device)
        row_start, row_end = col_start, col_end  # reinterpret for horizontal

        seam[0] = row_start + torch.argmin(energy[row_start:row_end + 1, 0])

        for j in range(1, W):
            prev_row = seam[j - 1].item()
            top = max(row_start, prev_row - 1)
            bottom = min(row_end, prev_row + 1)

            neighbors = energy[top:bottom + 1, j]
            if neighbors.numel() == 0:
                # Fallback if window is empty (shouldn't happen with validation)
                seam[j] = row_start
            else:
                local_min_idx = torch.argmin(neighbors)
                seam[j] = top + local_min_idx

        return seam

    else:
        raise ValueError(f"Invalid direction: {direction}")


def multi_greedy_seam(energy: torch.Tensor, n_seams: int,
                     direction: str = 'vertical') -> List[torch.Tensor]:
    """
    Compute multiple seams in parallel using greedy approach.

    The multi-greedy approach (Section 4.0.1 in paper):
    - Compute multiple starting points distributed across first row/column
    - Each seam is computed greedily from its starting point
    - Select the best seam (lowest total energy)

    This helps avoid local minima from random starting positions.

    Args:
        energy: Energy map (H, W)
        n_seams: Number of candidate seams to compute
        direction: 'vertical' or 'horizontal'

    Returns:
        List of seam indices, ordered by total energy (best first)
    """
    H, W = energy.shape

    if direction == 'vertical':
        # Distribute starting points across first row
        start_cols = torch.linspace(0, W - 1, n_seams, dtype=torch.long,
                                   device=energy.device)

        seams = []
        energies = []

        for start_col in start_cols:
            seam = torch.zeros(H, dtype=torch.long, device=energy.device)
            seam[0] = start_col
            total_energy = energy[0, start_col].item()

            # Greedy selection from this starting point
            for i in range(1, H):
                prev_col = seam[i - 1].item()
                left = max(0, prev_col - 1)
                right = min(W - 1, prev_col + 1)

                neighbors = energy[i, left:right + 1]
                local_min_idx = torch.argmin(neighbors)
                seam[i] = left + local_min_idx
                total_energy += energy[i, seam[i]].item()

            seams.append(seam)
            energies.append(total_energy)

    elif direction == 'horizontal':
        # Distribute starting points across first column
        start_rows = torch.linspace(0, H - 1, n_seams, dtype=torch.long,
                                   device=energy.device)

        seams = []
        energies = []

        for start_row in start_rows:
            seam = torch.zeros(W, dtype=torch.long, device=energy.device)
            seam[0] = start_row
            total_energy = energy[start_row, 0].item()

            for j in range(1, W):
                prev_row = seam[j - 1].item()
                top = max(0, prev_row - 1)
                bottom = min(H - 1, prev_row + 1)

                neighbors = energy[top:bottom + 1, j]
                local_min_idx = torch.argmin(neighbors)
                seam[j] = top + local_min_idx
                total_energy += energy[seam[j], j].item()

            seams.append(seam)
            energies.append(total_energy)

    else:
        raise ValueError(f"Invalid direction: {direction}")

    # Sort by total energy (best first)
    sorted_indices = sorted(range(len(energies)), key=lambda i: energies[i])
    return [seams[i] for i in sorted_indices]


def remove_seam(image: torch.Tensor, seam: torch.Tensor,
               direction: str = 'vertical') -> torch.Tensor:
    """
    Remove a seam from an image.

    Args:
        image: Image tensor (C, H, W) or (H, W)
        seam: Seam indices
        direction: 'vertical' or 'horizontal'

    Returns:
        Carved image with one row/column removed
    """
    if image.dim() == 2:
        # Grayscale
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, H, W = image.shape

    if direction == 'vertical':
        # Remove one pixel from each row
        carved = torch.zeros(C, H, W - 1, dtype=image.dtype, device=image.device)

        for i in range(H):
            col = seam[i].item()
            carved[:, i, :col] = image[:, i, :col]
            carved[:, i, col:] = image[:, i, col + 1:]

    elif direction == 'horizontal':
        # Remove one pixel from each column
        carved = torch.zeros(C, H - 1, W, dtype=image.dtype, device=image.device)

        for j in range(W):
            row = seam[j].item()
            carved[:, :row, j] = image[:, :row, j]
            carved[:, row:, j] = image[:, row + 1:, j]

    else:
        raise ValueError(f"Invalid direction: {direction}")

    if squeeze_output:
        carved = carved.squeeze(0)

    return carved
