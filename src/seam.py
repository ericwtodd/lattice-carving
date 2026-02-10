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
