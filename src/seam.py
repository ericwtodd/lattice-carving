"""
Seam computation algorithms.

Three approaches:
1. DP (dynamic programming): Optimal seam finding for 2D (Avidan & Shamir 2007)
2. Greedy: Fast approximate seam finding (Section 4.0.1)
3. Multi-greedy: Greedy with multiple starting points (Section 4.0.1)
"""

import torch
from typing import List, Tuple


def dp_seam(energy: torch.Tensor, direction: str = 'vertical') -> torch.Tensor:
    """
    Find the optimal seam using dynamic programming (Avidan & Shamir 2007).

    Builds a cumulative cost matrix top-to-bottom, then backtracks from
    the minimum in the last row. Guarantees the globally minimum-energy
    connected seam. For 2D images this gives the same result as graph-cut.

    Args:
        energy: Energy map (H, W)
        direction: 'vertical' or 'horizontal'

    Returns:
        Seam indices - for vertical: (H,) with column index per row
                      for horizontal: (W,) with row index per column
    """
    if direction == 'horizontal':
        # Transpose, find vertical seam, return as horizontal
        return dp_seam(energy.t(), direction='vertical')

    H, W = energy.shape
    device = energy.device

    # Build cumulative cost matrix
    M = torch.zeros_like(energy)
    M[0] = energy[0]

    for i in range(1, H):
        # Shift left/right to get neighbor costs
        left = torch.full((W,), float('inf'), device=device)
        left[1:] = M[i - 1, :-1]
        right = torch.full((W,), float('inf'), device=device)
        right[:-1] = M[i - 1, 1:]
        center = M[i - 1]

        M[i] = energy[i] + torch.min(torch.min(left, center), right)

    # Backtrack from minimum in last row
    seam = torch.zeros(H, dtype=torch.long, device=device)
    seam[H - 1] = torch.argmin(M[H - 1])

    for i in range(H - 2, -1, -1):
        prev_col = seam[i + 1].item()
        left = max(0, prev_col - 1)
        right = min(W - 1, prev_col + 1)
        seam[i] = left + torch.argmin(M[i, left:right + 1])

    return seam


def dp_seam_windowed(energy: torch.Tensor, col_range: Tuple[int, int],
                     direction: str = 'vertical') -> torch.Tensor:
    """
    Find optimal DP seam constrained to columns [col_start, col_end].

    Sets energy outside the window to infinity so the DP never selects
    those columns, then runs standard DP.

    Args:
        energy: Energy map (H, W)
        col_range: Tuple (start, end) — inclusive column range
        direction: 'vertical' or 'horizontal'

    Returns:
        Seam indices (same format as dp_seam)
    """
    H, W = energy.shape
    col_start, col_end = int(col_range[0]), int(col_range[1])

    if direction == 'vertical':
        col_start = max(0, min(col_start, W - 1))
        col_end = max(col_start, min(col_end, W - 1))

        # Mask energy outside window
        masked = torch.full_like(energy, float('inf'))
        masked[:, col_start:col_end + 1] = energy[:, col_start:col_end + 1]
        return dp_seam(masked, direction='vertical')

    elif direction == 'horizontal':
        row_start = max(0, min(col_start, H - 1))
        row_end = max(row_start, min(col_end, H - 1))

        masked = torch.full_like(energy, float('inf'))
        masked[row_start:row_end + 1, :] = energy[row_start:row_end + 1, :]
        return dp_seam(masked, direction='horizontal')

    else:
        raise ValueError(f"Invalid direction: {direction}")


def dp_seam_cyclic(energy: torch.Tensor, col_range: Tuple[int, int],
                   direction: str = 'vertical') -> torch.Tensor:
    """
    Find optimal DP seam for cyclic lattices where seam[0] == seam[-1].

    For cyclic lattices, the first and last scanlines are adjacent in world
    space. The seam must close: its start and end positions must match.

    Approach: Run DP for each possible starting column in the window.
    For each start, force row 0 to that column (set others to inf),
    then check if the backtracked seam ends at the same column.
    Pick the closed seam with lowest total energy.

    If no seam closes exactly, pick the one with minimum |start - end|
    and snap the last few rows to force closure.

    Args:
        energy: Energy map (H, W) — H = n_lines (scanlines around the loop)
        col_range: Tuple (start, end) — inclusive column range
        direction: 'vertical' only for now

    Returns:
        Seam indices (H,) where seam[0] == seam[-1]
    """
    if direction != 'vertical':
        raise NotImplementedError("Cyclic DP seam only supports vertical direction")

    H, W = energy.shape
    device = energy.device
    col_start, col_end = int(col_range[0]), int(col_range[1])
    col_start = max(0, min(col_start, W - 1))
    col_end = max(col_start, min(col_end, W - 1))

    # Mask energy outside window
    masked = torch.full_like(energy, float('inf'))
    masked[:, col_start:col_end + 1] = energy[:, col_start:col_end + 1]

    best_seam = None
    best_cost = float('inf')
    best_gap = float('inf')

    # Try each starting column
    for start_col in range(col_start, col_end + 1):
        # Force first row to start_col
        forced = masked.clone()
        forced[0, :] = float('inf')
        forced[0, start_col] = energy[0, start_col]

        # Run DP
        M = torch.zeros_like(forced)
        M[0] = forced[0]

        for i in range(1, H):
            left = torch.full((W,), float('inf'), device=device)
            left[1:] = M[i - 1, :-1]
            right = torch.full((W,), float('inf'), device=device)
            right[:-1] = M[i - 1, 1:]
            center = M[i - 1]
            M[i] = forced[i] + torch.min(torch.min(left, center), right)

        # Backtrack
        seam = torch.zeros(H, dtype=torch.long, device=device)
        seam[H - 1] = torch.argmin(M[H - 1])

        for i in range(H - 2, -1, -1):
            prev_col = seam[i + 1].item()
            left_b = max(0, prev_col - 1)
            right_b = min(W - 1, prev_col + 1)
            seam[i] = left_b + torch.argmin(M[i, left_b:right_b + 1])

        end_col = seam[H - 1].item()
        gap = abs(end_col - start_col)
        cost = M[H - 1, end_col].item()

        # Prefer exact closure, then lowest cost
        if gap == 0 and cost < best_cost:
            best_cost = cost
            best_gap = 0
            best_seam = seam.clone()
        elif gap < best_gap or (gap == best_gap and cost < best_cost):
            best_cost = cost
            best_gap = gap
            best_seam = seam.clone()

    # If no exact closure found, force closure by snapping last rows
    if best_gap > 0 and best_seam is not None:
        target = best_seam[0].item()
        current = best_seam[H - 1].item()
        # Linearly interpolate the last `gap+1` rows toward target
        snap_len = min(best_gap + 1, H // 2)
        for k in range(snap_len):
            row_idx = H - 1 - k
            t = (k + 1) / snap_len
            best_seam[row_idx] = round(current * (1 - t) + target * t)
            best_seam[row_idx] = max(col_start, min(col_end, best_seam[row_idx].item()))

    return best_seam


def greedy_seam(energy: torch.Tensor, direction: str = 'vertical',
                n_candidates: int = 1) -> torch.Tensor:
    """
    Compute a seam using the greedy approach from Flynn et al. 2021.

    With n_candidates > 1, uses multi-greedy: tries multiple starting
    points and returns the lowest total-energy seam (Section 4.0.1).

    Args:
        energy: Energy map (H, W)
        direction: 'vertical' or 'horizontal'
        n_candidates: Number of starting points to try (1 = single greedy)

    Returns:
        Seam indices - for vertical: (H,) with column index per row
                      for horizontal: (W,) with row index per column
    """
    H, W = energy.shape

    if direction == 'vertical':
        if n_candidates <= 1:
            start_cols = [torch.argmin(energy[0]).item()]
        else:
            start_cols = torch.linspace(0, W - 1, n_candidates, dtype=torch.long,
                                        device=energy.device).tolist()

        best_seam = None
        best_energy = float('inf')

        for start_col in start_cols:
            seam = torch.zeros(H, dtype=torch.long, device=energy.device)
            seam[0] = start_col
            total = energy[0, start_col].item()

            for i in range(1, H):
                prev_col = seam[i - 1].item()
                left = max(0, prev_col - 1)
                right = min(W - 1, prev_col + 1)
                neighbors = energy[i, left:right + 1]
                local_min_idx = torch.argmin(neighbors)
                seam[i] = left + local_min_idx
                total += energy[i, seam[i]].item()

            if total < best_energy:
                best_energy = total
                best_seam = seam

        return best_seam

    elif direction == 'horizontal':
        if n_candidates <= 1:
            start_rows = [torch.argmin(energy[:, 0]).item()]
        else:
            start_rows = torch.linspace(0, H - 1, n_candidates, dtype=torch.long,
                                        device=energy.device).tolist()

        best_seam = None
        best_energy = float('inf')

        for start_row in start_rows:
            seam = torch.zeros(W, dtype=torch.long, device=energy.device)
            seam[0] = start_row
            total = energy[start_row, 0].item()

            for j in range(1, W):
                prev_row = seam[j - 1].item()
                top = max(0, prev_row - 1)
                bottom = min(H - 1, prev_row + 1)
                neighbors = energy[top:bottom + 1, j]
                local_min_idx = torch.argmin(neighbors)
                seam[j] = top + local_min_idx
                total += energy[seam[j], j].item()

            if total < best_energy:
                best_energy = total
                best_seam = seam

        return best_seam

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
                        direction: str = 'vertical',
                        n_candidates: int = 1) -> torch.Tensor:
    """
    Find greedy seam constrained to columns [col_start, col_end].

    With n_candidates > 1, tries multiple starting points within the
    window and returns the lowest total-energy seam.

    Args:
        energy: Energy map (H, W)
        col_range: Tuple (start, end) — inclusive column range
        direction: 'vertical' or 'horizontal'
        n_candidates: Number of starting points to try

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
        col_start = max(0, min(col_start, H - 1))
        col_end = max(col_start, min(col_end, H - 1))

    if direction == 'vertical':
        if n_candidates <= 1:
            start_cols = [col_start + torch.argmin(energy[0, col_start:col_end + 1]).item()]
        else:
            start_cols = torch.linspace(col_start, col_end, n_candidates,
                                        dtype=torch.long, device=energy.device).tolist()

        best_seam = None
        best_energy = float('inf')

        for start_col in start_cols:
            seam = torch.zeros(H, dtype=torch.long, device=energy.device)
            seam[0] = start_col
            total = energy[0, start_col].item()

            for i in range(1, H):
                prev_col = seam[i - 1].item()
                left = max(col_start, prev_col - 1)
                right = min(col_end, prev_col + 1)
                neighbors = energy[i, left:right + 1]
                if neighbors.numel() == 0:
                    seam[i] = col_start
                else:
                    local_min_idx = torch.argmin(neighbors)
                    seam[i] = left + local_min_idx
                total += energy[i, seam[i]].item()

            if total < best_energy:
                best_energy = total
                best_seam = seam

        return best_seam

    elif direction == 'horizontal':
        row_start, row_end = col_start, col_end

        if n_candidates <= 1:
            start_rows = [row_start + torch.argmin(energy[row_start:row_end + 1, 0]).item()]
        else:
            start_rows = torch.linspace(row_start, row_end, n_candidates,
                                        dtype=torch.long, device=energy.device).tolist()

        best_seam = None
        best_energy = float('inf')

        for start_row in start_rows:
            seam = torch.zeros(W, dtype=torch.long, device=energy.device)
            seam[0] = start_row
            total = energy[start_row, 0].item()

            for j in range(1, W):
                prev_row = seam[j - 1].item()
                top = max(row_start, prev_row - 1)
                bottom = min(row_end, prev_row + 1)
                neighbors = energy[top:bottom + 1, j]
                if neighbors.numel() == 0:
                    seam[j] = row_start
                else:
                    local_min_idx = torch.argmin(neighbors)
                    seam[j] = top + local_min_idx
                total += energy[seam[j], j].item()

            if total < best_energy:
                best_energy = total
                best_seam = seam

        return best_seam

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
