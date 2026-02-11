"""
Energy functions for seam carving.

The energy function determines which pixels/voxels are "important".
Low-energy seams are preferred for removal.

For images, we use gradient magnitude (Avidan & Shamir 2007).
"""

import torch
import torch.nn.functional as F


def gradient_magnitude_energy(image: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient magnitude energy for an image (paper equation 6).

    Uses L1 norm of image gradients:
    E_I(i,j) = ||∂/∂x I(i,j)|| + ||∂/∂y I(i,j)||

    This is the standard energy function from Avidan & Shamir 2007.

    Args:
        image: RGB image tensor (C, H, W) or grayscale (H, W)

    Returns:
        Energy map (H, W)
    """
    if image.dim() == 2:
        # Grayscale image
        gray = image.unsqueeze(0)
    elif image.dim() == 3:
        # Convert RGB to grayscale
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            gray = gray.unsqueeze(0)
        else:
            gray = image

    # Compute gradients using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=gray.dtype, device=gray.device)
    sobel_x = sobel_x.view(1, 1, 3, 3)

    sobel_y = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=gray.dtype, device=gray.device)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Add batch dimension if needed
    if gray.dim() == 3:
        gray = gray.unsqueeze(0)

    # Compute gradients with padding
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # Gradient magnitude using L1 norm (paper equation 6)
    # E_I(i,j) = ||∂/∂x I|| + ||∂/∂y I||
    energy = torch.abs(grad_x) + torch.abs(grad_y)
    energy = energy.squeeze()

    return energy


def normalize_energy(energy: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Remap energy to [0, 1] range (paper page 10).

    Paper: "we also remap our energy functions to always be between 0 and 1"

    This is a monotonic transform so seam positions are unchanged.

    Args:
        energy: Energy map (H, W)
        eps: Small value to avoid division by zero

    Returns:
        Normalized energy map in [0, 1]
    """
    e_min = energy.min()
    e_max = energy.max()
    return (energy - e_min) / (e_max - e_min + eps)


def forward_energy(image: torch.Tensor) -> torch.Tensor:
    """Forward energy (Rubinstein et al. 2008).

    Considers the cost of new edges introduced by seam removal, rather
    than just the energy of the pixel being removed. For each pixel,
    computes three transition costs based on how removing it would
    change its neighbors' relationships.

    For vertical seams, the three costs at pixel (i, j) are:
      C_U = |I(i, j+1) - I(i, j-1)|
      C_L = C_U + |I(i-1, j) - I(i, j-1)|
      C_R = C_U + |I(i-1, j) - I(i, j+1)|

    The minimum cost path is found via dynamic programming.

    Args:
        image: RGB image tensor (C, H, W) or grayscale (H, W)

    Returns:
        Forward energy map (H, W) — cumulative minimum cost at each pixel
    """
    if image.dim() == 2:
        gray = image.clone()
    elif image.dim() == 3:
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        else:
            gray = image.squeeze(0)

    H, W = gray.shape

    # Compute neighbor images via indexing (avoids F.pad dimension issues)
    # I(i, j-1): shift right (left neighbor)
    left = torch.zeros_like(gray)
    left[:, 1:] = gray[:, :-1]
    left[:, 0] = gray[:, 0]

    # I(i, j+1): shift left (right neighbor)
    right = torch.zeros_like(gray)
    right[:, :-1] = gray[:, 1:]
    right[:, -1] = gray[:, -1]

    # I(i-1, j): shift down (above neighbor)
    above = torch.zeros_like(gray)
    above[1:, :] = gray[:-1, :]
    above[0, :] = gray[0, :]

    # Three transition costs (Rubinstein et al. 2008)
    C_U = torch.abs(right - left)
    C_L = C_U + torch.abs(above - left)
    C_R = C_U + torch.abs(above - right)

    # Dynamic programming: accumulate minimum cost top to bottom
    M = torch.zeros_like(gray)
    M[0] = C_U[0]

    for i in range(1, H):
        # Shifted versions of previous row's cumulative cost
        M_prev = M[i - 1]
        M_left = torch.full((W,), float('inf'), device=gray.device, dtype=gray.dtype)
        M_left[1:] = M_prev[:-1]
        M_right = torch.full((W,), float('inf'), device=gray.device, dtype=gray.dtype)
        M_right[:-1] = M_prev[1:]

        # Three options: come from above-left, above, or above-right
        cost_from_left = M_left + C_L[i]
        cost_from_center = M_prev + C_U[i]
        cost_from_right = M_right + C_R[i]

        M[i] = torch.min(torch.min(cost_from_left, cost_from_center), cost_from_right)

    return M
