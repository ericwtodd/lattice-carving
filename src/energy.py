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


def forward_energy(image: torch.Tensor) -> torch.Tensor:
    """
    Forward energy function that considers the cost of introducing edges.

    This is more sophisticated than gradient magnitude and can produce
    better results by considering what happens after seam removal.

    TODO: Implement forward energy from Rubinstein et al. 2008
    """
    raise NotImplementedError("Forward energy not yet implemented")
