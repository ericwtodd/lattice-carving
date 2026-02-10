"""
Basic seam carving example on the bagel image.

This demonstrates traditional (rectangular) seam carving before
we move to the lattice-guided approach.
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from energy import gradient_magnitude_energy
from seam import greedy_seam, multi_greedy_seam, remove_seam


def load_image(path: str, device='cpu'):
    """Load image and convert to torch tensor."""
    img = Image.open(path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)
    return img_tensor


def save_image(tensor: torch.Tensor, path: str):
    """Save torch tensor as image."""
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)
    print(f"Saved: {path}")


def visualize_seam(image: torch.Tensor, seam: torch.Tensor,
                  direction: str = 'vertical'):
    """Visualize a seam on an image."""
    img_vis = image.clone()

    if direction == 'vertical':
        for i, col in enumerate(seam):
            img_vis[:, i, col] = torch.tensor([1.0, 0.0, 0.0],
                                             device=image.device)
    else:
        for j, row in enumerate(seam):
            img_vis[:, row, j] = torch.tensor([1.0, 0.0, 0.0],
                                             device=image.device)

    return img_vis


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load bagel image
    print("Loading image...")
    image = load_image('../bagel.jpg', device=device)
    C, H, W = image.shape
    print(f"Image shape: {C} x {H} x {W}")

    # Compute energy
    print("Computing energy...")
    energy = gradient_magnitude_energy(image)

    # Compute seam (single greedy)
    print("Computing greedy seam...")
    seam = greedy_seam(energy, direction='vertical')

    # Visualize seam
    img_with_seam = visualize_seam(image, seam, direction='vertical')
    save_image(img_with_seam, '../output/bagel_with_seam.png')

    # Remove seams iteratively
    print("Carving image (removing 100 seams)...")
    carved_image = image.clone()
    n_seams = 100

    for i in range(n_seams):
        # Recompute energy for current image
        energy = gradient_magnitude_energy(carved_image)

        # Find and remove seam
        seam = greedy_seam(energy, direction='vertical')
        carved_image = remove_seam(carved_image, seam, direction='vertical')

        if (i + 1) % 20 == 0:
            print(f"  Removed {i + 1}/{n_seams} seams, size: {carved_image.shape}")

    save_image(carved_image, '../output/bagel_carved.png')

    # Compare with multi-greedy approach
    print("\nComparing single-greedy vs multi-greedy...")
    energy = gradient_magnitude_energy(image)

    # Multi-greedy with 8 candidate seams
    seams = multi_greedy_seam(energy, n_seams=8, direction='vertical')
    best_seam = seams[0]

    img_with_best_seam = visualize_seam(image, best_seam, direction='vertical')
    save_image(img_with_best_seam, '../output/bagel_with_multigreedy_seam.png')

    print("\nDone! Check the output/ directory for results.")


if __name__ == '__main__':
    main()
