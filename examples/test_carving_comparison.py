"""
Test and compare traditional vs lattice-guided seam carving on synthetic images.

This validates that our implementation works correctly by showing:
1. Traditional carving distorts features
2. Lattice-guided carving preserves features along lattice structure
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from energy import gradient_magnitude_energy
from seam import greedy_seam, remove_seam


def load_test_image(name, device='cpu'):
    """Load a test image."""
    path = f'../output/test_{name}.png'
    img = Image.open(path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).to(device)
    return img_tensor


def save_image(tensor, path):
    """Save tensor as image."""
    img_array = tensor.permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(path)


def traditional_carving(image, n_seams=50, direction='vertical'):
    """
    Apply traditional rectangular seam carving.

    This will distort curved features like circles and rivers.
    """
    carved = image.clone()

    for i in range(n_seams):
        energy = gradient_magnitude_energy(carved)
        seam = greedy_seam(energy, direction=direction)
        carved = remove_seam(carved, seam, direction=direction)

        if (i + 1) % 10 == 0:
            print(f"    Removed {i + 1}/{n_seams} seams")

    return carved


def test_river():
    """Test river image - traditional carving should distort the river."""
    print("\n" + "="*60)
    print("Test: River Image")
    print("="*60)
    print("Expected:")
    print("  - Traditional: River gets distorted/wavy")
    print("  - Lattice-guided: River preserved (TODO)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image('river', device=device)

    print("\nApplying traditional vertical carving...")
    carved_traditional = traditional_carving(image, n_seams=50, direction='vertical')

    # Save results
    save_image(image, '../output/river_original.png')
    save_image(carved_traditional, '../output/river_traditional_carved.png')

    print(f"Original size: {image.shape}")
    print(f"Carved size: {carved_traditional.shape}")
    print("Saved: river_original.png, river_traditional_carved.png")


def test_circle():
    """Test circle image - traditional carving should squish circle into oval."""
    print("\n" + "="*60)
    print("Test: Circle Image")
    print("="*60)
    print("Expected:")
    print("  - Traditional: Circle squished into horizontal oval")
    print("  - Lattice-guided: Circle stays circular (TODO)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image('circle', device=device)

    print("\nApplying traditional vertical carving...")
    carved_traditional = traditional_carving(image, n_seams=80, direction='vertical')

    # Save results
    save_image(image, '../output/circle_original.png')
    save_image(carved_traditional, '../output/circle_traditional_carved.png')

    print(f"Original size: {image.shape}")
    print(f"Carved size: {carved_traditional.shape}")
    print("Saved: circle_original.png, circle_traditional_carved.png")


def test_concentric_circles():
    """Test concentric circles - should see clear distortion."""
    print("\n" + "="*60)
    print("Test: Concentric Circles")
    print("="*60)
    print("Expected:")
    print("  - Traditional: Circles become ovals")
    print("  - Lattice-guided: Circles stay circular (TODO)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image('concentric_circles', device=device)

    print("\nApplying traditional vertical carving...")
    carved_traditional = traditional_carving(image, n_seams=80, direction='vertical')

    # Save results
    save_image(image, '../output/concentric_original.png')
    save_image(carved_traditional, '../output/concentric_traditional_carved.png')

    print(f"Original size: {image.shape}")
    print(f"Carved size: {carved_traditional.shape}")
    print("Saved: concentric_original.png, concentric_traditional_carved.png")


def test_grid():
    """Test grid - shows distortion patterns clearly."""
    print("\n" + "="*60)
    print("Test: Grid Image")
    print("="*60)
    print("Expected:")
    print("  - Traditional: Grid lines get wavy/distorted")
    print("  - Lattice-guided: Depends on lattice structure")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = load_test_image('grid', device=device)

    print("\nApplying traditional vertical carving...")
    carved_traditional = traditional_carving(image, n_seams=80, direction='vertical')

    # Save results
    save_image(image, '../output/grid_original.png')
    save_image(carved_traditional, '../output/grid_traditional_carved.png')

    print(f"Original size: {image.shape}")
    print(f"Carved size: {carved_traditional.shape}")
    print("Saved: grid_original.png, grid_traditional_carved.png")


def create_comparison_figure():
    """Create a comparison figure showing all results."""
    print("\nCreating comparison figure...")

    test_names = ['river', 'circle', 'concentric', 'grid']
    fig, axes = plt.subplots(len(test_names), 2, figsize=(12, 16))

    for i, name in enumerate(test_names):
        # Original
        original_path = f'../output/{name}_original.png'
        if name == 'concentric':
            original_path = '../output/concentric_original.png'

        try:
            original_img = Image.open(original_path)
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title(f'{name.capitalize()} - Original')
            axes[i, 0].axis('off')
        except:
            axes[i, 0].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[i, 0].axis('off')

        # Traditional carved
        carved_path = f'../output/{name}_traditional_carved.png'
        try:
            carved_img = Image.open(carved_path)
            axes[i, 1].imshow(carved_img)
            axes[i, 1].set_title(f'{name.capitalize()} - Traditional Carved')
            axes[i, 1].axis('off')
        except:
            axes[i, 1].text(0.5, 0.5, 'Not found', ha='center', va='center')
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('../output/carving_comparison.png', dpi=150)
    print("Saved: ../output/carving_comparison.png")


def main():
    print("="*60)
    print("Testing Seam Carving on Synthetic Images")
    print("="*60)

    # Run tests
    test_river()
    test_circle()
    test_concentric_circles()
    test_grid()

    # Create comparison figure
    create_comparison_figure()

    print("\n" + "="*60)
    print("Tests Complete!")
    print("="*60)
    print("\nNext step: Implement lattice-guided carving to preserve features")
    print("Check ../output/ for results")


if __name__ == '__main__':
    main()
