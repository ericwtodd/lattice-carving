"""
Create synthetic test images for validating lattice-guided seam carving.

These images have simple, obvious features that we can verify are preserved
during carving operations.
"""

import sys
sys.path.insert(0, '../src')

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def create_river_image(width=400, height=300):
    """
    Create a meandering river image.

    - Green background (grass)
    - Blue river flowing through (curved path)
    - The river should be preserved during vertical seam carving
    """
    img = np.ones((height, width, 3), dtype=np.uint8)

    # Green background
    img[:, :, 0] = 50   # R
    img[:, :, 1] = 150  # G
    img[:, :, 2] = 50   # B

    # Create meandering river (blue)
    river_width = 40
    for y in range(height):
        # Sinusoidal path
        center_x = width // 2 + int(60 * np.sin(2 * np.pi * y / height))

        # Draw river
        x_start = max(0, center_x - river_width // 2)
        x_end = min(width, center_x + river_width // 2)

        img[y, x_start:x_end, 0] = 30   # R
        img[y, x_start:x_end, 1] = 144  # G
        img[y, x_start:x_end, 2] = 255  # B (blue)

    return img


def create_circle_image(width=400, height=400):
    """
    Create an image with a centered circle.

    - White background
    - Red circle in center
    - The circle should maintain its shape during carving
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Draw red circle
    center_y, center_x = height // 2, width // 2
    radius = min(width, height) // 3

    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                img[y, x, 0] = 255  # R
                img[y, x, 1] = 50   # G
                img[y, x, 2] = 50   # B

    return img


def create_concentric_circles_image(width=400, height=400):
    """
    Create an image with concentric circles (like a target).

    - White background
    - Alternating red and blue circles
    - Good for testing circular lattice carving
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    center_y, center_x = height // 2, width // 2
    max_radius = min(width, height) // 2

    colors = [
        (255, 50, 50),   # Red
        (50, 50, 255),   # Blue
        (255, 50, 50),   # Red
        (50, 50, 255),   # Blue
    ]

    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Determine which ring
            for i, ring_radius in enumerate(range(max_radius, 0, -max_radius // 4)):
                if dist <= ring_radius:
                    color = colors[i % len(colors)]
                    img[y, x] = color
                    break

    return img


def create_grid_image(width=400, height=400, grid_size=40):
    """
    Create a simple grid pattern.

    - White background
    - Black grid lines
    - Good for visualizing distortion during carving
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    # Vertical lines
    for x in range(0, width, grid_size):
        img[:, x:x+2] = 0  # Black

    # Horizontal lines
    for y in range(0, height, grid_size):
        img[y:y+2, :] = 0  # Black

    return img


def create_bullseye_with_background(width=400, height=400):
    """
    Create a bullseye target on a textured background.

    - Noisy green background
    - Clean red/white bullseye in center
    - Bullseye should be preserved as high-energy region
    """
    # Green noisy background
    np.random.seed(42)
    img = np.random.randint(40, 100, (height, width, 3), dtype=np.uint8)
    img[:, :, 1] += 50  # Make it greenish

    # Draw bullseye
    center_y, center_x = height // 2, width // 2
    max_radius = min(width, height) // 3

    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # Alternating rings
            ring_index = int(dist / (max_radius / 4))
            if ring_index < 4:
                if ring_index % 2 == 0:
                    img[y, x] = [255, 255, 255]  # White
                else:
                    img[y, x] = [255, 50, 50]    # Red

    return img


def visualize_test_images():
    """Create and display all test images."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = {
        'River': create_river_image(),
        'Circle': create_circle_image(),
        'Concentric Circles': create_concentric_circles_image(),
        'Grid': create_grid_image(),
        'Bullseye': create_bullseye_with_background(),
    }

    for idx, (name, img) in enumerate(images.items()):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')

    # Hide last subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('../output/test_images_preview.png', dpi=150)
    print("Saved: ../output/test_images_preview.png")


def main():
    print("Creating synthetic test images...")

    # Create test images
    test_images = {
        'river': create_river_image(),
        'circle': create_circle_image(),
        'concentric_circles': create_concentric_circles_image(),
        'grid': create_grid_image(),
        'bullseye': create_bullseye_with_background(),
    }

    # Save each test image
    for name, img in test_images.items():
        filename = f'../output/test_{name}.png'
        Image.fromarray(img).save(filename)
        print(f"  Saved: {filename}")

    # Create preview
    visualize_test_images()

    print("\nTest images created!")
    print("\nExpected behavior for each image:")
    print("  - River: Should preserve the curved blue river during vertical carving")
    print("  - Circle: Should maintain circular shape (not squish into oval)")
    print("  - Concentric Circles: Rings should stay circular")
    print("  - Grid: Should show distortion patterns clearly")
    print("  - Bullseye: Background should be removed, target preserved")


if __name__ == '__main__':
    main()
