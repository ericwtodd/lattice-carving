"""
Generate pre-computed demo data for the browser-based seam pair viewer.

Inlines the seam pair loop from carve_seam_pairs() so we can capture all
intermediate state (lattice-space image, energy, seam overlays) at each step.

Output structure:
    demo_data/synthetic_bagel/
        metadata.json
        step_000/image.png, lattice_space.png, energy.png, seam_overlay.png
        step_001/...
        ...

Run:
    conda run -n lattice-carving python examples/generate_demo_data.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.lattice import Lattice2D
from src.carving import _precompute_forward_mapping, _warp_and_resample, _interpolate_seam
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import dp_seam_cyclic


def tensor_to_numpy(t):
    """Convert (C,H,W) or (H,W) tensor to numpy for display."""
    if t.dim() == 3:
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return t.clamp(0, 1).cpu().numpy()


def save_image_clean(arr, path):
    """Save a numpy array as a PNG with no axes/borders."""
    plt.imsave(str(path), arr)


def save_energy(energy_2d, path):
    """Save a 2D energy tensor as a hot-colormap PNG."""
    plt.imsave(str(path), energy_2d.cpu().numpy(), cmap='hot')


def save_seam_overlay(lattice_img_np, roi_seam, pair_seam, roi_range, pair_range, path):
    """Save lattice-space image with seam lines and window boundaries overlaid."""
    n_lines = lattice_img_np.shape[0]
    lattice_w = lattice_img_np.shape[1]
    n_idx = np.arange(n_lines)

    # Figure sized to match lattice-space aspect ratio
    fig_h = 6
    fig_w = fig_h * (lattice_w / n_lines)
    fig_w = max(fig_w, 2.0)  # minimum width

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(lattice_img_np, aspect='auto', interpolation='bilinear')

    # ROI seam (cyan) and pair seam (magenta)
    ax.plot(roi_seam.cpu().numpy(), n_idx, color='cyan', linewidth=2, label='ROI seam')
    ax.plot(pair_seam.cpu().numpy(), n_idx, color='magenta', linewidth=2, label='Pair seam')

    # Window boundaries (dashed)
    ax.axvline(roi_range[0], color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(roi_range[1], color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pair_range[0], color='magenta', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pair_range[1], color='magenta', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(0, lattice_w)
    ax.set_ylim(n_lines, 0)
    ax.axis('off')

    fig.savefig(str(path), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def make_synthetic_bagel(H, W):
    """Create a synthetic bagel image (ring on dark background)."""
    cx, cy = W / 2.0, H / 2.0
    inner_r = int(0.15 * min(H, W))
    outer_r = int(0.40 * min(H, W))

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    image = torch.full((3, H, W), 0.15)
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    image[0][ring_mask] = 0.85
    image[1][ring_mask] = 0.70
    image[2][ring_mask] = 0.40
    # Texture
    torch.manual_seed(42)
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    return image, cx, cy, inner_r, outer_r


def generate_demo(output_dir: Path, H=600, W=600, n_scanlines=256, n_seams=18):
    """Generate all demo data for the synthetic bagel seam pair demo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_seams} seam pair steps at {H}x{W}, {n_scanlines} scanlines...")

    # --- Create synthetic bagel ---
    image, cx, cy, inner_r, outer_r = make_synthetic_bagel(H, W)
    mid_r = (inner_r + outer_r) / 2

    # --- Build circular lattice ---
    theta = torch.linspace(0, 2 * np.pi, 80)
    circle_x = cx + mid_r * torch.cos(theta)
    circle_y = cy + mid_r * torch.sin(theta)
    curve_pts = torch.stack([circle_x, circle_y], dim=1)

    perp = (outer_r - inner_r) / 2 + 15
    lat = Lattice2D.from_curve_points(curve_pts, n_lines=n_scanlines, perp_extent=perp, cyclic=True)

    lattice_w = int(2 * perp)
    center_u = int(perp)

    # Shrink the ring body (ROI = ring center, pair = outer background)
    roi_range = (center_u - 5, center_u + 5)
    pair_range = (lattice_w - 11, lattice_w)

    # --- Precompute forward mapping ---
    u_map, n_map = _precompute_forward_mapping(lat, H, W, image.device)
    original_image = image.clone()
    cumulative_shift = torch.zeros_like(u_map)

    metadata = {
        'image_size': [H, W],
        'n_scanlines': n_scanlines,
        'n_seams': n_seams,
        'lattice_width': lattice_w,
        'roi_range': list(roi_range),
        'pair_range': list(pair_range),
        'steps': [],
    }

    # --- Step 0: Original (no seams) ---
    step_dir = output_dir / 'step_000'
    step_dir.mkdir(exist_ok=True)

    # World-space image
    save_image_clean(tensor_to_numpy(image), step_dir / 'image.png')

    # Lattice-space color image
    lattice_img = lat.resample_to_lattice_space(image, lattice_w)
    lattice_img_np = tensor_to_numpy(lattice_img)
    save_image_clean(lattice_img_np, step_dir / 'lattice_space.png')

    # Energy in lattice space
    energy = gradient_magnitude_energy(image)
    energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
    lattice_energy = lat.resample_to_lattice_space(energy_3d, lattice_w)
    if lattice_energy.dim() == 3:
        lattice_energy = lattice_energy.squeeze(0)
    lattice_energy = normalize_energy(lattice_energy)
    save_energy(lattice_energy, step_dir / 'energy.png')

    # Step 0: no seam overlay — save the plain lattice-space image
    fig, ax = plt.subplots(1, 1, figsize=(max(2.0, 6 * lattice_w / n_scanlines), 6))
    ax.imshow(lattice_img_np, aspect='auto')
    ax.axvline(roi_range[0], color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(roi_range[1], color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(pair_range[0], color='magenta', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(pair_range[1], color='magenta', linestyle='--', linewidth=1, alpha=0.5)
    ax.axis('off')
    fig.savefig(str(step_dir / 'seam_overlay.png'), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    metadata['steps'].append({'step': 0, 'roi_seam': None, 'pair_seam': None})
    print(f"  Step 0 (original) saved")

    # --- Seam pair loop (inlined from carve_seam_pairs) ---
    for i in range(n_seams):
        step_num = i + 1
        step_dir = output_dir / f'step_{step_num:03d}'
        step_dir.mkdir(exist_ok=True)

        # Step 1: Compute energy from current warped state
        if i == 0:
            energy = gradient_magnitude_energy(original_image)
        else:
            current_warped = _warp_and_resample(original_image, lat, u_map, n_map, cumulative_shift)
            energy = gradient_magnitude_energy(current_warped)

        # Step 2: Resample energy to lattice space + normalize
        energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
        lattice_energy = lat.resample_to_lattice_space(energy_3d, lattice_w)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)
        lattice_energy = normalize_energy(lattice_energy)

        # Step 3: Find seams (DP cyclic)
        roi_seam = dp_seam_cyclic(lattice_energy, roi_range, direction='vertical')
        pair_seam = dp_seam_cyclic(lattice_energy, pair_range, direction='vertical')

        # Step 4: Interpolate seams and apply shift
        roi_seam_interp = _interpolate_seam(roi_seam, n_map)
        pair_seam_interp = _interpolate_seam(pair_seam, n_map)

        u_adjusted = u_map + cumulative_shift
        new_shift = torch.zeros_like(u_map)
        new_shift = new_shift + torch.where(u_adjusted >= roi_seam_interp,
                                             torch.ones_like(u_map),
                                             torch.zeros_like(u_map))
        new_shift = new_shift + torch.where(u_adjusted > pair_seam_interp,
                                             -torch.ones_like(u_map),
                                             torch.zeros_like(u_map))
        cumulative_shift = cumulative_shift + new_shift

        # --- Save outputs for this step ---

        # World-space carved image
        carved = _warp_and_resample(original_image, lat, u_map, n_map, cumulative_shift)
        save_image_clean(tensor_to_numpy(carved), step_dir / 'image.png')

        # Lattice-space color image (from current warped state)
        if i == 0:
            lattice_img = lat.resample_to_lattice_space(original_image, lattice_w)
        else:
            warped_for_viz = _warp_and_resample(original_image, lat, u_map, n_map, cumulative_shift)
            lattice_img = lat.resample_to_lattice_space(warped_for_viz, lattice_w)
        lattice_img_np = tensor_to_numpy(lattice_img)
        save_image_clean(lattice_img_np, step_dir / 'lattice_space.png')

        # Energy heatmap
        save_energy(lattice_energy, step_dir / 'energy.png')

        # Seam overlay on lattice-space color image
        save_seam_overlay(lattice_img_np, roi_seam, pair_seam,
                          roi_range, pair_range, step_dir / 'seam_overlay.png')

        metadata['steps'].append({
            'step': step_num,
            'roi_seam': roi_seam.cpu().tolist(),
            'pair_seam': pair_seam.cpu().tolist(),
        })
        print(f"  Step {step_num}/{n_seams} saved")

    # --- Write metadata ---
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {n_seams + 1} steps saved to {output_dir}")
    print(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'demo_data' / 'synthetic_bagel'
    generate_demo(output_dir)
