"""
Generate pre-computed demo data for the browser-based seam pair viewer.

Inlines the seam pair loop so we can capture all intermediate state
(lattice-space image, energy, seam overlays) at each step.

Output structure:
    demo_data/<demo_name>/
        metadata.json
        step_000/image.png, image_seams.png, lattice_space.png, energy.png, seam_overlay.png
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
from PIL import Image

from src.lattice import Lattice2D
from src.carving import (
    _precompute_forward_mapping, _warp_and_resample,
    _interpolate_seam, _compute_valid_mask,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import dp_seam_windowed, dp_seam_cyclic


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def save_seam_overlay(lattice_img_np, roi_seam, pair_seam, roi_range, pair_range,
                      path, cyclic=False):
    """Save lattice-space image with seam lines and window boundaries overlaid."""
    n_lines = lattice_img_np.shape[0]
    lattice_w = lattice_img_np.shape[1]
    n_idx = np.arange(n_lines)

    roi_u = roi_seam.cpu().numpy()
    pair_u = pair_seam.cpu().numpy()
    if cyclic:
        n_idx = np.append(n_idx, n_lines)
        roi_u = np.append(roi_u, roi_u[0])
        pair_u = np.append(pair_u, pair_u[0])

    fig_h = 6
    fig_w = fig_h * (lattice_w / n_lines)
    fig_w = max(fig_w, 2.0)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(lattice_img_np, aspect='auto', interpolation='bilinear')

    ax.plot(roi_u, n_idx, color='cyan', linewidth=2, label='ROI seam')
    ax.plot(pair_u, n_idx, color='magenta', linewidth=2, label='Pair seam')

    ax.axvline(roi_range[0], color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(roi_range[1], color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pair_range[0], color='magenta', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(pair_range[1], color='magenta', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlim(0, lattice_w)
    ax.set_ylim(n_lines, 0)
    ax.axis('off')

    fig.savefig(str(path), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def seam_to_world(seam, lattice, cyclic=False):
    """Map a lattice-space seam to world-space (x, y) points."""
    n_lines = seam.shape[0]
    n_coords = torch.arange(n_lines, dtype=torch.float32, device=seam.device)
    lattice_pts = torch.stack([seam.float(), n_coords], dim=1)
    world_pts = lattice.inverse_mapping(lattice_pts)
    xy = world_pts.cpu().numpy()
    if cyclic:
        xy = np.vstack([xy, xy[:1]])
    return xy


def save_world_seam_overlay(image_np, all_seams, path, H, W):
    """Save world-space image with cumulative seam curves drawn on top.

    Uses fixed pixel-based figure size to guarantee output matches input dimensions.
    """
    dpi = 150
    fig_w = W / dpi
    fig_h = H / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure, no padding
    ax.imshow(image_np, interpolation='bilinear')

    for step_idx, (xy_roi, xy_pair) in enumerate(all_seams):
        alpha = 0.3 + 0.7 * (step_idx + 1) / len(all_seams)
        ax.plot(xy_roi[:, 0], xy_roi[:, 1], color='cyan', linewidth=1.5,
                alpha=alpha)
        ax.plot(xy_pair[:, 0], xy_pair[:, 1], color='magenta', linewidth=1.5,
                alpha=alpha)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis('off')
    fig.savefig(str(path), dpi=dpi, pad_inches=0)
    plt.close(fig)


def save_step0_seam_overlay(lattice_img_np, roi_range, pair_range, path, n_lines):
    """Save step-0 lattice overlay (just window boundaries, no seams)."""
    lattice_w = lattice_img_np.shape[1]
    fig_h = 6
    fig_w = max(2.0, fig_h * (lattice_w / n_lines))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(lattice_img_np, aspect='auto')
    ax.axvline(roi_range[0], color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(roi_range[1], color='cyan', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(pair_range[0], color='magenta', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(pair_range[1], color='magenta', linestyle='--', linewidth=1, alpha=0.5)
    ax.axis('off')
    fig.savefig(str(path), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic seam-pair demo generator
# ---------------------------------------------------------------------------

def save_lattice_overlay(image_np, lattice, lattice_w, path, H, W,
                         n_scanlines=48, n_u=12,
                         roi_range=None, pair_range=None):
    """Save world-space image with lattice grid drawn on top.

    Optionally draws ROI boundary curves (cyan) and pair boundary curves (magenta).
    """
    dpi = 150
    fig_w = W / dpi
    fig_h = H / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure, no padding
    ax.imshow(image_np, interpolation='bilinear')

    n_max = lattice.n_lines

    # Draw scanlines (constant n)
    for i in range(0, n_max, max(1, n_max // n_scanlines)):
        u_vals = torch.linspace(0, float(lattice_w), 80)
        n_val = float(i)
        pts = torch.stack([u_vals, torch.full_like(u_vals, n_val)], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'cyan', alpha=0.4, linewidth=0.7)

    # Draw perpendicular lines (constant u)
    for u_val in np.linspace(0, lattice_w, n_u):
        n_vals = torch.linspace(0, float(n_max - 1), 100)
        pts = torch.stack([torch.full_like(n_vals, u_val), n_vals], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'yellow', alpha=0.3, linewidth=0.7)

    # Draw ROI boundaries (constant u at roi endpoints) in cyan dashed
    if roi_range is not None:
        for u_val in roi_range:
            n_vals = torch.linspace(0, float(n_max - 1), 200)
            pts = torch.stack([torch.full_like(n_vals, float(u_val)), n_vals], dim=1)
            world_pts = lattice.inverse_mapping(pts).cpu().numpy()
            ax.plot(world_pts[:, 0], world_pts[:, 1], 'cyan',
                    linestyle='--', alpha=0.8, linewidth=1.5)

    # Draw pair boundaries (constant u at pair endpoints) in magenta dashed
    if pair_range is not None:
        for u_val in pair_range:
            n_vals = torch.linspace(0, float(n_max - 1), 200)
            pts = torch.stack([torch.full_like(n_vals, float(u_val)), n_vals], dim=1)
            world_pts = lattice.inverse_mapping(pts).cpu().numpy()
            ax.plot(world_pts[:, 0], world_pts[:, 1], 'magenta',
                    linestyle='--', alpha=0.8, linewidth=1.5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.axis('off')
    fig.savefig(str(path), dpi=dpi, pad_inches=0)
    plt.close(fig)


def generate_seam_pair_demo(output_dir, image, lattice, lattice_w,
                            roi_range, pair_range, n_seams, cyclic, title,
                            mode='shrink'):
    """Generate step-by-step seam pair demo data.

    Uses iterative warping (Section 3.3, paper's correct approach): each
    iteration computes energy on the current image state, finds seams in
    lattice space, then warps the current image by a single-step shift.
    This correctly handles composition of multiple g* mappings.

    Works for both cyclic (bagel) and non-cyclic (arch, river) lattices.

    Args:
        mode: 'shrink' to compress ROI / expand pair,
              'grow' to expand ROI / compress pair.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode not in ('shrink', 'grow'):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'shrink' or 'grow'.")
    roi_sign = 1.0 if mode == 'shrink' else -1.0

    if image.dim() == 2:
        image = image.unsqueeze(0)
    C, H, W = image.shape

    print(f"\n  [{title}] {n_seams} steps, {H}x{W}, {lattice.n_lines} scanlines, "
          f"lattice_w={lattice_w}, cyclic={cyclic}")

    # Precompute forward mapping
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, image.device)

    # Compute valid mask using the shared helper from carving.py
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, image.device,
                                     lattice_width=lattice_w)
    valid_mask_3d = valid_mask.unsqueeze(0).expand(C, -1, -1)
    print(f"    Valid pixels: {valid_mask.sum().item()}/{H*W} "
          f"({100*valid_mask.float().mean():.1f}%)")

    original_image = image.clone()
    current_image = image.clone()

    metadata = {
        'title': title,
        'image_size': [H, W],
        'n_scanlines': lattice.n_lines,
        'n_seams': n_seams,
        'lattice_width': lattice_w,
        'roi_range': list(roi_range),
        'pair_range': list(pair_range),
        'cyclic': cyclic,
        'steps': [],
    }

    # --- Step 0: Original ---
    step_dir = output_dir / 'step_000'
    step_dir.mkdir(exist_ok=True)

    save_image_clean(tensor_to_numpy(image), step_dir / 'image.png')
    save_image_clean(tensor_to_numpy(image), step_dir / 'image_seams.png')

    lattice_img = lattice.resample_to_lattice_space(image, lattice_w)
    lattice_img_np = tensor_to_numpy(lattice_img)
    save_image_clean(lattice_img_np, step_dir / 'lattice_space.png')

    energy = gradient_magnitude_energy(image)
    energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_w)
    if lattice_energy.dim() == 3:
        lattice_energy = lattice_energy.squeeze(0)
    lattice_energy = normalize_energy(lattice_energy)
    save_energy(lattice_energy, step_dir / 'energy.png')

    save_step0_seam_overlay(lattice_img_np, roi_range, pair_range,
                            step_dir / 'seam_overlay.png', lattice.n_lines)

    save_lattice_overlay(tensor_to_numpy(image), lattice, lattice_w,
                         step_dir / 'image_lattice.png', H, W,
                         roi_range=roi_range, pair_range=pair_range)

    metadata['steps'].append({'step': 0})
    print(f"    Step 0 (original)")

    # Track cumulative world-space seams for overlay visualization
    all_world_seams = []

    # --- Seam pair loop (iterative warping — paper Section 3.3) ---
    for i in range(n_seams):
        step_num = i + 1
        step_dir = output_dir / f'step_{step_num:03d}'
        step_dir.mkdir(exist_ok=True)

        # Energy from current image state (iterative, not original)
        energy = gradient_magnitude_energy(current_image)

        # Resample energy to lattice space + normalize
        energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
        lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_w)
        if lattice_energy.dim() == 3:
            lattice_energy = lattice_energy.squeeze(0)
        lattice_energy = normalize_energy(lattice_energy)

        # Find seams via DP (matching carving.py default method='dp')
        if cyclic:
            roi_seam = dp_seam_cyclic(lattice_energy, roi_range, direction='vertical')
            pair_seam = dp_seam_cyclic(lattice_energy, pair_range, direction='vertical')
        else:
            roi_seam = dp_seam_windowed(lattice_energy, roi_range, direction='vertical')
            pair_seam = dp_seam_windowed(lattice_energy, pair_range, direction='vertical')

        # Interpolate seams at fractional scanline positions
        roi_seam_interp = _interpolate_seam(roi_seam, n_map, cyclic=cyclic)
        pair_seam_interp = _interpolate_seam(pair_seam, n_map, cyclic=cyclic)

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

        # --- Save outputs ---
        # Restore original pixels outside lattice for clean output
        carved = torch.where(valid_mask_3d, current_image, original_image)
        save_image_clean(tensor_to_numpy(carved), step_dir / 'image.png')

        lattice_img = lattice.resample_to_lattice_space(carved, lattice_w)
        lattice_img_np = tensor_to_numpy(lattice_img)
        save_image_clean(lattice_img_np, step_dir / 'lattice_space.png')

        save_energy(lattice_energy, step_dir / 'energy.png')

        save_seam_overlay(lattice_img_np, roi_seam, pair_seam,
                          roi_range, pair_range, step_dir / 'seam_overlay.png',
                          cyclic=cyclic)

        xy_roi = seam_to_world(roi_seam, lattice, cyclic=cyclic)
        xy_pair = seam_to_world(pair_seam, lattice, cyclic=cyclic)
        all_world_seams.append((xy_roi, xy_pair))
        save_world_seam_overlay(tensor_to_numpy(carved), all_world_seams,
                                step_dir / 'image_seams.png', H, W)

        save_lattice_overlay(tensor_to_numpy(carved), lattice, lattice_w,
                             step_dir / 'image_lattice.png', H, W,
                             roi_range=roi_range, pair_range=pair_range)

        metadata['steps'].append({
            'step': step_num,
            'roi_seam': roi_seam.cpu().tolist(),
            'pair_seam': pair_seam.cpu().tolist(),
        })
        print(f"    Step {step_num}/{n_seams}")

    # Write metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Done: {output_dir}")
    return metadata


# ---------------------------------------------------------------------------
# Demo-specific setup functions
# ---------------------------------------------------------------------------

def setup_synthetic_bagel():
    """Synthetic ring (bagel) — cyclic seam pairs to shrink the ring body."""
    H, W = 600, 600
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

    # Add sesame seed texture to ring body (provides energy gradients)
    torch.manual_seed(42)
    n_seeds = 200
    for _ in range(n_seeds):
        angle = torch.rand(1) * 2 * np.pi
        radius = inner_r + torch.rand(1) * (outer_r - inner_r)
        seed_x = cx + radius * torch.cos(angle)
        seed_y = cy + radius * torch.sin(angle)
        seed_dist = torch.sqrt((xx - seed_x)**2 + (yy - seed_y)**2)
        seed_mask = seed_dist < 3.0
        image[0][seed_mask] = 0.95
        image[1][seed_mask] = 0.90
        image[2][seed_mask] = 0.70

    torch.manual_seed(42)
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    mid_r = (inner_r + outer_r) / 2
    theta = torch.linspace(0, 2 * np.pi, 200)
    circle_x = cx + mid_r * torch.cos(theta)
    circle_y = cy + mid_r * torch.sin(theta)
    curve_pts = torch.stack([circle_x, circle_y], dim=1)

    perp = (outer_r - inner_r) / 2 + 45  # 120 — wider to cover more background
    lattice = Lattice2D.from_curve_points(
        curve_pts, n_lines=1024, perp_extent=perp, cyclic=True)

    lattice_w = int(2 * perp)       # 240
    center_u = int(perp)            # 120
    ring_half = (outer_r - inner_r) / 2  # 75
    buf = 5  # buffer between ROI and pair windows
    # Full ring body as ROI, pair in outer background with buffer
    roi_range = (int(center_u - ring_half), int(center_u + ring_half))
    pair_range = (int(center_u + ring_half + buf), lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=20, cyclic=True)


def setup_arch():
    """Semicircular arch — non-cyclic seam pairs to thin the arch."""
    H, W = 400, 600
    cy, cx = H - 40, W // 2
    outer_r, inner_r = 160, 110

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    image = torch.full((3, H, W), 0.15)
    arch_mask = (dist >= inner_r) & (dist <= outer_r) & (yy < cy)
    image[0][arch_mask] = 0.85
    image[1][arch_mask] = 0.65
    image[2][arch_mask] = 0.35
    torch.manual_seed(42)
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    angles = torch.linspace(np.pi, 0, 200)
    mid_r = (inner_r + outer_r) / 2
    arch_x = cx + mid_r * torch.cos(angles)
    arch_y = cy - mid_r * torch.sin(angles)
    curve_pts = torch.stack([arch_x, arch_y], dim=1)

    arch_half = (outer_r - inner_r) / 2  # 25 — half the arch body width
    perp = arch_half + 40              # 65 — cover body + background
    lattice = Lattice2D.from_curve_points(
        curve_pts, n_lines=1024, perp_extent=perp)

    lattice_w = int(2 * perp)          # 130
    center_u = int(perp)               # 65
    buf = 5  # buffer between ROI and pair windows
    roi_range = (int(center_u - arch_half), int(center_u + arch_half))
    pair_range = (int(center_u + arch_half + buf), lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=15, cyclic=False)


def setup_river():
    """Sinusoidal river — non-cyclic seam pairs to narrow the river."""
    H, W = 400, 600
    band_width = 60

    x_vals = torch.linspace(0, W, 200, dtype=torch.float32)
    y_vals = H / 2 + 60 * torch.sin(2 * np.pi * x_vals / W)
    curve_pts = torch.stack([x_vals, y_vals], dim=1)

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    curve_np = curve_pts.numpy()
    xx_np = xx.numpy()
    yy_np = yy.numpy()
    min_dist = np.full((H, W), float('inf'), dtype=np.float32)
    for cx_pt, cy_pt in curve_np:
        d = np.sqrt((xx_np - cx_pt)**2 + (yy_np - cy_pt)**2)
        min_dist = np.minimum(min_dist, d)
    min_dist_t = torch.from_numpy(min_dist)

    river_mask = min_dist_t < band_width
    image = torch.zeros(3, H, W)
    image[0] = 0.25
    image[1] = 0.50
    image[2] = 0.20
    image[0][river_mask] = 0.15
    image[1][river_mask] = 0.35
    image[2][river_mask] = 0.75
    torch.manual_seed(42)
    image += (torch.rand(3, H, W) - 0.5) * 0.06
    image = image.clamp(0, 1)

    perp = band_width + 40              # 100 — cover river + background
    lattice = Lattice2D.from_curve_points(
        curve_pts, n_lines=1024, perp_extent=perp)

    lattice_w = int(2 * perp)          # 200
    center_u = int(perp)               # 100
    buf = 5  # buffer between ROI and pair windows
    roi_range = (int(center_u - band_width), int(center_u + band_width))
    pair_range = (int(center_u + band_width + buf), lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=15, cyclic=False)


def _load_river_centerline():
    """Load river centerline: try cached JSON, then SAM, then hardcoded."""
    project_root = Path(__file__).parent.parent

    # 1. Cached centerline from SAM demo
    cached_path = project_root / "output" / "river_centerline.json"
    if cached_path.exists():
        with open(cached_path) as f:
            pts = json.load(f)
        print("  Centerline: loaded from cached JSON")
        return torch.tensor(pts, dtype=torch.float32), "cached"

    # 2. Try running SAM
    try:
        from src.roi_extraction import segment_river
        river_path = project_root / "assets" / "river.jpg"
        control_pts = segment_river(str(river_path), n_control_points=40)
        cached_path.parent.mkdir(exist_ok=True)
        with open(cached_path, "w") as f:
            json.dump(control_pts.tolist(), f, indent=2)
        print("  Centerline: extracted via SAM (cached for reuse)")
        return control_pts, "sam"
    except Exception as e:
        print(f"  SAM unavailable ({e}), using hardcoded centerline")

    # 3. Hardcoded manual trace
    control_pts = torch.tensor([
        [30, 310], [70, 275], [120, 220], [155, 175], [155, 140],
        [140, 110], [165, 85], [215, 75], [265, 105], [310, 145],
        [355, 175], [395, 155], [440, 110], [490, 65], [540, 55],
        [590, 65], [640, 95], [670, 120],
    ], dtype=torch.float32)
    print("  Centerline: using hardcoded manual trace")
    return control_pts, "hardcoded"


def setup_real_river():
    """Real river image with SAM-derived or manually traced centerline."""
    river_path = Path(__file__).parent.parent / "assets" / "river.jpg"
    if not river_path.exists():
        print("  river.jpg not found, skipping real river demo")
        return None

    pil_img = Image.open(river_path).convert('RGB')
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    image = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    C, H, W = image.shape
    print(f"  Real river: {W}x{H}")

    control_pts, source = _load_river_centerline()

    perp = 50
    lattice = Lattice2D.from_curve_points(
        control_pts, n_lines=512, perp_extent=perp)
    lattice.smooth(max_iterations=500)

    lattice_w = int(2 * perp)
    center_u = int(perp)
    river_half = 20
    buf = 3
    roi_range = (center_u - river_half, center_u + river_half)
    pair_range = (center_u + river_half + buf, lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=12, cyclic=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_web_config(demo_dir, demos):
    """Write demo_config.js listing all available demos."""
    web_dir = demo_dir.parent / 'web'
    web_dir.mkdir(exist_ok=True)
    config_path = web_dir / 'demo_config.js'

    lines = ['// Auto-generated by generate_demo_data.py — do not edit',
             'const DEMOS = [']
    for name, meta in demos:
        n_steps = len(meta['steps']) - 1
        title = meta['title']
        lines.append(f'  {{ name: "{name}", title: "{title}", steps: {n_steps} }},')
    lines.append('];')
    lines.append(f'const DEMO_DATA_ROOT = "../demo_data";')

    config_path.write_text('\n'.join(lines) + '\n')
    print(f"\nConfig: {config_path}")


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    demo_dir = project_root / 'demo_data'

    all_demos = [
        ('synthetic_bagel_shrink', {**setup_synthetic_bagel(), 'mode': 'shrink', 'title':'Synthetic Bagel -- Shrink'}),
        ('synthetic_bagel_grow',   {**setup_synthetic_bagel(), 'mode': 'grow', 'title':'Synthetic Bagel -- Grow'}),
        ('arch_shrink', {**setup_arch(), 'mode': 'shrink', 'title':'Arch -- Shrink'}),
        ('arch_grow', {**setup_arch(), 'mode':'grow', 'title':'Arch -- Grow'}),
        ('river_shrink', {**setup_river(), 'mode': 'shrink', 'title':'River -- Shrink'}),
        ('river_grow', {**setup_river(), 'mode': 'grow', 'title':'River -- Grow'}),
    ]

    # Add real river demos if image is available
    real_river = setup_real_river()
    if real_river is not None:
        all_demos.extend([
            ('real_river_shrink', {**real_river, 'mode': 'shrink', 'title': 'Real River -- Shrink'}),
            ('real_river_grow', {**real_river, 'mode': 'grow', 'title': 'Real River -- Grow'}),
        ])

    results = []
    for name, params in all_demos:
        output_dir = demo_dir / name
        meta = generate_seam_pair_demo(output_dir, **params)
        results.append((name, meta))

    write_web_config(demo_dir, results)
    print("\nAll demos generated!")