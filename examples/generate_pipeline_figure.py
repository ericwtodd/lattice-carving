"""
Generate a single wide pipeline figure for presentation slides.

5 panels (left to right):
  1. Original image
  2. ROI highlight (semi-transparent overlay on river body)
  3. Lattice overlay with ROI (cyan) and pair (magenta) boundaries
  4. Lattice-space energy with seam pair
  5. Before / after result

Run:
    conda run -n lattice-carving python examples/generate_pipeline_figure.py
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
    _precompute_forward_mapping, _compute_valid_mask,
    _interpolate_seam, carve_seam_pairs,
)
from src.energy import gradient_magnitude_energy, normalize_energy
from src.seam import dp_seam_windowed, dp_seam_cyclic

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def tensor_to_numpy(t):
    """Convert (C,H,W) or (H,W) tensor to numpy for display."""
    if t.dim() == 3:
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return t.clamp(0, 1).cpu().numpy()


# ---------------------------------------------------------------------------
# Setup functions — one for synthetic river, one for real river
# ---------------------------------------------------------------------------

def setup_synthetic_river():
    """Synthetic sinusoidal river for quick iteration on layout."""
    H, W = 400, 600
    band_width = 60

    x_vals = torch.linspace(0, W, 200, dtype=torch.float32)
    y_vals = H / 2 + 60 * torch.sin(2 * np.pi * x_vals / W)
    curve_pts = torch.stack([x_vals, y_vals], dim=1)

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    curve_np = curve_pts.numpy()
    xx_np, yy_np = xx.numpy(), yy.numpy()
    min_dist = np.full((H, W), float('inf'), dtype=np.float32)
    for cx_pt, cy_pt in curve_np:
        d = np.sqrt((xx_np - cx_pt)**2 + (yy_np - cy_pt)**2)
        min_dist = np.minimum(min_dist, d)
    min_dist_t = torch.from_numpy(min_dist)

    river_mask = min_dist_t < band_width
    image = torch.zeros(3, H, W)
    image[0] = 0.25; image[1] = 0.50; image[2] = 0.20
    image[0][river_mask] = 0.15
    image[1][river_mask] = 0.35
    image[2][river_mask] = 0.75
    torch.manual_seed(42)
    image += (torch.rand(3, H, W) - 0.5) * 0.06
    image = image.clamp(0, 1)

    perp = band_width + 40
    lattice = Lattice2D.from_curve_points(curve_pts, n_lines=1024, perp_extent=perp)

    lattice_w = int(2 * perp)
    center_u = int(perp)
    buf = 5
    roi_range = (int(center_u - band_width), int(center_u + band_width))
    pair_range = (int(center_u + band_width + buf), lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=15, cyclic=False, curve_pts=curve_pts,
                title="Synthetic River")


def _load_river_centerline():
    """Load river centerline, trying cached JSON first, then SAM, then hardcoded.

    Returns:
        (control_pts, source_label) — tensor (N, 2) and string describing source.
    """
    project_root = Path(__file__).parent.parent

    # 1. Try cached centerline from SAM demo
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
        # Cache for next time
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
        print(f"  river.jpg not found, falling back to synthetic")
        return None

    pil_img = Image.open(river_path).convert('RGB')
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    image = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    C, H, W = image.shape
    print(f"  Real river: {W}x{H}")

    control_pts, source = _load_river_centerline()

    perp = 50
    lattice = Lattice2D.from_curve_points(
        control_pts, n_lines=2048, perp_extent=perp)
    lattice.smooth(max_iterations=500)

    lattice_w = int(2 * perp)
    center_u = int(perp)
    river_half = 20  # approximate half-width of river body in lattice u
    buf = 3
    roi_range = (center_u - river_half, center_u + river_half)
    pair_range = (center_u + river_half + buf, lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=12, cyclic=False, curve_pts=control_pts,
                title="River")


# ---------------------------------------------------------------------------
# Panel rendering helpers
# ---------------------------------------------------------------------------

def render_roi_highlight(image, lattice, lattice_w, roi_range, H, W):
    """Panel 2: Original image with ROI region highlighted via semi-transparent overlay."""
    u_map, n_map = _precompute_forward_mapping(lattice, H, W, image.device)
    valid_mask = _compute_valid_mask(lattice, u_map, n_map, H, W, image.device,
                                     lattice_width=lattice_w)

    roi_mask = valid_mask & (u_map >= roi_range[0]) & (u_map <= roi_range[1])

    img_np = tensor_to_numpy(image)
    overlay = img_np.copy()
    # Cyan highlight on ROI pixels
    roi_np = roi_mask.cpu().numpy()
    overlay[roi_np, 0] = overlay[roi_np, 0] * 0.4 + 0.0 * 0.6
    overlay[roi_np, 1] = overlay[roi_np, 1] * 0.4 + 0.9 * 0.6
    overlay[roi_np, 2] = overlay[roi_np, 2] * 0.4 + 0.9 * 0.6
    return overlay


def render_lattice_overlay(image, lattice, lattice_w, roi_range, pair_range,
                           H, W, n_scanlines=48, n_u=12, ax=None):
    """Panel 3: Image with lattice grid + ROI/pair boundary curves."""
    if ax is None:
        return
    ax.imshow(tensor_to_numpy(image), interpolation='bilinear')

    n_max = lattice.n_lines

    # Scanlines (constant n) — thin cyan
    for i in range(0, n_max, max(1, n_max // n_scanlines)):
        u_vals = torch.linspace(0, float(lattice_w), 80)
        pts = torch.stack([u_vals, torch.full_like(u_vals, float(i))], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'cyan', alpha=0.25, linewidth=0.5)

    # Perpendicular lines (constant u) — thin yellow
    for u_val in np.linspace(0, lattice_w, n_u):
        n_vals = torch.linspace(0, float(n_max - 1), 100)
        pts = torch.stack([torch.full_like(n_vals, u_val), n_vals], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'yellow', alpha=0.2, linewidth=0.5)

    # ROI boundaries — bold cyan dashed
    for u_val in roi_range:
        n_vals = torch.linspace(0, float(n_max - 1), 200)
        pts = torch.stack([torch.full_like(n_vals, float(u_val)), n_vals], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'cyan',
                linestyle='--', alpha=0.9, linewidth=1.5)

    # Pair boundaries — bold magenta dashed
    for u_val in pair_range:
        n_vals = torch.linspace(0, float(n_max - 1), 200)
        pts = torch.stack([torch.full_like(n_vals, float(u_val)), n_vals], dim=1)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()
        ax.plot(world_pts[:, 0], world_pts[:, 1], 'magenta',
                linestyle='--', alpha=0.9, linewidth=1.5)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)


def compute_seams(image, lattice, lattice_w, roi_range, pair_range, cyclic):
    """Compute one ROI seam and one pair seam. Returns (roi_seam, pair_seam) tensors."""
    energy = gradient_magnitude_energy(image)
    energy_3d = energy.unsqueeze(0) if energy.dim() == 2 else energy
    lattice_energy = lattice.resample_to_lattice_space(energy_3d, lattice_w)
    if lattice_energy.dim() == 3:
        lattice_energy = lattice_energy.squeeze(0)
    lattice_energy = normalize_energy(lattice_energy)

    if cyclic:
        roi_seam = dp_seam_cyclic(lattice_energy, roi_range, direction='vertical')
        pair_seam = dp_seam_cyclic(lattice_energy, pair_range, direction='vertical')
    else:
        roi_seam = dp_seam_windowed(lattice_energy, roi_range, direction='vertical')
        pair_seam = dp_seam_windowed(lattice_energy, pair_range, direction='vertical')

    return roi_seam, pair_seam, lattice_energy


def render_energy_seams(image, lattice, lattice_w, roi_range, pair_range,
                        cyclic, ax=None):
    """Panel 4: Lattice-space energy heatmap with one ROI seam and one pair seam."""
    if ax is None:
        return

    roi_seam, pair_seam, lattice_energy = compute_seams(
        image, lattice, lattice_w, roi_range, pair_range, cyclic)

    n_lines = lattice.n_lines
    n_idx = np.arange(n_lines)

    ax.imshow(lattice_energy.cpu().numpy(), cmap='hot', aspect='auto',
              interpolation='bilinear')
    ax.plot(roi_seam.cpu().numpy(), n_idx, color='cyan', linewidth=1.5,
            label='ROI seam')
    ax.plot(pair_seam.cpu().numpy(), n_idx, color='magenta', linewidth=1.5,
            label='Pair seam')
    ax.axvline(roi_range[0], color='cyan', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(roi_range[1], color='cyan', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(pair_range[0], color='magenta', linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(pair_range[1], color='magenta', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlim(0, lattice_w)
    ax.set_ylim(n_lines, 0)
    ax.legend(fontsize=7, loc='upper right')


def render_seams_world_space(image, lattice, lattice_w, roi_range, pair_range,
                             cyclic, H, W, ax=None):
    """Panel: Image with seam pair paths drawn in world space.

    Maps each scanline's seam u-value back to world coordinates via inverse_mapping,
    then draws the resulting curves on the image. This shows exactly where each seam
    cuts through the real image.
    """
    if ax is None:
        return

    roi_seam, pair_seam, _ = compute_seams(
        image, lattice, lattice_w, roi_range, pair_range, cyclic)

    ax.imshow(tensor_to_numpy(image), interpolation='bilinear')

    n_lines = lattice.n_lines
    n_vals = torch.arange(n_lines, dtype=torch.float32)

    for seam, color, label in [
        (roi_seam, 'cyan', 'ROI seam'),
        (pair_seam, 'magenta', 'Pair seam'),
    ]:
        # Each entry seam[n] is the u-value of the seam at scanline n.
        # Map (seam[n], n) back to world space.
        pts = torch.stack([seam.float(), n_vals], dim=1)  # (n_lines, 2) as (u, n)
        world_pts = lattice.inverse_mapping(pts).cpu().numpy()   # (n_lines, 2) as (x, y)
        ax.plot(world_pts[:, 0], world_pts[:, 1], color=color,
                linewidth=2, label=label, alpha=0.9)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------

def generate_pipeline_figure(setup_fn, output_name):
    """Generate the 5-panel pipeline figure."""
    params = setup_fn()
    if params is None:
        return

    image = params['image']
    lattice = params['lattice']
    lattice_w = params['lattice_w']
    roi_range = params['roi_range']
    pair_range = params['pair_range']
    n_seams = params['n_seams']
    cyclic = params['cyclic']
    title = params['title']

    if image.dim() == 2:
        image = image.unsqueeze(0)
    C, H, W = image.shape

    print(f"\nGenerating pipeline figure: {title}")
    print(f"  Image: {W}x{H}, lattice_w={lattice_w}, "
          f"roi={roi_range}, pair={pair_range}, n_seams={n_seams}")

    # --- Compute panels ---

    # Panel 2: ROI highlight
    roi_highlight = render_roi_highlight(image, lattice, lattice_w, roi_range, H, W)

    # Panel 5: Carve result
    print("  Computing carving result...")
    result = carve_seam_pairs(
        image, lattice, n_seams=n_seams,
        roi_range=roi_range, pair_range=pair_range,
        lattice_width=lattice_w, mode='shrink')
    result_np = tensor_to_numpy(result)

    # --- Assemble figure ---
    # 6 panels: original | ROI | lattice+bounds | seams (lattice space) | seams (world space) | result
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    panel_titles = [
        "1. Original",
        "2. ROI Region",
        "3. Lattice + Boundaries",
        "4. Seam Pair\n(lattice space)",
        "5. Seam Pair\n(world space)",
        f"6. Result\n({n_seams} seam pairs)",
    ]

    # Panel 1: Original
    axes[0].imshow(tensor_to_numpy(image))
    axes[0].axis('off')
    axes[0].set_title(panel_titles[0], fontsize=10, fontweight='bold')

    # Panel 2: ROI highlight
    axes[1].imshow(roi_highlight)
    axes[1].axis('off')
    axes[1].set_title(panel_titles[1], fontsize=10, fontweight='bold')

    # Panel 3: Lattice overlay
    render_lattice_overlay(image, lattice, lattice_w, roi_range, pair_range,
                           H, W, ax=axes[2])
    axes[2].axis('off')
    axes[2].set_title(panel_titles[2], fontsize=10, fontweight='bold')

    # Panel 4: Energy + seams (lattice space)
    render_energy_seams(image, lattice, lattice_w, roi_range, pair_range,
                        cyclic, ax=axes[3])
    axes[3].set_title(panel_titles[3], fontsize=10, fontweight='bold')

    # Panel 5: Seam pair projected back into world space
    render_seams_world_space(image, lattice, lattice_w, roi_range, pair_range,
                             cyclic, H, W, ax=axes[4])
    axes[4].set_title(panel_titles[4], fontsize=10, fontweight='bold')

    # Panel 6: Result
    axes[5].imshow(result_np)
    axes[5].axis('off')
    axes[5].set_title(panel_titles[5], fontsize=10, fontweight='bold')

    fig.suptitle(f"Lattice-Guided Seam Carving Pipeline — {title}",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_name
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def setup_synthetic_bagel():
    """Synthetic bagel: circular lattice, shrink the ring body.

    Use 600px and 1024 scanlines to avoid sawtooth artifacts from lattice discretization.
    """
    torch.manual_seed(42)
    H, W = 600, 600
    cx, cy = 300.0, 300.0
    inner_r, outer_r = 90, 240

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    image = torch.full((3, H, W), 0.15)
    ring_mask = (dist >= inner_r) & (dist <= outer_r)
    image[0][ring_mask] = 0.85
    image[1][ring_mask] = 0.70
    image[2][ring_mask] = 0.40
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    theta = torch.linspace(0, 2 * np.pi, 200)
    mid_r = (inner_r + outer_r) / 2
    curve_pts = torch.stack([
        cx + mid_r * torch.cos(theta),
        cy + mid_r * torch.sin(theta),
    ], dim=1)

    # pair region needs ≥ 2×n_seams width; add extra perp extent for background
    perp = (outer_r - inner_r) / 2 + 90   # ring_half=75 + 90 background
    lattice = Lattice2D.from_curve_points(curve_pts, n_lines=1024,
                                          perp_extent=perp, cyclic=True)
    lattice_w = int(2 * perp)
    center_u = int(perp)
    ring_half = (outer_r - inner_r) / 2   # 75
    roi_range = (center_u - 15, center_u + 15)
    pair_range = (int(center_u + ring_half + 5), lattice_w)

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=40, cyclic=True, curve_pts=curve_pts,
                title="Synthetic Bagel (shrink ring body)")


def setup_synthetic_arch():
    """Synthetic arch: semicircular lattice, shrink the arch thickness."""
    torch.manual_seed(42)
    H, W = 200, 300
    cy, cx = H - 20, W // 2
    outer_r, inner_r = 80, 55

    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    image = torch.full((3, H, W), 0.15)
    arch_mask = (dist >= inner_r) & (dist <= outer_r) & (yy < cy)
    image[0][arch_mask] = 0.85
    image[1][arch_mask] = 0.65
    image[2][arch_mask] = 0.35
    image += (torch.rand(3, H, W) - 0.5) * 0.05
    image = image.clamp(0, 1)

    # Arch is the UPPER semicircle (yy < cy), so the centerline uses cy - mid_r*sin.
    # angles π→0 traces left-base → top → right-base correctly when y = cy - mid_r*sin.
    angles = torch.linspace(np.pi, 0, 80)
    mid_r = (inner_r + outer_r) / 2
    curve_pts = torch.stack([
        cx + mid_r * torch.cos(angles),
        cy - mid_r * torch.sin(angles),   # minus: arch goes UP (toward smaller y)
    ], dim=1)

    perp = (outer_r - inner_r) / 2 + 20
    lattice = Lattice2D.from_curve_points(curve_pts, n_lines=H, perp_extent=perp)
    lattice_w = int(2 * perp)
    center_u = int(perp)
    # u=0 → inner hollow, u=center_u → arch midline, u=lattice_w → outer background
    roi_range = (center_u - 8, center_u + 8)   # arch body (±8 around midline)
    pair_range = (lattice_w - 15, lattice_w)    # outer background

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=8, cyclic=False, curve_pts=curve_pts,
                title="Synthetic Arch (shrink arch thickness)")


def setup_real_bagel():
    """Real bagel image (bagel.jpg, 295x246) with circular lattice."""
    bagel_path = Path(__file__).parent.parent / "assets" / "bagel.jpg"
    if not bagel_path.exists():
        print(f"  bagel.jpg not found, skipping")
        return None

    pil_img = Image.open(bagel_path).convert('RGB')
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    image = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    C, H, W = image.shape
    print(f"  Real bagel: {W}x{H}")

    # Circular lattice centred on the bagel
    cx, cy = W / 2, H / 2
    inner_r = 28
    outer_r = 92
    mid_r = (inner_r + outer_r) / 2

    theta = torch.linspace(0, 2 * np.pi, 200)
    curve_pts = torch.stack([
        cx + mid_r * torch.cos(theta),
        cy + mid_r * torch.sin(theta),
    ], dim=1)

    ring_half = (outer_r - inner_r) / 2   # 32 px
    # pair needs ≥ 2×n_seams=40 units → add 50px background buffer
    perp = ring_half + 55
    lattice = Lattice2D.from_curve_points(
        curve_pts, n_lines=1024, perp_extent=perp, cyclic=True)

    lattice_w = int(2 * perp)
    center_u = int(perp)
    # Ring body: [center_u - ring_half, center_u + ring_half]
    roi_range  = (center_u - 15, center_u + 15)
    pair_range = (int(center_u + ring_half + 8), lattice_w)   # ≥40 units wide

    return dict(image=image, lattice=lattice, lattice_w=lattice_w,
                roi_range=roi_range, pair_range=pair_range,
                n_seams=20, cyclic=True, curve_pts=curve_pts,
                title="Real Bagel")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    generate_pipeline_figure(setup_synthetic_river, "pipeline_synthetic_river.png")
    generate_pipeline_figure(setup_synthetic_bagel, "pipeline_synthetic_bagel.png")
    generate_pipeline_figure(setup_synthetic_arch,  "pipeline_synthetic_arch.png")
    generate_pipeline_figure(setup_real_bagel,      "pipeline_real_bagel.png")
    generate_pipeline_figure(setup_real_river,      "pipeline_real_river.png")

    print("\nDone!")
