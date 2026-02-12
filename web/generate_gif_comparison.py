#!/usr/bin/env python3
"""
Generate a side-by-side comparison GIF:
    [shrink animation] | [original (static)] | [grow animation]

Usage:
    python generate_gif_comparison.py --demo synthetic_bagel
    python generate_gif_comparison.py --demo arch --fps 8 --seams
    python generate_gif_comparison.py --all
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def load_demo_config():
    """Parse the demo_config.js file to get available demos."""
    config_path = Path("demo_config.js")
    if not config_path.exists():
        print("Error: demo_config.js not found")
        sys.exit(1)

    demos = []
    with open(config_path, 'r') as f:
        for line in f:
            if line.strip().startswith('{ name:'):
                parts = line.split(',')
                name = parts[0].split('"')[1]
                steps = int(parts[2].split(':')[1].strip().rstrip(' },'))
                demos.append({'name': name, 'steps': steps})
    return demos


def load_frame(step_dir, show_seams=False):
    """Load the image for this step."""
    if show_seams:
        seams_path = step_dir / "image_seams.png"
        if seams_path.exists():
            return Image.open(seams_path).convert('RGB')
    return Image.open(step_dir / "image.png").convert('RGB')


def add_label(image, text, position='bottom'):
    """Add a text label to an image, returning a new image."""
    label_h = 28
    w, h = image.size
    new_h = h + label_h
    labeled = Image.new('RGB', (w, new_h), (0, 0, 0))

    if position == 'bottom':
        labeled.paste(image, (0, 0))
        text_y = h + 4
    else:
        labeled.paste(image, (0, label_h))
        text_y = 4

    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (w - text_w) // 2
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return labeled


def compose_frame(shrink_img, original_img, grow_img, gap=4):
    """Combine three images side by side with a gap between them."""
    # Resize all to match height
    target_h = max(shrink_img.height, original_img.height, grow_img.height)

    def resize_to_height(img, h):
        if img.height == h:
            return img
        w = int(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    shrink_img = resize_to_height(shrink_img, target_h)
    original_img = resize_to_height(original_img, target_h)
    grow_img = resize_to_height(grow_img, target_h)

    total_w = shrink_img.width + original_img.width + grow_img.width + gap * 2
    combined = Image.new('RGB', (total_w, target_h), (0, 0, 0))

    x = 0
    combined.paste(shrink_img, (x, 0))
    x += shrink_img.width + gap
    combined.paste(original_img, (x, 0))
    x += original_img.width + gap
    combined.paste(grow_img, (x, 0))

    return combined


def generate_comparison_gif(demo_base, steps, output_path, fps=5,
                            demo_data_root="../demo_data", show_seams=False,
                            labels=True):
    """
    Generate a side-by-side comparison GIF.

    Expects demo data directories:
        {demo_base}_shrink/  and  {demo_base}_grow/

    Or if those don't exist, tries:
        {demo_base}/  (as shrink) — and skips grow if missing.

    Args:
        demo_base: Base demo name (e.g., "synthetic_bagel")
        steps: Number of seam steps
        output_path: Output GIF path
        fps: Frames per second
        demo_data_root: Root directory for demo data
        show_seams: Overlay seam curves on frames
        labels: Add text labels to each panel
    """
    root = Path(demo_data_root)

    # Locate shrink and grow directories
    shrink_dir = root / f"{demo_base}_shrink"
    grow_dir = root / f"{demo_base}_grow"

    if not shrink_dir.exists():
        shrink_dir = root / demo_base
    if not shrink_dir.exists():
        print(f"Error: Could not find shrink data at {shrink_dir}")
        sys.exit(1)
    if not grow_dir.exists():
        print(f"Error: Could not find grow data at {grow_dir}")
        sys.exit(1)

    # Load original (step 0 from either directory)
    original = load_frame(shrink_dir / "step_000", show_seams=False)
    if labels:
        original = add_label(original, "Original")

    print(f"Generating comparison GIF for {demo_base} ({steps} steps)...")

    frames = []
    for step in range(steps + 1):
        step_str = f"step_{step:03d}"

        shrink_step = shrink_dir / step_str
        grow_step = grow_dir / step_str

        if not shrink_step.exists() or not grow_step.exists():
            print(f"  Warning: step {step} missing, skipping")
            continue

        shrink_frame = load_frame(shrink_step, show_seams=show_seams)
        grow_frame = load_frame(grow_step, show_seams=show_seams)

        if labels:
            shrink_frame = add_label(shrink_frame, f"Shrink (step {step})")
            grow_frame = add_label(grow_frame, f"Grow (step {step})")

        combined = compose_frame(shrink_frame, original, grow_frame)
        frames.append(combined)

        if step % 5 == 0:
            print(f"  Processed step {step}/{steps}")

    if not frames:
        print("Error: No frames generated")
        sys.exit(1)

    duration = int(1000 / fps)

    # Hold first and last frames longer
    durations = [duration] * len(frames)
    durations[0] = duration * 3
    durations[-1] = duration * 3

    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    total_dur = sum(durations) / 1000
    print(f"✓ GIF created: {output_path}")
    print(f"  Frames: {len(frames)}, Duration: {total_dur:.1f}s, "
          f"Size: {frames[0].width}x{frames[0].height}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side shrink/original/grow comparison GIF"
    )
    parser.add_argument(
        '--demo', type=str,
        help='Base demo name (e.g., synthetic_bagel). '
             'Expects {name}_shrink/ and {name}_grow/ directories.'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output GIF filename (default: {demo}_comparison.gif)'
    )
    parser.add_argument(
        '--fps', type=int, default=5,
        help='Frames per second (default: 5)'
    )
    parser.add_argument(
        '--seams', action='store_true',
        help='Overlay seam curves on frames'
    )
    parser.add_argument(
        '--no-labels', action='store_true',
        help='Omit text labels'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Generate comparison GIFs for all demos that have shrink/grow pairs'
    )
    parser.add_argument(
        '--demo-data-root', type=str, default='../demo_data',
        help='Root directory containing demo data (default: ../demo_data)'
    )

    args = parser.parse_args()
    demos = load_demo_config()

    # Group demos by base name (strip _shrink/_grow suffix)
    base_names = {}
    for d in demos:
        name = d['name']
        if name.endswith('_shrink'):
            base = name[:-7]
        elif name.endswith('_grow'):
            base = name[:-5]
        else:
            base = name
        if base not in base_names:
            base_names[base] = d['steps']

    labels = not args.no_labels

    if args.all:
        root = Path(args.demo_data_root)
        for base, steps in base_names.items():
            shrink_exists = (root / f"{base}_shrink").exists() or (root / base).exists()
            grow_exists = (root / f"{base}_grow").exists()
            if shrink_exists and grow_exists:
                output = f"{base}_comparison.gif"
                generate_comparison_gif(base, steps, output, args.fps,
                                        args.demo_data_root, args.seams, labels)
                print()
            else:
                print(f"Skipping {base}: need both _shrink and _grow directories")
    elif args.demo:
        base = args.demo
        # Find step count
        steps = base_names.get(base)
        if steps is None:
            # Try looking up with suffix
            for d in demos:
                if d['name'].startswith(base):
                    steps = d['steps']
                    break
        if steps is None:
            print(f"Error: Demo '{base}' not found")
            print(f"Available bases: {', '.join(base_names.keys())}")
            sys.exit(1)

        output = args.output or f"{base}_comparison.gif"
        generate_comparison_gif(base, steps, output, args.fps,
                                args.demo_data_root, args.seams, labels)
    else:
        print("Available demo bases:")
        root = Path(args.demo_data_root)
        for i, (base, steps) in enumerate(base_names.items(), 1):
            has_pair = ((root / f"{base}_shrink").exists() or (root / base).exists()) \
                       and (root / f"{base}_grow").exists()
            status = "✓ pair found" if has_pair else "✗ missing grow/shrink"
            print(f"  {i}. {base} ({steps} steps) — {status}")

        print("\nUsage: python generate_gif_comparison.py --demo <name>")
        print("       python generate_gif_comparison.py --all")


if __name__ == "__main__":
    main()