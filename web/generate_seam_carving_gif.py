#!/usr/bin/env python3
"""
Generate a GIF from seam carving demo data showing the original image
with the current seam overlay at each step.
"""

import os
import sys
from pathlib import Path
from PIL import Image
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
    """Load the image for this step.
    
    Args:
        step_dir: Path to the step directory
        show_seams: If True, use image_seams.png (with overlaid seam curves).
                    If False, use image.png (clean image).
    """
    if show_seams:
        seams_path = step_dir / "image_seams.png"
        if seams_path.exists():
            return Image.open(seams_path).convert('RGB')
    
    return Image.open(step_dir / "image.png").convert('RGB')


def generate_gif(demo_name, steps, output_path, fps=5,
                 demo_data_root="../demo_data", show_seams=False):
    """
    Generate a GIF from seam carving steps.
    
    Args:
        demo_name: Name of the demo (e.g., "synthetic_bagel")
        steps: Number of steps in the demo
        output_path: Path to save the output GIF
        fps: Frames per second for the GIF
        demo_data_root: Root directory containing demo data
        show_seams: If True, overlay seam curves on frames
    """
    frames = []
    
    label = "with seams" if show_seams else "clean"
    print(f"Generating GIF for {demo_name} ({steps} steps, {label})...")
    
    for step in range(steps + 1):
        step_dir = Path(demo_data_root) / demo_name / f"step_{step:03d}"
        
        if not step_dir.exists():
            print(f"Warning: Directory for step {step} not found, skipping...")
            continue
        
        frame = load_frame(step_dir, show_seams=show_seams)
        frames.append(frame)
        
        if (step + 1) % 5 == 0 or step == 0:
            print(f"  Processed step {step}/{steps}")
    
    if not frames:
        print("Error: No frames were generated")
        sys.exit(1)
    
    duration = int(1000 / fps)
    
    print(f"Saving GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    print(f" GIF created successfully: {output_path}")
    print(f"  Frames: {len(frames)}, Duration: {len(frames) * duration / 1000:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF from seam carving demo data"
    )
    parser.add_argument(
        '--demo',
        type=str,
        help='Demo name (e.g., synthetic_bagel, arch, river)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output GIF filename (default: {demo_name}.gif)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help='Frames per second (default: 5)'
    )
    parser.add_argument(
        '--seams',
        action='store_true',
        help='Overlay seam curves on frames (default: clean image)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate GIFs for all demos'
    )
    parser.add_argument(
        '--demo-data-root',
        type=str,
        default='../demo_data',
        help='Root directory containing demo data (default: ../demo_data)'
    )
    
    args = parser.parse_args()
    
    demos = load_demo_config()
    
    if args.all:
        for demo in demos:
            output = f"{demo['name']}.gif"
            generate_gif(demo['name'], demo['steps'], output, args.fps,
                        args.demo_data_root, show_seams=args.seams)
            print()
    elif args.demo:
        demo = next((d for d in demos if d['name'] == args.demo), None)
        if not demo:
            print(f"Error: Demo '{args.demo}' not found")
            print(f"Available demos: {', '.join(d['name'] for d in demos)}")
            sys.exit(1)
        
        output = args.output or f"{demo['name']}.gif"
        generate_gif(demo['name'], demo['steps'], output, args.fps,
                    args.demo_data_root, show_seams=args.seams)
    else:
        print("Available demos:")
        for i, demo in enumerate(demos, 1):
            print(f"  {i}. {demo['name']} ({demo['steps']} steps)")
        
        choice = input("\nSelect demo number (or 'all' for all demos): ").strip()
        
        if choice.lower() == 'all':
            for demo in demos:
                output = f"{demo['name']}.gif"
                generate_gif(demo['name'], demo['steps'], output, args.fps,
                            args.demo_data_root, show_seams=args.seams)
                print()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(demos):
                    demo = demos[idx]
                    output = f"assets/{demo['name']}.gif"
                    generate_gif(demo['name'], demo['steps'], output, args.fps,
                                args.demo_data_root, show_seams=args.seams)
                else:
                    print("Invalid selection")
                    sys.exit(1)
            except ValueError:
                print("Invalid input")
                sys.exit(1)


if __name__ == "__main__":
    main()