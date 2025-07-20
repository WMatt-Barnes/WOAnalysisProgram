#!/usr/bin/env python3
"""
Fix icon sizes and regenerate proper icon files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image, ImageDraw
import os

def create_fixed_icons():
    """Create properly sized icons"""
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='#2F2F2F')
    ax.set_facecolor('#2F2F2F')
    
    # Set up the plot area
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Create grid pattern (subtle white lines)
    for i in range(0, 11, 2):
        ax.axhline(y=i, color='white', alpha=0.1, linewidth=0.5)
        ax.axvline(x=i, color='white', alpha=0.1, linewidth=0.5)
    
    # Create the main red arrow (3D glossy effect)
    arrow_start = (2, 2)
    arrow_end = (8, 8)
    
    # Main arrow body
    arrow = FancyArrowPatch(
        arrow_start, arrow_end,
        arrowstyle='->',
        mutation_scale=30,
        color='#FF4444',
        linewidth=8,
        alpha=0.9
    )
    ax.add_patch(arrow)
    
    # Add highlight for glossy effect
    highlight_arrow = FancyArrowPatch(
        (2.2, 2.2), (7.8, 7.8),
        arrowstyle='->',
        mutation_scale=25,
        color='#FF8888',
        linewidth=4,
        alpha=0.6
    )
    ax.add_patch(highlight_arrow)
    
    # Create Windows-style logo in upper left (2x2 grid of rectangles)
    logo_x, logo_y = 1, 7
    logo_size = 0.8
    gap = 0.1
    
    # Create 4 rectangles with glowing effect
    for i in range(2):
        for j in range(2):
            x = logo_x + i * (logo_size + gap)
            y = logo_y + j * (logo_size + gap)
            
            # Glow effect (outer glow)
            glow_rect = patches.Rectangle(
                (x-0.05, y-0.05), logo_size+0.1, logo_size+0.1,
                facecolor='#4A90E2', alpha=0.3, edgecolor='none'
            )
            ax.add_patch(glow_rect)
            
            # Main rectangle
            rect = patches.Rectangle(
                (x, y), logo_size, logo_size,
                facecolor='#2F2F2F', edgecolor='#4A90E2', linewidth=2
            )
            ax.add_patch(rect)
            
            # Inner highlight for 3D effect
            highlight_rect = patches.Rectangle(
                (x+0.02, y+0.02), logo_size-0.04, logo_size-0.04,
                facecolor='#4A90E2', alpha=0.2, edgecolor='none'
            )
            ax.add_patch(highlight_rect)
    
    # Add subtle baseline path (faint gray lines)
    baseline_points = np.array([[2, 2], [3, 2.5], [4, 2.8], [5, 3.2], [6, 3.8], [7, 4.5], [8, 5.2]])
    ax.plot(baseline_points[:, 0], baseline_points[:, 1], 
            color='#888888', alpha=0.3, linewidth=2, linestyle='--')
    
    # Save the icon in multiple formats
    icon_dir = "icons"
    if not os.path.exists(icon_dir):
        os.makedirs(icon_dir)
    
    # Save as high-resolution PNG first
    plt.savefig(os.path.join(icon_dir, 'app_icon_highres.png'), 
                dpi=300, bbox_inches='tight', facecolor='#2F2F2F')
    
    # Close the matplotlib figure
    plt.close(fig)
    
    # Now create properly sized versions using PIL
    try:
        # Load the high-res image
        highres_path = os.path.join(icon_dir, 'app_icon_highres.png')
        if os.path.exists(highres_path):
            img = Image.open(highres_path)
            
            # Create different sizes
            sizes = [16, 32, 48, 64, 128, 256]
            for size in sizes:
                resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
                output_path = os.path.join(icon_dir, f'app_icon_{size}x{size}.png')
                resized_img.save(output_path)
                print(f"✓ Created {size}x{size} icon: {output_path}")
            
            # Create ICO file with multiple sizes
            icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            icon_images = []
            for size in icon_sizes:
                resized = img.resize(size, Image.Resampling.LANCZOS)
                icon_images.append(resized)
            
            ico_path = os.path.join(icon_dir, 'app_icon.ico')
            icon_images[0].save(ico_path, format='ICO', sizes=icon_sizes)
            print(f"✓ Created ICO file: {ico_path}")
            
            # Also save a general PNG
            general_png = os.path.join(icon_dir, 'app_icon.png')
            img.save(general_png)
            print(f"✓ Created general PNG: {general_png}")
            
            # Clean up high-res file
            os.remove(highres_path)
            print("✓ Cleaned up temporary files")
            
        else:
            print("✗ High-res image not found")
            
    except Exception as e:
        print(f"✗ Error creating icon files: {e}")
    
    print(f"\nIcon creation completed in {icon_dir}/ directory")

if __name__ == "__main__":
    create_fixed_icons() 