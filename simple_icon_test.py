#!/usr/bin/env python3
"""
Simple test to verify icon files can be loaded and displayed.
"""

import os
import sys

def test_icon_files():
    """Test if icon files exist and can be loaded"""
    print("Testing icon files...")
    
    # Check if icons directory exists
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    if not os.path.exists(icons_dir):
        print(f"✗ Icons directory not found: {icons_dir}")
        return False
    
    print(f"✓ Icons directory found: {icons_dir}")
    
    # Check each icon file
    icon_files = [
        'app_icon.ico',
        'app_icon_64x64.png',
        'app_icon_128x128.png',
        'app_icon_256x256.png'
    ]
    
    for icon_file in icon_files:
        icon_path = os.path.join(icons_dir, icon_file)
        if os.path.exists(icon_path):
            size = os.path.getsize(icon_path)
            print(f"✓ {icon_file} exists ({size} bytes)")
        else:
            print(f"✗ {icon_file} not found")
    
    # Test PIL import
    try:
        from PIL import Image, ImageTk
        print("✓ PIL/Pillow imported successfully")
        
        # Test loading a PNG file
        png_path = os.path.join(icons_dir, 'app_icon_64x64.png')
        if os.path.exists(png_path):
            try:
                img = Image.open(png_path)
                print(f"✓ PNG file loaded successfully: {img.size} pixels")
                
                # Test resizing
                resized = img.resize((64, 64), Image.Resampling.LANCZOS)
                print("✓ Image resizing successful")
                
                # Test PhotoImage creation
                photo = ImageTk.PhotoImage(resized)
                print("✓ PhotoImage created successfully")
                
                return True
            except Exception as e:
                print(f"✗ Error processing PNG file: {e}")
                return False
        else:
            print("✗ PNG file not found for testing")
            return False
            
    except ImportError as e:
        print(f"✗ PIL/Pillow not available: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_icon_files()
    if success:
        print("\n✅ All icon tests passed!")
    else:
        print("\n❌ Some icon tests failed!") 