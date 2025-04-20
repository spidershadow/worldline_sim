#!/usr/bin/env python3
"""
Clean up unnecessary files from the worldline simulator project.
This script removes old PNG files and pycache directories while preserving MP4 and JPG files.

Usage:
    python cleanup_files.py

The script will:
1. Remove all __pycache__ directories
2. Remove old visualization files (retro_tstar_hist_old*.png)
3. Clean up the empty animation_frames directory
4. Note: All MP4 and JPG files are preserved
"""

import os
import shutil
import re
from pathlib import Path

def remove_pycache_dirs():
    """Remove all __pycache__ directories recursively"""
    removed = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            print(f"Removing: {cache_dir}")
            shutil.rmtree(cache_dir)
            removed += 1
    print(f"Removed {removed} __pycache__ directories")

def remove_specific_png_files():
    """Remove specific PNG files that are no longer needed"""
    # Old visualization files
    old_files = [
        "worldline_sim/retro_tstar_hist_old.png",
        "worldline_sim/retro_tstar_hist_old2.png",
        "worldline_sim/retro_tstar_hist_old3.png"
    ]
    
    removed = 0
    for file_path in old_files:
        if os.path.exists(file_path):
            print(f"Removing: {file_path}")
            os.remove(file_path)
            removed += 1
    
    print(f"Removed {removed} old visualization files")

def remove_redundant_timeline_pngs():
    """Remove all the super_lenient_timeline_t*.png files"""
    timeline_pattern = re.compile(r"super_lenient_timeline_t\d+\.png$")
    removed = 0
    
    for file in os.listdir():
        if os.path.isfile(file) and timeline_pattern.match(file):
            print(f"Removing: {file}")
            os.remove(file)
            removed += 1
    
    print(f"Removed {removed} redundant timeline PNG files")

def cleanup_empty_dirs():
    """Remove empty directories"""
    if os.path.exists('animation_frames') and os.path.isdir('animation_frames'):
        if not os.listdir('animation_frames'):
            print("Removing empty directory: animation_frames/")
            os.rmdir('animation_frames')
        else:
            print("Note: animation_frames/ is not empty, keeping it")

def main():
    print("Cleaning up unnecessary files...")
    
    # Ask for confirmation
    confirm = input("This will remove unnecessary files. Continue? (y/n): ")
    if confirm.lower() not in ('y', 'yes'):
        print("Cleanup cancelled.")
        return
    
    remove_pycache_dirs()
    remove_specific_png_files()
    remove_redundant_timeline_pngs()
    cleanup_empty_dirs()
    
    print("\nCleanup complete!")
    print("All MP4 and JPG files were preserved as requested.")

if __name__ == "__main__":
    main() 