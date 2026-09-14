#!/usr/bin/env python3
"""
Batch copy script for phase image files

Copies /Volumes/QPI_0_.01_r/251212/wo_cell/ph_2/Pos*/output_phase/img_000000000_ph_021_phase.tif
to /Volumes/QPI_0_.01_r/251212/ph_1/Pos*/output_phase/img_000000000_ph_000_phase.tif (overwrite).
"""

import shutil
from pathlib import Path
from typing import List, Tuple


def find_pos_folders(base_path: Path) -> List[Path]:
    """
    Detect all Pos folders under the specified path and return them sorted by number.

    Args:
        base_path: Base path to search

    Returns:
        Sorted list of Pos folders
    """
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return []
    
    pos_folders = sorted(
        base_path.glob("Pos*"),
        key=lambda p: int(p.name.replace("Pos", "")) if p.name.replace("Pos", "").isdigit() else 0
    )
    
    return [p for p in pos_folders if p.is_dir()]


def copy_phase_file(source_path: Path, dest_path: Path) -> Tuple[bool, str]:
    """
    Copy a phase file.

    Args:
        source_path: Source file path
        dest_path: Destination file path

    Returns:
        Tuple of (success flag, message)
    """
    try:
        # Check if source file exists
        if not source_path.exists():
            return False, f"Source file does not exist: {source_path}"

        # Create destination directory if it does not exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file (preserving metadata)
        shutil.copy2(source_path, dest_path)

        return True, f"Copy succeeded: {source_path.name} -> {dest_path}"

    except PermissionError as e:
        return False, f"Permission error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Main processing"""
    print("=" * 80)
    print("Batch copy script for phase image files")
    print("=" * 80)
    print()

    # Set base paths
    base_volume = Path(r"F:\251212")
    source_base = base_volume / "wo_cell" / "ph_2"
    dest_base = base_volume / "ph_1"

    # Set file names
    source_filename = "img_000000000_ph_021_phase.tif"
    dest_filename = "img_000000000_ph_000_phase.tif"

    print(f"Source base path: {source_base}")
    print(f"Destination base path: {dest_base}")
    print()

    # Check if volume exists
    if not base_volume.exists():
        print(f"Error: Volume is not mounted: {base_volume}")
        return

    # Detect Pos folders
    print("Detecting Pos folders...")
    pos_folders = find_pos_folders(source_base)

    if not pos_folders:
        print("Error: No Pos folders found")
        return

    print(f"Number of detected Pos folders: {len(pos_folders)}")
    print(f"Range: {pos_folders[0].name} - {pos_folders[-1].name}")
    print()
    print("-" * 80)
    print()

    # Record processing results
    success_count = 0
    fail_count = 0
    skip_count = 0

    # Process each Pos folder
    for pos_folder in pos_folders:
        pos_name = pos_folder.name

        # Build source file path
        source_file = pos_folder / "output_phase" / source_filename

        # Build destination file path
        dest_file = dest_base / pos_name / "output_phase" / dest_filename

        print(f"[{pos_name}]", end=" ")

        # Copy file
        success, message = copy_phase_file(source_file, dest_file)

        if success:
            print(f"✓ {message}")
            success_count += 1
        else:
            if "does not exist" in message:
                print(f"⊘ Skipped: {message}")
                skip_count += 1
            else:
                print(f"✗ Failed: {message}")
                fail_count += 1

    # Processing results summary
    print()
    print("-" * 80)
    print()
    print("Processing results summary:")
    print(f"  Succeeded: {success_count} files")
    print(f"  Failed: {fail_count} files")
    print(f"  Skipped: {skip_count} files")
    print(f"  Total: {len(pos_folders)} folders")
    print()
    print("=" * 80)

    if fail_count > 0:
        print("Warning: Errors occurred for some files")
    elif success_count > 0:
        print("All files were copied successfully")
    else:
        print("No files were copied")

    print("=" * 80)


if __name__ == "__main__":
    main()

