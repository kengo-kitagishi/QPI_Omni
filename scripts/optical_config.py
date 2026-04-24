"""
optical_config.py -- Common QPI optical system parameters

Usage:
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE, CROP_REGION

Update before experiments:
    - OFFAXIS_CENTER : Value obtained from CursorVisualizer (get_offaxis_center.py)
    - CROP_REGION    : Crop region to use (only when changed)

Do not modify WAVELENGTH / NA / PIXELSIZE unless hardware changes.
"""

# ============================================================
# Parameters to update before experiments
# ============================================================

OFFAXIS_CENTER = (1640, 432)   # (row, col) -- updated 2026-04-23 (crop 400:2448 right channel)

# Crop region (row_start, row_end, col_start, col_end)
# Update when camera position changes
# Basler aca2440 (max 2048 rows)  -> (0, 2048, 208, 2256)  = 2048x2048
# MicroManager 1.4 (with margin)  -> (8, 2056, 208, 2256)  = 2048x2048
CROP_REGION = (0, 2048, 208, 2256)

# Reconstruction crop used by the raw-raw subtraction pipeline
# (grid_subtract.py, correct_0pergluc.py, batch_pipeline_all_pos.py).
# Must match the crop used when output_phase_raw/ TIFs were produced.
RAW_CROP = (0, 2048, 400, 2448)

# ============================================================
# Fixed optical parameters
# ============================================================

WAVELENGTH   = 658e-9           # m  (658 nm laser)
NA           = 0.95             # Objective lens numerical aperture
PIXELSIZE    = 3.45e-6 / 40     # m/px  (sensor 3.45 um, 40x objective)

# ============================================================
# OFFAXIS_CENTER history
# "What was the offaxis center for that experiment?" -> look here
# ============================================================
# Format: {"date": "YYYY-MM-DD", "center": (row, col), "note": "memo"}
# Add new entries at the top for each experiment

OFFAXIS_HISTORY = [
    {"date": "2026-04-23", "center": (1640,  432), "note": "re-measured for new experiment session (crop 400:2448)"},
    {"date": "2026-04-15", "center": (1633,  439), "note": "re-measured for new timelapse (crop 400:2448)"},
    {"date": "2026-03-23", "center": (1632,  445), "note": "re-measured with right channel crop 400:2448"},
    {"date": "2026-03-21", "center": (1634,  532), "note": "after diffraction grating change (measured with crop 208:2256)"},
    {"date": "2026-02-28", "center": (1712,  532), "note": ""},
    {"date": "2025-12-12", "center": (1664,  485), "note": "ph_1 / Pos20"},
    {"date": "unknown",    "center": (1642,  443), "note": "value from realtime monitor"},
    {"date": "unknown",    "center": (1623, 1621), "note": "value from qpi_03 batch"},
    {"date": "unknown",    "center": (1504, 1708), "note": "value from focus setup"},
    {"date": "unknown",    "center": ( 858,  759), "note": "250522 test timelapse"},
    {"date": "unknown",    "center": ( 857,  759), "note": "250528 visibility test"},
]


def get_offaxis_for_date(date_str: str):
    """
    Return offaxis_center corresponding to a date string ('YYYY-MM-DD').
    Returns current OFFAXIS_CENTER if not found.

    Example:
        center = get_offaxis_for_date("2025-12-12")
        # -> (1664, 485)
    """
    for entry in OFFAXIS_HISTORY:
        if entry["date"] == date_str:
            return entry["center"]
    return OFFAXIS_CENTER


def print_history():
    """Display OFFAXIS_HISTORY as a list."""
    print(f"{'Date':<14} {'Center':<16} Note")
    print("-" * 50)
    for e in OFFAXIS_HISTORY:
        print(f"{e['date']:<14} {str(e['center']):<16} {e['note']}")


if __name__ == "__main__":
    print_history()
