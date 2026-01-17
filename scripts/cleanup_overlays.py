#!/usr/bin/env python3
"""Script to remove duplicate overlay functions from main.py."""

from pathlib import Path

# Read the file
codroid_picker_path = Path(
    "/Users/levko/repos/codroid-ball-picker/app/main.py")
lines = codroid_picker_path.read_text().splitlines(keepends=True)

# Define ranges to remove (1-indexed line numbers from grep)
# Lines 3107-3783: All overlay drawing functions (_draw_label through _ensure_robot_coord5_for_uri)
# But we need to keep _format_device_list, _probe_robot, _with_robot_api, _ensure_robot_coord5_for_uri
# So remove lines 3107-3746 (up to draw_monitor_overlay)

# We'll remove lines 3107-3746
# Then remove the task runner functions 3783-3852 (about 70 lines)

# Since we're removing from bottom to top, start with the task runners
# After first removal, line numbers will shift, so we need to be careful

# Actually, let's be more surgical and just remove the draw functions, keeping everything else

# Remove in reverse order to preserve line numbers
# Lines to remove:
# 3783-3852: Task runner functions (_run_charuco_task through _select_overlay_task)
# 3637-3746: draw_detection_overlay, draw_drop_marker_overlay, draw_monitor_overlay
# 3411-3635: draw_plate_overlay, _refresh_plate_status
# 3257-3409: draw_workspace_overlay
# 3200-3256: draw_charuco_overlay
# 3189-3199: _apply_workspace_overlay
# 3158-3188: _draw_dotted_polyline
# 3110-3157: _workspace_corners_px
# 3107-3109: _draw_label

ranges_to_remove = [
    (3783, 3852),  # Task runners
    (3637, 3746),  # Detection overlays
    (3411, 3635),  # Plate overlay
    (3257, 3409),  # Workspace overlay
    (3200, 3256),  # ChArUco overlay
    (3189, 3199),  # apply workspace
    (3158, 3188),  # dotted polyline
    (3110, 3157),  # workspace corners
    (3107, 3109),  # draw label
]

# Remove from bottom to top
for start, end in ranges_to_remove:
    # Convert to 0-indexed
    start_idx = start - 1
    end_idx = end  # end is inclusive in our notation, so this works

    # Remove the lines
    del lines[start_idx:end_idx]
    print(f"Removed lines {start}-{end} ({end-start+1} lines)")

# Write back
codroid_picker_path.write_text(''.join(lines))
print(f"\\nWrote {len(lines)} lines to {codroid_picker_path}")
print(f"Original: 8762 lines")
print(f"New: {len(lines)} lines")
print(f"Removed: {8762 - len(lines)} lines")
