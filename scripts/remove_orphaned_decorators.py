#!/usr/bin/env python3
"""Remove orphaned @app.callback decorators and duplicate callbacks from main.py.

Orphaned decorators are those followed by another @app.callback instead of a function def.
Duplicate callbacks are functions that exist in both main.py and extracted modules.
"""

import re
from pathlib import Path

CODROID_PICKER = Path(__file__).parent.parent / "app" / "main.py"


def find_all_callbacks():
    """Parse all @app.callback blocks and their following functions."""
    content = CODROID_PICKER.read_text()
    lines = content.split('\n')

    callbacks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '@app.callback(' in line or '@ app.callback(' in line:
            decorator_start = i
            paren_count = line.count('(') - line.count(')')
            j = i + 1

            # Find where the decorator ends
            while j < len(lines) and paren_count > 0:
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1

            decorator_end = j - 1

            # Skip empty lines and orphaned fragments after decorator
            k = j
            orphaned_end = decorator_end
            while k < len(lines):
                next_line = lines[k].strip()
                if not next_line:
                    k += 1
                    continue
                # Check for orphaned content
                if next_line in (')', '}', '],', '):', '},'):
                    orphaned_end = k
                    k += 1
                    continue
                if 'prevent_initial_call' in next_line and '=' in next_line:
                    orphaned_end = k
                    k += 1
                    continue
                break

            # Now check what comes at position k
            func_start = None
            func_name = None
            func_end = None
            next_type = 'unknown'

            if k < len(lines):
                check_line = lines[k].strip()
                if check_line.startswith('@app.callback'):
                    next_type = 'decorator'
                elif check_line.startswith('@'):
                    next_type = 'other_decorator'
                    # Skip to the function after other decorators
                    while k < len(lines) and lines[k].strip().startswith('@'):
                        k += 1
                    if k < len(lines):
                        match = re.match(
                            r'\s*(?:async\s+)?def\s+(\w+)\s*\(', lines[k])
                        if match:
                            func_name = match.group(1)
                            func_start = k
                            next_type = 'function'
                elif check_line.startswith('def ') or check_line.startswith('async def '):
                    match = re.match(
                        r'\s*(?:async\s+)?def\s+(\w+)\s*\(', check_line)
                    if match:
                        func_name = match.group(1)
                        func_start = k
                        next_type = 'function'
                else:
                    next_type = 'other'

            # If we have a function, find where it ends
            if func_start is not None:
                func_indent = len(lines[func_start]) - \
                    len(lines[func_start].lstrip())

                # Find where the function signature ends (the colon after all params/return type)
                # Track parentheses and brackets to handle multi-line signatures
                m = func_start
                paren_count = 0
                bracket_count = 0
                found_colon = False

                while m < len(lines) and not found_colon:
                    line = lines[m]
                    for i, char in enumerate(line):
                        if char == '(':
                            paren_count += 1
                        elif char == ')':
                            paren_count -= 1
                        elif char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                        elif char == ':' and paren_count == 0 and bracket_count == 0:
                            # Found the function body colon
                            found_colon = True
                            break
                    m += 1

                # m now points to the line AFTER the function signature ends
                # Now find the actual end of the function body
                func_end = m - 1 if m > func_start else func_start
                while m < len(lines):
                    current_line = lines[m]
                    if not current_line.strip():
                        m += 1
                        continue
                    current_indent = len(current_line) - \
                        len(current_line.lstrip())
                    # Function ends when we hit same or lower indent level
                    if current_indent <= func_indent:
                        break
                    func_end = m
                    m += 1

            callbacks.append({
                'decorator_start': decorator_start,
                'decorator_end': orphaned_end,  # Include orphaned lines
                'func_start': func_start,
                'func_end': func_end,
                'func_name': func_name,
                'next_type': next_type,
                'orphaned': next_type == 'decorator',
            })

            if func_end is not None:
                i = func_end + 1
            else:
                i = orphaned_end + 1
        else:
            i += 1

    return callbacks, lines


def find_orphaned_regions():
    """Find regions containing orphaned decorators that should be removed."""
    callbacks, lines = find_all_callbacks()

    # Group consecutive orphaned decorators
    regions = []
    current_region = None

    for cb in callbacks:
        if cb['orphaned']:
            if current_region is None:
                current_region = {
                    'start': cb['decorator_start'], 'end': cb['decorator_end']}
            else:
                # Check if this decorator is adjacent or close to current region
                if cb['decorator_start'] <= current_region['end'] + 5:
                    current_region['end'] = cb['decorator_end']
                else:
                    regions.append(current_region)
                    current_region = {
                        'start': cb['decorator_start'], 'end': cb['decorator_end']}
        else:
            if current_region is not None:
                regions.append(current_region)
                current_region = None

    if current_region is not None:
        regions.append(current_region)

    return regions, lines, callbacks


# Functions that exist in extracted modules and should be removed from main file
DUPLICATE_FUNCTIONS = {
    '_ack_emergency_stop',
    '_advance_to_verification_after_accuracy',
    '_apply_gripper_orientation',
    '_apply_robot_speed',
    '_enable_rescue_mode',
    '_exit_rescue_mode',
    '_handle_camera_calibration_actions',
    '_handle_drag_not_allowed_response',
    '_handle_gripper',
    '_handle_joint_protection_response',
    '_handle_overspeed_response',
    '_handle_plate_actions',
    '_handle_power_toggle',
    '_handle_program_controls',
    '_handle_rescue_joint_jog',
    '_handle_safety_stop',
    '_handle_workspace_action',
    '_handle_wrong_tool_response',
    '_load_saved_robot_calibration',
    '_nudge_robot_speed',
    '_reset_calibration_after_accuracy',
    '_run_detection_action',
    '_run_robot_calibration_verification',
    '_scan_camera',
    '_start_robot_calibration',
    '_sync_footer_gripper_values',
    '_sync_operation_state',
    '_sync_power_toggle',
    '_sync_robot_calibration_accuracy_panel',
    '_sync_robot_calibration_state',
    '_sync_robot_controls',
    '_sync_robot_error_state',
    '_sync_robot_power_state',
    '_sync_robot_rescue_state',
    '_test_robot',
    '_toggle_camera',
    '_toggle_rescue_controls',
    '_toggle_robot_calibration_modal',
    '_update_calibration_progress',
    '_update_camera_view',
    '_update_detection_display',
    '_update_detection_parameters',
    '_update_drag_not_allowed_modal',
    '_update_emergency_modal',
    '_update_footer_flange_indicators',
    '_update_footer_robot_position',
    '_update_holding_badge',
    '_update_joint_protection_modal',
    '_update_overspeed_modal',
    '_update_program_controls',
    '_update_robot_calibration_guidance',
    '_update_robot_calibration_modal',
    '_update_robot_position_display',
    '_update_wrong_tool_modal',
}


def remove_orphaned_decorators(dry_run=True):
    """Remove orphaned decorator regions and duplicate callbacks."""
    regions, lines, callbacks = find_orphaned_regions()

    # Also find duplicate function callbacks
    duplicate_callbacks = []
    for cb in callbacks:
        if cb['func_name'] in DUPLICATE_FUNCTIONS and not cb['orphaned']:
            duplicate_callbacks.append({
                'name': cb['func_name'],
                'start': cb['decorator_start'],
                'end': cb['func_end'],
            })

    print(f"Found {len(regions)} orphaned decorator regions to remove:")
    for i, region in enumerate(regions):
        print(f"  Region {i+1}: lines {region['start']+1}-{region['end']+1}")
        for line_idx in range(region['start'], min(region['end']+1, region['start']+3)):
            print(f"    {line_idx+1}: {lines[line_idx][:60]}...")

    print(
        f"\nFound {len(duplicate_callbacks)} duplicate function callbacks to remove:")
    for cb in duplicate_callbacks:
        print(f"  {cb['name']}: lines {cb['start']+1}-{cb['end']+1}")

    if dry_run:
        print("\nDry run - no changes made. Run with --apply to remove.")
        return

    # Combine all regions to remove
    all_regions = []
    for region in regions:
        all_regions.append((region['start'], region['end']))
    for cb in duplicate_callbacks:
        all_regions.append((cb['start'], cb['end']))

    # Sort by start line descending (remove from bottom first to preserve line numbers)
    all_regions.sort(key=lambda x: x[0], reverse=True)

    # Remove regions from bottom to top (no merging - each region is independent)
    total_removed = 0
    for start, end in all_regions:
        del lines[start:end+1]
        total_removed += (end - start + 1)
        print(f"Removed lines {start+1}-{end+1} ({end - start + 1} lines)")

    # Write back
    new_content = '\n'.join(lines)
    CODROID_PICKER.write_text(new_content)

    print(
        f"\nTotal removed: {total_removed} lines. New file has {len(lines)} lines.")


if __name__ == "__main__":
    import sys
    dry_run = "--apply" not in sys.argv
    remove_orphaned_decorators(dry_run=dry_run)
