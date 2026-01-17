#!/usr/bin/env python3
"""Remove extracted UI builder functions from main.py."""
from pathlib import Path

source = Path("app/main.py")
content = source.read_text()
lines = content.split('\n')

# Line ranges to remove (1-indexed, inclusive start, exclusive end)
# Must be removed from bottom to top to preserve line numbers
ranges_to_remove = [
    # Main block of builders (lines 4051-6555)
    (4051, 6555),
    # build_tracking_mode_controls (lines 236-282)
    (236, 282),
]

# Sort by start line descending (remove from bottom first)
ranges_to_remove.sort(reverse=True)

print(f"Original file: {len(lines)} lines")

for start, end in ranges_to_remove:
    # Convert to 0-indexed
    del lines[start-1:end-1]
    print(f"Removed lines {start}-{end-1} ({end - start} lines)")

# Write back
new_content = '\n'.join(lines)
source.write_text(new_content)
print(f"New file: {len(lines)} lines")
print(f"Removed: {11729 - len(lines)} lines total")
