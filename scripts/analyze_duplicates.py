#!/usr/bin/env python3
"""Analyze main.py to find orphaned decorators and duplicate callbacks."""

import re
from pathlib import Path

CODROID_PICKER = Path(__file__).parent.parent / "app" / "main.py"


def find_orphaned_decorators():
    """Find @app.callback decorators that are not followed by a function definition."""
    content = CODROID_PICKER.read_text()
    lines = content.split('\n')

    orphaned = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '@app.callback(' in line:
            # Found a decorator, find where it ends
            decorator_start = i
            paren_count = line.count('(') - line.count(')')
            j = i + 1

            # Find the end of the decorator
            while j < len(lines) and paren_count > 0:
                paren_count += lines[j].count('(') - lines[j].count(')')
                j += 1

            decorator_end = j - 1

            # Check if next non-empty/non-decorator line is a function def
            k = j
            while k < len(lines):
                next_line = lines[k].strip()
                if not next_line:
                    k += 1
                    continue
                if next_line.startswith('@'):
                    # Another decorator - check if it's app.callback (stacked decorators)
                    if '@app.callback' in next_line:
                        # This means the previous decorator is orphaned
                        orphaned.append({
                            'start': decorator_start + 1,
                            'end': decorator_end + 1,
                            'preview': lines[decorator_start:decorator_start+3]
                        })
                    break
                elif next_line.startswith('def ') or next_line.startswith('async def '):
                    # Found a function, decorator is valid
                    break
                else:
                    # Something else - could be orphaned content
                    orphaned.append({
                        'start': decorator_start + 1,
                        'end': decorator_end + 1,
                        'preview': lines[decorator_start:decorator_start+3],
                        'followed_by': next_line[:50]
                    })
                    break
                k += 1
        i += 1

    return orphaned


def find_callbacks_in_file(filepath):
    """Find all callback functions in a file with their Outputs."""
    content = Path(filepath).read_text()
    lines = content.split('\n')

    callbacks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '@app.callback(' in line:
            decorator_start = i
            paren_count = line.count('(') - line.count(')')
            decorator_lines = [line]
            j = i + 1

            while j < len(lines) and paren_count > 0:
                paren_count += lines[j].count('(') - lines[j].count(')')
                decorator_lines.append(lines[j])
                j += 1

            decorator_end = j - 1

            # Find the function definition
            k = j
            func_name = None
            while k < len(lines):
                next_line = lines[k].strip()
                if not next_line:
                    k += 1
                    continue
                if next_line.startswith('@'):
                    k += 1
                    continue
                match = re.match(r'(?:async\s+)?def\s+(\w+)\s*\(', next_line)
                if match:
                    func_name = match.group(1)
                break

            # Extract outputs
            decorator_text = '\n'.join(decorator_lines)
            outputs = re.findall(
                r'Output\(["\']([^"\']+)["\']', decorator_text)

            if func_name:
                callbacks.append({
                    'name': func_name,
                    'outputs': outputs,
                    'start': decorator_start + 1,
                    'line': i + 1,
                    'file': str(filepath)
                })
        i += 1

    return callbacks


def main():
    print("=" * 70)
    print("ANALYSIS OF main.py")
    print("=" * 70)

    # Find orphaned decorators
    print("\n1. ORPHANED DECORATORS (decorators followed by more decorators):")
    print("-" * 70)
    orphaned = find_orphaned_decorators()
    if orphaned:
        for item in orphaned:
            print(f"  Line {item['start']}-{item['end']}:")
            for line in item['preview']:
                print(f"    {line}")
            if 'followed_by' in item:
                print(f"    Followed by: {item['followed_by']}")
            print()
    else:
        print("  None found")

    # Find callbacks in main file
    print("\n2. CALLBACKS IN main.py:")
    print("-" * 70)
    main_callbacks = find_callbacks_in_file(CODROID_PICKER)
    for cb in main_callbacks:
        print(
            f"  Line {cb['line']}: {cb['name']}() -> {', '.join(cb['outputs'][:2])}")
    print(f"\n  Total: {len(main_callbacks)} callbacks")

    # Find callbacks in extracted modules
    print("\n3. CALLBACKS IN EXTRACTED MODULES:")
    print("-" * 70)

    extracted_modules = [
        Path(__file__).parent.parent / "app" /
        "codroid_picker" / "camera_callbacks.py",
    ]

    robot_callbacks_dir = Path(__file__).parent.parent / \
        "app" / "codroid_picker" / "robot_callbacks"
    for f in robot_callbacks_dir.glob("*.py"):
        if f.name != "__init__.py":
            extracted_modules.append(f)

    extracted_callbacks = []
    for module in extracted_modules:
        if module.exists():
            cbs = find_callbacks_in_file(module)
            for cb in cbs:
                extracted_callbacks.append(cb)
                print(f"  {module.name}: {cb['name']}()")

    print(
        f"\n  Total: {len(extracted_callbacks)} callbacks in extracted modules")

    # Find duplicates
    print("\n4. DUPLICATE FUNCTIONS (exist in both main file and extracted modules):")
    print("-" * 70)
    main_names = {cb['name'] for cb in main_callbacks}
    extracted_names = {cb['name'] for cb in extracted_callbacks}
    duplicates = main_names & extracted_names

    if duplicates:
        for name in sorted(duplicates):
            main_cb = next(cb for cb in main_callbacks if cb['name'] == name)
            ext_cb = next(
                cb for cb in extracted_callbacks if cb['name'] == name)
            print(f"  {name}:")
            print(f"    - Main file line: {main_cb['line']}")
            print(f"    - Extracted to: {Path(ext_cb['file']).name}")
    else:
        print("  None found")

    print(f"\n  Total duplicates: {len(duplicates)}")


if __name__ == "__main__":
    main()
