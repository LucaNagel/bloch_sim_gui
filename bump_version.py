#!/usr/bin/env python3
"""
bump_version.py - Automatically update version numbers across the project.

Usage:
    python bump_version.py 1.0.2
"""

import sys
import re
from pathlib import Path

def update_file(path, pattern, replacement):
    if not path.exists():
        print(f"Warning: {path} not found. Skipping.")
        return False
    
    content = path.read_text()
    new_content = re.sub(pattern, replacement, content)
    
    if content != new_content:
        path.write_text(new_content)
        print(f"✓ Updated {path}")
        return True
    else:
        print(f"ℹ No changes needed in {path}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py <new_version>")
        print("Example: python bump_version.py 1.0.2")
        sys.exit(1)

    new_version = sys.argv[1]
    
    # 1. pyproject.toml
    # pattern: version = "1.0.1"
    update_file(
        Path("pyproject.toml"),
        r'version = "[0-9.]+"',
        f'version = "{new_version}"'
    )

    # 2. setup.py
    # pattern: version="1.0.1",
    update_file(
        Path("setup.py"),
        r'version="[0-9.]+"',
        f'version="{new_version}"'
    )

    # 3. simulator.py
    # pattern: 'simulator_version' = '1.0.1'
    update_file(
        Path("src/blochsimulator/simulator.py"),
        r"'simulator_version': '[0-9.]+'",
        f"'simulator_version': '{new_version}'"
    )
    update_file(
        Path("src/blochsimulator/simulator.py"),
        r"attrs['simulator_version'] = '[0-9.]+'",
        f"attrs['simulator_version'] = '{new_version}'"
    )

    # 4. gui.py (About dialog)
    # pattern: "Version 1.0.1"
    update_file(
        Path("src/blochsimulator/gui.py"),
        r'Version [0-9.]+',
        f'Version {new_version}'
    )

    # 5. docs/conf.py
    # pattern: release = '1.0.1'
    update_file(
        Path("docs/conf.py"),
        r"release = '[0-9.]+'",
        f"release = '{new_version}'"
    )

    print(f"\nDone! Project version bumped to {new_version}")
    print("Next steps:")
    print(f"  git add .")
    print(f"  git commit -m \"Bump version to {new_version}\"")
    print(f"  git tag v{new_version}")
    print(f"  git push origin main v{new_version}")

if __name__ == "__main__":
    main()
