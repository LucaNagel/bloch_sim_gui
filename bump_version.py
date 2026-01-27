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
        print(f"[OK] Updated {path}")
        return True
    else:
        print(f"[-] No changes needed in {path}")
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
        Path("pyproject.toml"), r'version = "[0-9.]+"', f'version = "{new_version}"'
    )

    # 2. setup.py
    # pattern: version="1.0.1",
    update_file(Path("setup.py"), r'version="[0-9.]+"', f'version="{new_version}"')

    # 3. gui.py (About dialog)
    # pattern: "Version 1.0.1"
    update_file(
        Path("src/blochsimulator/gui.py"), r"Version [0-9.]+", f"Version {new_version}"
    )

    # 4. docs/conf.py
    # pattern: release = "1.0.1" or release = '1.0.1'
    update_file(
        Path("docs/conf.py"),
        r"release = ['\"][0-9.]+['\"]",
        f'release = "{new_version}"',
    )

    # 5. src/blochsimulator/__init__.py
    # pattern: __version__ = "1.0.1"
    update_file(
        Path("src/blochsimulator/__init__.py"),
        r'__version__ = "[0-9.]+"',
        f'__version__ = "{new_version}"',
    )

    # 6. web/partials/home.html
    # pattern: >v1.0.1</a>
    # pattern: tag/v1.0.1"
    update_file(
        Path("web/partials/home.html"),
        r">v[0-9.]+</a>",
        f">v{new_version}</a>",
    )
    update_file(
        Path("web/partials/home.html"),
        r"tag/v[0-9.]+\"",
        f'tag/v{new_version}"',
    )

    # 7. web/partials/footer.html
    # pattern: >v1.0.1</span>
    update_file(
        Path("web/partials/footer.html"),
        r">v[0-9.]+</span>",
        f">v{new_version}</span>",
    )

    print(f"\nDone! Project version bumped to {new_version}")
    print("Next steps:")
    print(f"  git add .")
    print(f'  git commit -m "Bump version to {new_version}"')
    print(f"  git tag v{new_version}")
    print(f"  git push origin main v{new_version}")


if __name__ == "__main__":
    main()
