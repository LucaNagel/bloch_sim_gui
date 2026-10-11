"""Extract one changelog entry for a GitHub Release."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, for example v2.8.1")
    parser.add_argument("output", type=Path, help="Output Markdown file")
    arguments = parser.parse_args()
    version = arguments.tag.removeprefix("v")
    changelog = (Path(__file__).resolve().parents[1] / "CHANGELOG.md").read_text()
    entry = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        changelog,
        flags=re.MULTILINE | re.DOTALL,
    )
    if entry is None:
        parser.error(f"No changelog entry for {version}")
    notes = entry.group(1).strip()
    source = f"https://github.com/LucaNagel/bloch_sim_gui/blob/{arguments.tag}/"
    notes = re.sub(r"\]\((docs/[^)]+)\)", rf"]({source}\1)", notes)
    arguments.output.write_text(notes + "\n")


if __name__ == "__main__":
    main()
