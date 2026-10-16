"""Filename policies shared by session exports and direct parameter-loop saves."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def project_timestamp(created_at_utc):
    """Format a completion time for a local export filename."""
    try:
        completed = datetime.fromisoformat(str(created_at_utc))
        if completed.tzinfo is not None:
            completed = completed.astimezone()
        return completed.strftime("%Y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class ProjectExportNaming:
    mode: str = "existing"
    prefix: str = "simulation"
    start_number: int = 1
    digits: int = 2

    def validated_prefix(self):
        if self.mode not in {"existing", "timestamp", "numbered"}:
            raise ValueError("Select a filename format.")
        if self.start_number < 0 or not 1 <= self.digits <= 9:
            raise ValueError("Use a nonnegative start number and 1 to 9 digits.")
        if self.mode == "existing":
            return ""
        prefix = self.prefix.strip().rstrip("_")
        if not prefix or prefix in {".", ".."}:
            raise ValueError("Enter a filename prefix, for example simulation.")
        if re.search(r'[<>:"/\\|?*\x00-\x1f]', prefix):
            raise ValueError(
                'The prefix must be a filename, without paths or <>:"/\\|?*.'
            )
        return prefix


class ProjectExportNamer:
    """Allocate distinct names, including before any batch file has been written."""

    def __init__(self, directory, naming=None):
        self.directory = Path(directory)
        self.naming = naming or ProjectExportNaming()
        self.prefix = self.naming.validated_prefix()
        self.next_number = self.naming.start_number
        self._reserved_names = set()

    def _available(self, path):
        return path.name.casefold() not in self._reserved_names and not path.exists()

    def next_path(self, existing_stem, timestamp):
        if self.naming.mode == "numbered":
            while True:
                stem = f"{self.prefix}_{self.next_number:0{self.naming.digits}d}"
                self.next_number += 1
                path = self.directory / f"{stem}.blochproj"
                if self._available(path):
                    break
        else:
            stem = (
                f"{self.prefix}_{timestamp}"
                if self.naming.mode == "timestamp"
                else existing_stem
            )
            if not stem or re.search(r'[<>:"/\\|?*\x00-\x1f]', stem):
                raise ValueError("Invalid project filename.")
            path = self.directory / f"{stem}.blochproj"
            suffix = 2
            while not self._available(path):
                path = self.directory / f"{stem}_{suffix}.blochproj"
                suffix += 1
        self._reserved_names.add(path.name.casefold())
        return path
