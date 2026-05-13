"""
File I/O safety utilities: atomic writes and periodic cleanup of outputs.

Why:
- Direct writes to `outputs/*.tif` leave torn files if the process is killed
  mid-write. A subsequent request can then read a truncated file.
- The outputs directory grew unbounded — disk fills up over time.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Optional


def atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes to `path` atomically.

    Writes to a temp file in the same directory (so `os.replace` stays on the
    same filesystem, which is atomic on POSIX) and renames on success.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # some filesystems don't support fsync; ignore
                pass
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def temp_path_for(path: str) -> str:
    """Return a sibling `.partial` path for callers that must write through
    a library (e.g. rasterio) that opens paths directly. The caller writes to
    this path and calls `os.replace(tmp, path)` when done.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    base = os.path.basename(path)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f".partial_{os.getpid()}_{base}")


def cleanup_old_files(
    directory: str,
    max_age_hours: float = 24.0,
    suffixes: Optional[tuple[str, ...]] = None,
    dry_run: bool = False,
) -> int:
    """Delete files in `directory` older than `max_age_hours`.

    Returns the number of files deleted (or that would be deleted, if dry_run).
    Silently skips files it cannot stat or unlink.
    Does NOT recurse into subdirectories.
    """
    if not os.path.isdir(directory):
        return 0
    now = time.time()
    max_age_sec = max_age_hours * 3600.0
    count = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if not os.path.isfile(path):
                continue
            if suffixes and not name.endswith(suffixes):
                continue
            age = now - os.path.getmtime(path)
            if age < max_age_sec:
                continue
            if dry_run:
                count += 1
                continue
            os.unlink(path)
            count += 1
        except OSError:
            continue
    return count


