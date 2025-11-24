
"""
Module to compute unified diffs between original and patched source code.

Responsibilities:
    • Accept old source and new source as strings
    • Produce a unified diff string suitable for SWE-bench consumption

Usage:
    from app.ast_repair.diff import unified_diff_str
    diff = unified_diff_str(old_source, new_source, filename)

Notes:
    - We use `difflib.unified_diff` which produces standard unified diff format.
    - The filename is optional; when provided, it becomes the diff header.
"""

import difflib
from typing import Optional


def unified_diff_str(
    old_source: str,
    new_source: str,
    filename: Optional[str] = None,
) -> str:
    """
    Produce a unified diff between old_source and new_source.

    Parameters:
        old_source: str
        new_source: str
        filename: Optional[str]
            Optional filename to display in diff headers.

    Returns:
        A unified diff string (may be empty if no differences).
    """
    old_lines = old_source.splitlines(keepends=True)
    new_lines = new_source.splitlines(keepends=True)

    # difflib.unified_diff expects path1/path2 but filename applies to both.
    file_a = filename or "original"
    file_b = filename or "patched"

    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=file_a,
        tofile=file_b,
        lineterm="",
    )

    # Join into a single string
    return "\n".join(diff_lines) + "\n"
