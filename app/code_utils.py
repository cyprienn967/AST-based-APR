from pathlib import Path


def is_test_file(file_path: str) -> bool:
    """
    Heuristic to determine whether a path points to a test file.
    """
    path = Path(file_path)
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    return (
        "test" in parts
        or "tests" in parts
        or name.endswith("_test.py")
        or name.startswith("test_")
    )


def read_code_snippet(
    file_full_path: str, start: int, end: int, with_lineno: bool = True
) -> str:
    """
    Read a snippet of code between start and end lines (inclusive).
    """
    path = Path(file_full_path)
    if not path.exists() or start < 1 or end < start:
        return ""

    lines = path.read_text().splitlines(keepends=True)
    if start > len(lines):
        return ""

    snippet_parts: list[str] = []
    start_idx = max(start - 1, 0)
    end_idx = min(end, len(lines))

    for idx in range(start_idx, end_idx):
        prefix = f"{idx + 1} " if with_lineno else ""
        snippet_parts.append(f"{prefix}{lines[idx]}")

    return "".join(snippet_parts)
