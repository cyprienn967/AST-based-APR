"""
stacktrace_anchor.py

Extracts localization signals from a Python traceback.

Given:
    - The traceback text (stderr from failing test run)
    - ASTMetadata (node spans)
    - The path of the file being localized

Returns:
    { node_id : float score }

Where:
    - Nodes whose span covers the deepest failing line get score 1.0
    - Their ancestors get small decreasing boosts
"""

from __future__ import annotations
import re
from typing import Dict, Optional, Tuple

from app.ast_repair.metadata import ASTMetadata


# ---------------------------------------------------------------------------
# Extract deepest Python frame for the user's project
# ---------------------------------------------------------------------------

TRACEBACK_LINE_RE = re.compile(
    r'File "([^"]+)", line (\d+), in ([A-Za-z0-9_<>]*)'
)

def extract_deepest_relevant_frame(traceback_text: str, project_root: str) -> Optional[Tuple[str, int]]:
    """
    Extract the deepest (last) frame that belongs to the target project.
    SWE-bench-lite test failures include full Python tracebacks.

    Returns:
        (file_path, line_number)  OR  None if not found.
    """

    matches = TRACEBACK_LINE_RE.findall(traceback_text)
    if not matches:
        return None

    # matches is a list of tuples: (file, line, function)
    # We want the *last* one that is not in tests
    for file_path, line_str, _func in reversed(matches):
        line = int(line_str)

        # Ignore test files ― they are irrelevant as fault locations
        if "test" in file_path.lower():
            continue

        # Optional: if project_root is given, ensure file is inside project
        if project_root and project_root not in file_path:
            continue

        return file_path, line

    return None


# ---------------------------------------------------------------------------
# Map frame → AST nodes via line spans
# ---------------------------------------------------------------------------

def nodes_covering_line(md: ASTMetadata, line: int):
    """
    Return all node_ids whose (start_line, end_line) span covers the given line.
    """
    result = []
    for nid, (start, end) in md.line_map.items():
        if start is None or end is None:
            continue
        if start <= line <= end:
            result.append(nid)
    return result


# ---------------------------------------------------------------------------
# Main scoring entrypoint
# ---------------------------------------------------------------------------

def stacktrace_anchor(
    traceback_text: str,
    md: ASTMetadata,
    project_root: str = ""
) -> Dict[int, float]:
    """
    Produce {node_id → score} from traceback text.

    Strategy:
        1. Extract deepest non-test frame
        2. Find all nodes covering that line
        3. Assign:
            - smallest node: 1.0
            - its ancestors: decreasing weights
            - other nodes: 0

    If no traceback hits a project file, return empty dict.
    """

    frame = extract_deepest_relevant_frame(traceback_text, project_root)
    if frame is None:
        return {}

    file_path, line = frame

    covering = nodes_covering_line(md, line)
    if not covering:
        return {}

    # Pick the "smallest" node = minimum span length
    def span_length(nid):
        start, end = md.line_map[nid]
        return (end - start) if (start is not None and end is not None) else float("inf")

    smallest = min(covering, key=span_length)

    node_scores: Dict[int, float] = {}

    # Direct hit gets 1.0
    node_scores[smallest] = 1.0

    # Boost ancestors: e.g. 0.6, 0.3, 0.1
    weight = 0.6
    current = md.parent.get(smallest)

    while current is not None:
        node_scores[current] = weight
        weight *= 0.5  # decay
        current = md.parent.get(current)

    return node_scores
