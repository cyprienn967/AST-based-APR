"""
SBFL → AST node projection.

Takes:
    - AST root
    - Metadata (node ranges, parent/child maps)
    - SBFL suspiciousness (line → score)

Produces:
    { node_id : score }  (float per node)

This is one of the three "big hitter" localization signals:
    1. SBFL→AST projection
    2. Stack-trace anchoring
    3. Backward slicing
"""

from typing import Dict, List, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: determine whether a line number falls inside a node's span.
# Your metadata must provide: metadata.start_lineno[node_id], metadata.end_lineno[node_id]
# ---------------------------------------------------------------------------

def line_in_node(node_id: int, line: int, md) -> bool:
    start = md.start_lineno.get(node_id)
    end = md.end_lineno.get(node_id)

    if start is None or end is None:
        return False

    return start <= line <= end


# ---------------------------------------------------------------------------
# Main projection function
# ---------------------------------------------------------------------------

def project_sbfl_to_nodes(
    sbfl_scores: Dict[int, float],
    md,
) -> Dict[int, float]:
    """
    Args:
        sbfl_scores: dict mapping line → suspiciousness (float)
        md: ASTMetadata object from parser.py / metadata.py
            MUST expose:
                - md.node_index: { node_id → ast.AST node }
                - md.start_lineno[node_id]
                - md.end_lineno[node_id]
                - md.children[node_id]    (optional but recommended)
                - md.parent[node_id]      (optional but recommended)

    Returns:
        node_scores: dict mapping node_id → float
    """

    node_scores = defaultdict(float)

    # --- Step 1: raw assignment of SBFL scores to nodes by line overlap ---
    for line, score in sbfl_scores.items():
        for node_id in md.node_index.keys():
            if line_in_node(node_id, line, md):
                node_scores[node_id] += score

    # --- Step 2: structural smoothing -------------------------------------
    # The smoothing formula is:
    #    score(node) = α * raw(node)
    #                 + β * avg(child scores)
    #                 + γ * parent score
    #
    # α, β, γ are hand-picked heuristics. You do NOT need training.
    # These values are conservative and work well for SWE-bench-lite.

    alpha = 1.0   # raw contribution
    beta = 0.5    # average child influence
    gamma = 0.25  # parent influence

    smoothed = defaultdict(float)

    # NOTE: optionally, run multiple passes.
    # But for SWE-bench-lite, one pass is sufficient.
    for node_id in md.node_index.keys():

        raw = node_scores[node_id]

        # Child influence
        child_ids = getattr(md, "children", {}).get(node_id, [])
        if child_ids:
            child_avg = sum(node_scores[c] for c in child_ids) / len(child_ids)
        else:
            child_avg = 0.0

        # Parent influence
        parent_id = getattr(md, "parent", {}).get(node_id)
        if parent_id is not None:
            parent_raw = node_scores[parent_id]
        else:
            parent_raw = 0.0

        smoothed[node_id] = (
            alpha * raw +
            beta * child_avg +
            gamma * parent_raw
        )

    return dict(smoothed)


# ---------------------------------------------------------------------------
# Public API: reducer wrapper
# ---------------------------------------------------------------------------

def sbfl_project(
    sbfl_line_scores: Dict[int, float],
    root,
    metadata
) -> Dict[int, float]:
    """
    Thin wrapper that:
        - normalizes SBFL input (e.g., remove zero-suspiciousness lines)
        - calls the projection function
        - returns node → score mapping

    Args:
        sbfl_line_scores: { line_number → float score }
        root:   the AST root node (not used in this file, but kept for API symmetry)
        metadata: ASTMetadata from parser.py

    Returns:
        node_scores: { node_id → float }
    """

    # Optional: prune zero-suspicion lines to save work
    cleaned = {ln: sc for ln, sc in sbfl_line_scores.items() if sc > 0.0}

    return project_sbfl_to_nodes(cleaned, metadata)
