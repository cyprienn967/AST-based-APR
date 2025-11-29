"""
localize.py

This is the SINGLE orchestrator for AST-based localization.

Pipeline:
    1. SBFL → AST node projection
    2. Stacktrace anchoring (if traceback exists)
    3. Backward slicing (from failing line)
    4. Score fusion (weights, size penalty)
    5. Pick top-K suspicious nodes
    6. Extract the minimal subtree for each suspicious node
    7. Provide subtree expansion strategy

Used by:
    • Person B (AST-edit LLM agent)
    • Patch manager in ACR pipeline
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import ast

from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.localization.sbfl_project import sbfl_project
from app.ast_repair.localization.stacktrace_anchor import stacktrace_anchor
from app.ast_repair.localization.backward_slice import backward_slice
from app.ast_repair.localization.score import combine_scores, pick_top_nodes


# -----------------------------------------------------------------------------
# Helper: extract subtree given a node_id
# -----------------------------------------------------------------------------

def extract_subtree(md: ASTMetadata, node_id: int) -> ast.AST:
    """
    Return the AST subtree rooted at node_id.
    Person B will give edits relative to this subtree.

    NOTE:
        We retrieve the node directly from md.node_index.
        We do NOT deep-copy here; the edit pipeline will apply modifications
        through node_id → AST node references.
    """
    return md.node_index[node_id]


# -----------------------------------------------------------------------------
# Helper: expansion strategy (expr → stmt → block → function → file)
# -----------------------------------------------------------------------------

def expansion_chain(md: ASTMetadata, nid: int) -> List[int]:
    """
    Return a list of node_ids representing the expansion schedule:
        start at smallest node (expression)
        → parent statement
        → parent block/if/loop
        → parent function
        → file/module
    """
    chain = [nid]

    current = md.parent.get(nid)
    while current is not None:
        chain.append(current)
        current = md.parent.get(current)

    return chain  # smallest → largest


# -----------------------------------------------------------------------------
# BugLocation record returned to Person B
# -----------------------------------------------------------------------------

class BugLocation:
    """
    Lightweight container:
        - node_id: AST node_id
        - subtree: actual AST subtree
        - expansion_chain: list of node_ids (smallest → largest)
    """
    def __init__(self, node_id: int, subtree: ast.AST, expansion_chain: List[int]):
        self.node_id = node_id
        self.subtree = subtree
        self.expansion_chain = expansion_chain

    def __repr__(self):
        return f"BugLocation(node_id={self.node_id}, chain_len={len(self.expansion_chain)})"


# -----------------------------------------------------------------------------
# Main orchestrator API
# -----------------------------------------------------------------------------

def localize_fault(
    root: ast.AST,
    md: ASTMetadata,
    sbfl_line_scores: Dict[int, float],
    traceback_text: str,
    failing_line: Optional[int],
    project_root: str = "",
    top_k: int = 3,
) -> List[BugLocation]:
    """
    Combine SBFL, stacktrace, slicing → return top-K BugLocation objects.

    Args:
        root             - AST root node
        md               - ASTMetadata
        sbfl_line_scores - line → score from SBFL
        traceback_text   - stderr from failing test (may be empty)
        failing_line     - deepest failing line number extracted previously
        project_root     - for filtering stacktrace frames
        top_k            - how many suspicious nodes to return

    Returns:
        List[BugLocation]
    """

    # 1. --- SBFL → AST -------------------------------------------------------
    sbfl_scores = sbfl_project(sbfl_line_scores, root, md)

    # 2. --- Stacktrace anchoring --------------------------------------------
    trace_scores = {}
    if traceback_text:
        trace_scores = stacktrace_anchor(traceback_text, md, project_root=project_root)

    # 3. --- Backward slicing -------------------------------------------------
    slice_scores = {}
    if failing_line is not None:
        slice_scores = backward_slice(md, failing_line)

    # 4. --- Combine into final scores ---------------------------------------
    combined = combine_scores(
        md,
        sbfl_scores=sbfl_scores,
        trace_scores=trace_scores,
        slice_scores=slice_scores,
    )

    if not combined:
        # Fallback: when we have zero localization signals (no SBFL, no traceback, no failing line),
        # use uniform scoring over statement-level nodes to avoid complete failure.
        # This ensures we can still attempt repairs even with minimal information.
        import logging
        logging.warning("No localization signals available; using uniform fallback scoring")
        
        fallback_scores = {}
        for nid, node in md.node_index.items():
            # Score statement-level and definition nodes (not tiny expressions)
            # These are the most likely edit targets
            if isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.If, ast.For, ast.While, ast.With, ast.Try,
                ast.Assign, ast.AugAssign, ast.AnnAssign,
                ast.Return, ast.Raise, ast.Assert,
                ast.Expr  # Expression statements (often function calls)
            )):
                # Give a weak uniform score
                # Prefer smaller nodes (more specific)
                start, end = md.line_map.get(nid, (None, None))
                if start is not None and end is not None:
                    span = end - start + 1
                    # Inverse of span: smaller nodes get higher score
                    fallback_scores[nid] = 1.0 / max(span, 1)
        
        if fallback_scores:
            combined = fallback_scores
        else:
            # Truly no viable candidates
            return []

    # 5. --- Pick top-K suspicious node_ids ----------------------------------
    top_nodes = pick_top_nodes(combined, max_nodes=top_k)

    # 6. --- Build BugLocation objects ----------------------------------------
    bug_locations: List[BugLocation] = []

    for nid in top_nodes:
        subtree = extract_subtree(md, nid)
        chain = expansion_chain(md, nid)
        bug_locations.append(BugLocation(nid, subtree, chain))

    return bug_locations
