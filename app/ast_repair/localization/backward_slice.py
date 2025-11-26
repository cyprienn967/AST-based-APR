"""
backward_slice.py

A lightweight backward slicing implementation for SWE-bench-lite.

Goal:
    From the failing location (assertion or exception), walk backwards through
    the AST using a simple def-use graph:
        - Name → where it was defined (Assign, Function arg, etc.)
        - Propagate through parent statements
        - Produce node_id → score, where score ∈ {0.0, 1.0}

This is NOT full static analysis.
It's a deliberately shallow, robust signal:
    • If a variable is used in the failing expression, score all statements
      that define or modify it.
    • Score enclosing statements, loops, conditionals.

This complements:
    • SBFL projection
    • Stack-trace anchoring
"""

from __future__ import annotations
import ast
from typing import Dict, Set, List, Optional

from app.ast_repair.metadata import ASTMetadata


# -----------------------------------------------------------------------------
# Utilities for def-use extraction
# -----------------------------------------------------------------------------

def collect_name_uses(node: ast.AST) -> Set[str]:
    """
    Return all variable names *used* inside this node.
    """
    names = set()

    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            names.add(sub.id)

    return names


def collect_name_defs(node: ast.AST) -> Set[str]:
    """
    Return all variable names *defined/assigned* inside this node.
    """
    defs = set()

    # Assign, AnnAssign, AugAssign
    if isinstance(node, ast.Assign):
        for target in node.targets:
            defs |= _extract_lvalue_names(target)

    elif isinstance(node, ast.AnnAssign):
        defs |= _extract_lvalue_names(node.target)

    elif isinstance(node, ast.AugAssign):
        defs |= _extract_lvalue_names(node.target)

    # Function args
    elif isinstance(node, ast.FunctionDef):
        for arg in node.args.args:
            defs.add(arg.arg)

    return defs


def _extract_lvalue_names(t) -> Set[str]:
    """Helper for extracting names on LHS of assignments."""
    if isinstance(t, ast.Name):
        return {t.id}
    elif isinstance(t, ast.Attribute):
        # For x.y = ... we treat x as defined (good enough for SWE-lite)
        if isinstance(t.value, ast.Name):
            return {t.value.id}
    elif isinstance(t, (ast.Tuple, ast.List)):
        names = set()
        for elt in t.elts:
            names |= _extract_lvalue_names(elt)
        return names
    return set()


# -----------------------------------------------------------------------------
# Identify failing AST node from line number
# -----------------------------------------------------------------------------

def smallest_node_covering_line(md: ASTMetadata, line: int) -> Optional[int]:
    """
    Find the smallest node spanning 'line' by minimal (end - start).
    """
    best = None
    best_span = float("inf")

    for nid, (start, end) in md.line_map.items():
        if start is None or end is None:
            continue
        if start <= line <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best = nid

    return best


# -----------------------------------------------------------------------------
# Main backward slicing
# -----------------------------------------------------------------------------

def backward_slice(
    md: ASTMetadata,
    failing_line: int,
) -> Dict[int, float]:
    """
    Return a dict {node_id → 1.0} for all nodes that affect the failing location.

    Steps:
        1. Identify the smallest node covering the failing line.
        2. Extract all names used in that node.
        3. For each name, find all statements that define it.
        4. Walk upward to include enclosing statements / blocks.

    This is intentionally coarse but extremely effective for SWE-bench-lite bugs
    involving conditional logic, off-by-one, wrong variable usage, etc.
    """

    # 1. Locate failing AST node
    failing_node_id = smallest_node_covering_line(md, failing_line)
    if failing_node_id is None:
        return {}

    failing_node = md.get_node_by_id(failing_node_id)

    # 2. Extract variable uses from failing node
    used_vars = collect_name_uses(failing_node)
    if not used_vars:
        # If no variables, just score failing node + ancestors
        return _ancestors_only(md, failing_node_id)

    # 3. Find all nodes that define these variables
    defining_nodes = set()

    for nid, node in md.node_index.items():
        defs = collect_name_defs(node)
        if defs & used_vars:
            defining_nodes.add(nid)

    # 4. Collect ancestors (control structure context)
    slice_nodes: Set[int] = set()
    for nid in defining_nodes:
        slice_nodes.add(nid)
        # add ancestors up to root
        curr = md.parent.get(nid)
        while curr is not None:
            slice_nodes.add(curr)
            curr = md.parent.get(curr)

    # Always include the failing node itself + ancestors
    slice_nodes |= set(_ancestors_only(md, failing_node_id).keys())

    # Return as 1.0 for all nodes in slice
    return {nid: 1.0 for nid in slice_nodes}


# -----------------------------------------------------------------------------
# Helper: failing-node-only slice (no vars)
# -----------------------------------------------------------------------------

def _ancestors_only(md: ASTMetadata, nid: int) -> Dict[int, float]:
    """
    Return {nid → 1.0} for nid and all ancestors of nid.
    """
    scores = {nid: 1.0}
    curr = md.parent.get(nid)

    while curr is not None:
        scores[curr] = 1.0
        curr = md.parent.get(curr)

    return scores
