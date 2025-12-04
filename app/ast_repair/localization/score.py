"""
score.py

Combine multiple localization signals:
    - SBFL projection:       node_id → float
    - Stacktrace anchoring:  node_id → float
    - Backward slicing:      node_id → float

Output:
    node_id → total_score (float)

This module provides:
    1. combine_scores()      → merges & weights raw signals
    2. rank_nodes()          → sorted list of (node_id, total_score)
    3. pick_top_nodes()      → final top-K nodes to use for subtree extraction
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from app.ast_repair.metadata import ASTMetadata
from typing import cast



# ============================================================================
# Weight configuration (hand-tuned for SWE-bench-lite)
# ============================================================================

DEFAULT_WEIGHTS = {
    "sbfl":       1.0,   # base signal from spectrum-based fault localization
    "trace":      2.5,   # stacktrace is STRONGEST when available (increased from 1.5)
    "slice":      1.2,   # helpful for data-flow bugs (increased from 0.8)
    "semantic":   1.8,   # semantic similarity to issue (embedding-based retrieval)
    "structural": 2.0,   # structural neighbors of issue-mentioned identifiers (KG-lite)
    "negative":   1.0,   # negative signals penalty (boilerplate, pass-only, etc.)
    "size_pen":   0.3,   # penalize large spans
    "issue":      5.0,   # issue-mentioned methods get strong boost (NEW)
    "utility_pen": 0.1,  # utility methods get penalized (NEW)
}


# ============================================================================
# Score combination
# ============================================================================

def combine_scores(
    md: ASTMetadata,
    sbfl_scores: Dict[int, float],
    trace_scores: Dict[int, float],
    slice_scores: Dict[int, float],
    semantic_scores: Dict[int, float] = None,
    structural_scores: Dict[int, float] = None,
    negative_scores: Dict[int, float] = None,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> Dict[int, float]:
    """
    Merge the localization signals into one {node_id → total_score} dictionary.

    Missing signals cause no issue – empty dicts just contribute nothing.

    The formula per node:
        total = w_sbfl * sbfl
              + w_trace * trace
              + w_slice * slice
              + w_semantic * semantic
              + w_structural * structural
              - w_negative * negative
              - w_size  * size_penalty

    Penalty signals (subtracted):
        - negative: boilerplate, pass-only execution, too simple, etc.
        - size_penalty: (end - start + 1) normalized by file span
    """
    if semantic_scores is None:
        semantic_scores = {}
    if structural_scores is None:
        structural_scores = {}
    if negative_scores is None:
        negative_scores = {}

    node_ids = (
        set(sbfl_scores.keys()) | 
        set(trace_scores.keys()) | 
        set(slice_scores.keys()) | 
        set(semantic_scores.keys()) |
        set(structural_scores.keys()) |
        set(negative_scores.keys())
    )

    if not node_ids:
        return {}

    # Determine global file span for size normalization
    # Determine global file span for size normalization
    all_spans = [
    (md.start_lineno[nid], md.end_lineno[nid])
    for nid in md.node_index
    if md.start_lineno[nid] is not None and md.end_lineno[nid] is not None
    ]

    if all_spans:
        starts: List[int] = [cast(int, s) for s, _ in all_spans]
        ends:   List[int] = [cast(int, e) for _, e in all_spans]

        global_start = min(starts)
        global_end   = max(ends)
        global_span  = max(1, global_end - global_start + 1)
    else:
        global_span = 1

    scores = {}

    for nid in node_ids:
        sbfl_score  = sbfl_scores.get(nid, 0.0)
        trace_score = trace_scores.get(nid, 0.0)
        slice_score = slice_scores.get(nid, 0.0)
        semantic_score = semantic_scores.get(nid, 0.0)
        structural_score = structural_scores.get(nid, 0.0)
        negative_score = negative_scores.get(nid, 0.0)

        # --- size penalty ---
        start, end = md.line_map.get(nid, (None, None))
        if start is None or end is None:
            size_pen = 0.0
        else:
            span = end - start + 1
            size_pen = span / global_span

        total = (
            weights["sbfl"]       * sbfl_score
            + weights["trace"]    * trace_score
            + weights["slice"]    * slice_score
            + weights["semantic"] * semantic_score
            + weights["structural"] * structural_score
            - weights["negative"] * negative_score
            - weights["size_pen"] * size_pen
        )

        scores[nid] = total


    return scores


# ============================================================================
# Ranking utilities
# ============================================================================

def rank_nodes(scores: Dict[int, float]) -> List[Tuple[int, float]]:
    """
    Return list of (node_id, score) sorted by descending score.
    """
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def pick_top_nodes(
    scores: Dict[int, float],
    max_nodes: int = 5,
    min_score: Optional[float] = None,
) -> List[int]:
    """
    Select the top-K nodes that exceed an optional minimum score.

    This is what your orchestrator will use to determine which nodes become
    the initial subtree extraction points.
    """

    ranked = rank_nodes(scores)

    # Filter by score threshold if provided
    if min_score is not None:
        ranked = [pair for pair in ranked if pair[1] >= min_score]

    top = ranked[:max_nodes]
    return [nid for nid, _ in top]
