"""
Localization submodule for AST-based fault localization.

This module combines multiple signals to identify the most likely buggy nodes:
    - SBFL projection: Maps line-level scores to AST nodes
    - Stacktrace anchoring: Boosts nodes matching stacktrace frames
    - Backward slicing: Identifies nodes in the data-flow slice
    - Semantic retrieval: Embedding-based similarity to issue text
    - Issue boosting: Boosts nodes whose names match issue text

The main entry point is `localize_fault()` in the parent `localize.py` module.
"""

from app.ast_repair.localization.sbfl_project import sbfl_project
from app.ast_repair.localization.stacktrace_anchor import stacktrace_anchor
from app.ast_repair.localization.backward_slice import backward_slice
from app.ast_repair.localization.score import combine_scores, pick_top_nodes, rank_nodes
from app.ast_repair.localization.issue_boost import (
    boost_nodes_from_issue,
    extract_method_names_from_issue,
    extract_class_names_from_issue,
    is_utility_method,
)
from app.ast_repair.localization.semantic_retrieval import (
    semantic_retrieval_scores,
    compute_semantic_scores,
)
from app.ast_repair.localization.structural_boost import (
    structural_boost_scores,
    compute_structural_boost,
    extract_identifiers_from_issue,
    build_symbol_table,
    build_call_graph,
)
from app.ast_repair.localization.negative_signals import (
    negative_signal_scores,
    compute_negative_signals,
    NegativeSignals,
    is_boilerplate_method,
    is_test_or_debug_code,
    compute_complexity_signals,
    is_in_except_handler,
    compute_docstring_quality,
)

__all__ = [
    # SBFL
    "sbfl_project",
    # Stacktrace
    "stacktrace_anchor",
    # Slicing
    "backward_slice",
    # Score combination
    "combine_scores",
    "pick_top_nodes",
    "rank_nodes",
    # Issue boosting
    "boost_nodes_from_issue",
    "extract_method_names_from_issue",
    "extract_class_names_from_issue",
    "is_utility_method",
    # Semantic retrieval
    "semantic_retrieval_scores",
    "compute_semantic_scores",
    # Structural boost (KG-lite)
    "structural_boost_scores",
    "compute_structural_boost",
    "extract_identifiers_from_issue",
    "build_symbol_table",
    "build_call_graph",
    # Negative signals
    "negative_signal_scores",
    "compute_negative_signals",
    "NegativeSignals",
    "is_boilerplate_method",
    "is_test_or_debug_code",
    "compute_complexity_signals",
    "is_in_except_handler",
    "compute_docstring_quality",
]

