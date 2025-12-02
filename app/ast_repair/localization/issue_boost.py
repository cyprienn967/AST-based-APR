"""
issue_boost.py

Extract method/function names from issue text and boost their scores.

This addresses the key limitation where SBFL might point to utility methods
(like __str__, _sympystr) but the issue explicitly mentions the actual buggy
method (like _eval_expand_tensorproduct).

Strategy:
    1. Parse method names mentioned in issue text
    2. Boost nodes whose function/method name matches issue keywords
    3. Penalize known utility patterns (string conversion, repr, etc.)
"""

from __future__ import annotations
import ast
import re
from typing import Dict, Set, Optional
from loguru import logger

from app.ast_repair.metadata import ASTMetadata


# ============================================================================
# Utility method patterns to penalize
# ============================================================================

UTILITY_METHOD_PATTERNS = [
    r'^__str__$',
    r'^__repr__$',
    r'^__unicode__$',
    r'^__format__$',
    r'^__hash__$',
    r'^__eq__$',
    r'^__ne__$',
    r'^__lt__$',
    r'^__le__$',
    r'^__gt__$',
    r'^__ge__$',
    r'^_print.*',           # Sympy printing methods
    r'^_sympystr$',
    r'^_sympyrepr$',
    r'^_latex$',
    r'^_pretty$',
    r'^.*_str$',            # Generic string methods
    r'^to_string$',
    r'^format.*',
    r'^print_.*',
    r'^display.*',
    r'^render.*',
    r'^serialize$',
    r'^deserialize$',
    r'^__getattr__$',
    r'^__setattr__$',
    r'^__delattr__$',
    r'^__getitem__$',
    r'^__setitem__$',
    r'^__delitem__$',
    r'^__len__$',
    r'^__iter__$',
    r'^__next__$',
    r'^__contains__$',
    r'^__bool__$',
    r'^__int__$',
    r'^__float__$',
    r'^__complex__$',
    r'^__bytes__$',
]

# Compile patterns for efficiency
_UTILITY_PATTERNS_COMPILED = [re.compile(p) for p in UTILITY_METHOD_PATTERNS]


# ============================================================================
# Extract method names from issue text
# ============================================================================

def extract_method_names_from_issue(issue_text: str) -> Set[str]:
    """
    Extract method/function names mentioned in issue text.
    
    Patterns matched:
        - "method foo_bar"
        - "function calculate_score"
        - "def process_data"
        - `method_name()` (markdown code)
        - obj.method_name()
        - _private_method
        - _eval_expand_tensorproduct (underscore-prefixed methods)
    
    Returns:
        Set of method names (without parentheses)
    """
    if not issue_text:
        return set()
    
    method_names: Set[str] = set()
    
    # Pattern 1: Explicit mentions - "method foo", "function bar", "def baz"
    explicit_patterns = [
        r'(?:method|func(?:tion)?|def)\s+([a-z_][a-z0-9_]*)',
        r'(?:the|a|this)\s+([a-z_][a-z0-9_]*)\s+(?:method|function)',
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, issue_text, re.IGNORECASE):
            method_names.add(match.group(1))
    
    # Pattern 2: Markdown inline code - `method_name()` or `method_name`
    for match in re.finditer(r'`([a-z_][a-z0-9_]*)\s*\(?`', issue_text):
        method_names.add(match.group(1))
    
    # Pattern 3: Method calls - obj.method_name() or .method_name(
    for match in re.finditer(r'\.([a-z_][a-z0-9_]*)\s*\(', issue_text):
        method_names.add(match.group(1))
    
    # Pattern 4: Underscore-prefixed methods (often private/internal)
    # These are commonly mentioned in bug reports
    for match in re.finditer(r'\b(_[a-z][a-z0-9_]*)\b', issue_text):
        name = match.group(1)
        # Filter out dunder methods from this pattern
        if not name.startswith('__'):
            method_names.add(name)
    
    # Pattern 5: Error messages often mention method names
    # "in _eval_expand_tensorproduct", "at calculate_result"
    for match in re.finditer(r'(?:in|at|from)\s+([a-z_][a-z0-9_]*)', issue_text):
        method_names.add(match.group(1))
    
    # Filter out common words that aren't method names
    common_words = {
        'the', 'this', 'that', 'with', 'from', 'have', 'should', 'would',
        'could', 'when', 'where', 'which', 'their', 'there', 'about', 'error',
        'issue', 'problem', 'function', 'method', 'class', 'module', 'file',
        'code', 'return', 'value', 'need', 'want', 'expect', 'expected',
        'actual', 'result', 'output', 'input', 'test', 'true', 'false',
        'none', 'self', 'args', 'kwargs', 'data', 'item', 'items', 'list',
        'dict', 'string', 'integer', 'float', 'bool', 'type', 'name',
    }
    method_names = {m for m in method_names if m.lower() not in common_words}
    
    # Filter out very short names (likely false positives)
    method_names = {m for m in method_names if len(m) >= 3}
    
    if method_names:
        logger.debug(f"Extracted method names from issue: {method_names}")
    
    return method_names


def extract_class_names_from_issue(issue_text: str) -> Set[str]:
    """
    Extract class names (PascalCase) from issue text.
    """
    if not issue_text:
        return set()
    
    class_names: Set[str] = set()
    
    # PascalCase identifiers (at least 2 chars, starts with uppercase)
    for match in re.finditer(r'\b([A-Z][a-zA-Z0-9]{2,})\b', issue_text):
        name = match.group(1)
        # Filter out common words that happen to be capitalized
        if name not in {'The', 'This', 'That', 'When', 'Where', 'Which', 'Error', 
                        'Issue', 'Problem', 'None', 'True', 'False', 'Type'}:
            class_names.add(name)
    
    return class_names


# ============================================================================
# Check if a method name is a utility method
# ============================================================================

def is_utility_method(method_name: str) -> bool:
    """
    Check if a method name matches known utility patterns.
    
    Utility methods (like __str__, _print_*, etc.) are less likely to be
    the root cause of bugs - they're usually just formatting/display code.
    """
    for pattern in _UTILITY_PATTERNS_COMPILED:
        if pattern.match(method_name):
            return True
    return False


# ============================================================================
# Main boosting function
# ============================================================================

def compute_issue_boost(
    scores: Dict[int, float],
    md: ASTMetadata,
    issue_text: str,
    method_boost: float = 5.0,
    class_boost: float = 2.0,
    utility_penalty: float = 0.1,
) -> Dict[int, float]:
    """
    Boost scores for nodes matching issue-mentioned methods/classes.
    Penalize utility methods that are unlikely to be buggy.
    
    Args:
        scores: Current node_id → score mapping
        md: AST metadata with node_index
        issue_text: The issue/problem statement
        method_boost: Multiplier for exact method name matches
        class_boost: Multiplier for nodes inside matching classes
        utility_penalty: Multiplier for utility methods (< 1.0 = penalty)
    
    Returns:
        Updated node_id → score mapping
    """
    if not issue_text or not scores:
        return scores
    
    # Extract keywords from issue
    method_keywords = extract_method_names_from_issue(issue_text)
    class_keywords = extract_class_names_from_issue(issue_text)
    
    if not method_keywords and not class_keywords:
        logger.debug("No method/class keywords extracted from issue")
        return scores
    
    logger.info(f"Issue keywords - methods: {method_keywords}, classes: {class_keywords}")
    
    # Build a map of class name → node_ids inside that class
    class_to_nodes: Dict[str, Set[int]] = {}
    for nid, node in md.node_index.items():
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            # Collect all descendant node_ids
            descendants = _get_descendant_ids(md, nid)
            class_to_nodes[class_name] = descendants
    
    # Apply boosts and penalties
    boosted_scores = scores.copy()
    
    for nid, score in scores.items():
        node = md.node_index.get(nid)
        if node is None:
            continue
        
        boost_factor = 1.0
        reasons = []
        
        # Check if this is a function/method definition
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            
            # Strong boost if exact match with issue-mentioned method
            if func_name in method_keywords:
                boost_factor *= method_boost
                reasons.append(f"method '{func_name}' mentioned in issue")
            
            # Partial match boost (e.g., "expand" matches "_eval_expand_tensorproduct")
            else:
                for keyword in method_keywords:
                    if keyword in func_name or func_name in keyword:
                        boost_factor *= (method_boost / 2)  # Half boost for partial
                        reasons.append(f"partial match: '{keyword}' ~ '{func_name}'")
                        break
            
            # Utility penalty
            if is_utility_method(func_name):
                boost_factor *= utility_penalty
                reasons.append(f"utility method '{func_name}'")
        
        # Check if node is inside a matching class
        for class_name, descendant_ids in class_to_nodes.items():
            if class_name in class_keywords and nid in descendant_ids:
                boost_factor *= class_boost
                reasons.append(f"inside class '{class_name}' mentioned in issue")
                break
        
        if boost_factor != 1.0:
            boosted_scores[nid] = score * boost_factor
            if reasons:
                logger.debug(f"Node {nid}: {score:.3f} → {boosted_scores[nid]:.3f} ({', '.join(reasons)})")
    
    return boosted_scores


def _get_descendant_ids(md: ASTMetadata, node_id: int) -> Set[int]:
    """Get all descendant node_ids of a given node."""
    descendants = set()
    
    # Use the children map if available
    if hasattr(md, 'children') and md.children:
        stack = list(md.children.get(node_id, []))
        while stack:
            child_id = stack.pop()
            descendants.add(child_id)
            stack.extend(md.children.get(child_id, []))
    else:
        # Fallback: use parent map to find children
        for nid, parent_id in md.parent.items():
            if parent_id == node_id or nid in descendants:
                descendants.add(nid)
                # Need multiple passes for nested nodes
        # Multiple passes to catch all descendants
        changed = True
        while changed:
            changed = False
            for nid, parent_id in md.parent.items():
                if parent_id in descendants and nid not in descendants:
                    descendants.add(nid)
                    changed = True
    
    return descendants


# ============================================================================
# Convenience function for use in localize.py
# ============================================================================

def boost_nodes_from_issue(
    scores: Dict[int, float],
    md: ASTMetadata,
    issue_text: str,
) -> Dict[int, float]:
    """
    Convenience wrapper for compute_issue_boost with default parameters.
    
    Use this in localize_fault() after combining SBFL/trace/slice scores.
    """
    return compute_issue_boost(
        scores=scores,
        md=md,
        issue_text=issue_text,
        method_boost=5.0,      # 5x for exact method match
        class_boost=2.0,       # 2x for nodes in mentioned class
        utility_penalty=0.1,   # 0.1x for utility methods
    )

