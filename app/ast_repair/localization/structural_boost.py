"""
structural_boost.py

Lightweight structural boosting for bug localization (KG-lite).

This module provides a repository-level structural signal without requiring
a full knowledge graph. It identifies functions/methods that are structurally
adjacent to identifiers mentioned in the issue text.

Architecture:
    1. Identifier Extraction: Parse the issue text for function names, class names,
       error types, module paths, and other code identifiers.
    
    2. Symbol Table: Scan the AST to build a mapping from identifier names to
       node_ids (function/class definitions).
    
    3. Call Graph (Coarse): Build a lightweight call graph by analyzing ast.Call
       nodes to identify caller-callee relationships within the file.
    
    4. Structural Scoring:
       - Direct matches (identifier in issue → matching definition): HIGH boost
       - 1-hop neighbors (functions that call/are called by matches): MEDIUM boost
       - Class membership (functions in a matched class): MEDIUM boost

Key Insight (from KGCompass paper):
    Most bugs occur in functions that are NOT directly mentioned in the issue,
    but are structurally adjacent to mentioned functions. This module captures
    that signal cheaply.

Usage:
    structural_scores = compute_structural_boost(md, source_code, issue_text)
    # Returns: Dict[int, float] mapping node_id -> structural score
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger

from app.ast_repair.metadata import ASTMetadata


# ============================================================================
# Configuration
# ============================================================================

# Boost weights for different structural relationships
DIRECT_MATCH_BOOST = 3.0      # Identifier directly mentioned in issue
CALLER_BOOST = 1.5            # Function that calls a matched function
CALLEE_BOOST = 1.5            # Function called by a matched function
CLASS_MEMBER_BOOST = 1.2      # Function inside a matched class
PARTIAL_MATCH_BOOST = 1.5     # Partial name match (substring)

# Maximum hops for structural propagation (keep at 1 for precision)
MAX_HOPS = 1


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SymbolTable:
    """
    Maps identifier names to their AST node_ids.
    """
    # function_name -> list of node_ids (can have multiple with same name)
    functions: Dict[str, List[int]] = field(default_factory=dict)
    
    # class_name -> node_id
    classes: Dict[str, int] = field(default_factory=dict)
    
    # class_name -> list of method node_ids
    class_methods: Dict[str, List[int]] = field(default_factory=dict)
    
    # node_id -> function/class name (reverse lookup)
    node_to_name: Dict[int, str] = field(default_factory=dict)


@dataclass
class CallGraph:
    """
    Lightweight call graph built from AST analysis.
    """
    # caller_node_id -> set of callee names (we use names because callees might be external)
    calls: Dict[int, Set[str]] = field(default_factory=dict)
    
    # callee_name -> set of caller_node_ids (reverse lookup)
    called_by: Dict[str, Set[int]] = field(default_factory=dict)


# ============================================================================
# Identifier extraction from issue text
# ============================================================================

def extract_identifiers_from_issue(issue_text: str) -> Dict[str, Set[str]]:
    """
    Extract various types of identifiers from issue text.
    
    Returns:
        Dict with keys: 'functions', 'classes', 'errors', 'modules', 'all'
    """
    if not issue_text:
        return {'functions': set(), 'classes': set(), 'errors': set(), 'modules': set(), 'all': set()}
    
    identifiers: Dict[str, Set[str]] = {
        'functions': set(),
        'classes': set(),
        'errors': set(),
        'modules': set(),
        'all': set(),
    }
    
    # --- Function/method names ---
    # Pattern: snake_case identifiers that look like function names
    # Match: method_name, _private_method, __dunder__, func123
    func_patterns = [
        r'`([a-z_][a-z0-9_]*)`',                          # `func_name`
        r'\.([a-z_][a-z0-9_]*)\s*\(',                     # .method_name(
        r'(?:def|function|method)\s+([a-z_][a-z0-9_]*)',  # def func_name
        r'(?:call(?:ing)?|invoke|run)\s+([a-z_][a-z0-9_]*)', # calling func_name
        r'(?:in|at|from)\s+([a-z_][a-z0-9_]*)\s*(?:\(|$|\s)', # in func_name
        r'\b(_[a-z][a-z0-9_]*)\b',                        # _private_method
        r'\b(__[a-z][a-z0-9_]*__)\b',                     # __dunder__
    ]
    
    for pattern in func_patterns:
        for match in re.finditer(pattern, issue_text, re.IGNORECASE):
            name = match.group(1)
            if len(name) >= 3 and not _is_common_word(name):
                identifiers['functions'].add(name)
    
    # --- Class names (PascalCase) ---
    class_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z][a-zA-Z0-9]*)+)\b'
    for match in re.finditer(class_pattern, issue_text):
        name = match.group(1)
        if not _is_common_class_word(name):
            identifiers['classes'].add(name)
    
    # Also match simple capitalized words in specific contexts
    class_context_patterns = [
        r'class\s+([A-Z][a-zA-Z0-9]+)',           # class ClassName
        r'(?:the|a|an)\s+([A-Z][a-zA-Z0-9]+)\s+(?:class|object|instance)',
        r'`([A-Z][a-zA-Z0-9]+)`',                 # `ClassName`
        r'([A-Z][a-zA-Z0-9]+)\s*\(',              # ClassName(
        r'isinstance\s*\([^,]+,\s*([A-Z][a-zA-Z0-9]+)\)', # isinstance(x, Class)
    ]
    
    for pattern in class_context_patterns:
        for match in re.finditer(pattern, issue_text):
            name = match.group(1)
            if len(name) >= 2 and not _is_common_class_word(name):
                identifiers['classes'].add(name)
    
    # --- Error types ---
    error_pattern = r'\b([A-Z][a-zA-Z]*(?:Error|Exception|Warning|Fault))\b'
    for match in re.finditer(error_pattern, issue_text):
        identifiers['errors'].add(match.group(1))
    
    # --- Module paths ---
    # Match: module.submodule.function or module.Class
    module_pattern = r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)\b'
    for match in re.finditer(module_pattern, issue_text):
        path = match.group(1)
        identifiers['modules'].add(path)
        # Also extract the last component as potential function/class
        parts = path.split('.')
        last = parts[-1]
        if last[0].isupper():
            identifiers['classes'].add(last)
        else:
            identifiers['functions'].add(last)
    
    # --- Traceback function names ---
    # Match: File "...", line N, in function_name
    traceback_pattern = r'File "[^"]+", line \d+, in ([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(traceback_pattern, issue_text):
        name = match.group(1)
        if name not in ('<module>', '<lambda>'):
            identifiers['functions'].add(name)
    
    # Build 'all' set
    identifiers['all'] = (
        identifiers['functions'] | 
        identifiers['classes'] | 
        identifiers['errors']
    )
    
    if identifiers['all']:
        logger.debug(f"Extracted identifiers: functions={identifiers['functions']}, "
                    f"classes={identifiers['classes']}, errors={identifiers['errors']}")
    
    return identifiers


def _is_common_word(word: str) -> bool:
    """Check if word is a common English word (not a code identifier)."""
    common = {
        'the', 'this', 'that', 'with', 'from', 'have', 'has', 'had',
        'should', 'would', 'could', 'when', 'where', 'which', 'their',
        'there', 'about', 'error', 'issue', 'problem', 'bug', 'fix',
        'function', 'method', 'class', 'module', 'file', 'code', 'line',
        'return', 'value', 'need', 'want', 'expect', 'expected', 'get',
        'set', 'add', 'del', 'new', 'old', 'result', 'output', 'input',
        'test', 'true', 'false', 'none', 'self', 'args', 'kwargs',
        'data', 'item', 'items', 'list', 'dict', 'str', 'int', 'for',
        'and', 'not', 'but', 'use', 'used', 'using', 'can', 'also',
    }
    return word.lower() in common


def _is_common_class_word(word: str) -> bool:
    """Check if word is a common capitalized word (not a class name)."""
    common = {
        'The', 'This', 'That', 'When', 'Where', 'Which', 'What', 'How',
        'If', 'Then', 'Error', 'Issue', 'Problem', 'Bug', 'Fix', 'None',
        'True', 'False', 'Type', 'Note', 'Warning', 'TODO', 'FIXME',
        'Example', 'See', 'Also', 'File', 'Line', 'Code', 'Output',
    }
    return word in common


# ============================================================================
# Symbol table construction
# ============================================================================

def build_symbol_table(md: ASTMetadata) -> SymbolTable:
    """
    Build a symbol table mapping names to node_ids from AST metadata.
    """
    symbols = SymbolTable()
    current_class: Optional[str] = None
    class_node_stack: List[Tuple[str, int]] = []  # (class_name, end_line)
    
    # First pass: collect all definitions
    for node_id, node in md.node_index.items():
        # Track class context
        start_line, end_line = md.line_map.get(node_id, (None, None))
        
        # Pop classes we've exited
        while class_node_stack and end_line and class_node_stack[-1][1] < start_line:
            class_node_stack.pop()
        
        current_class = class_node_stack[-1][0] if class_node_stack else None
        
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            symbols.classes[class_name] = node_id
            symbols.node_to_name[node_id] = class_name
            symbols.class_methods[class_name] = []
            
            # Push class onto stack
            if end_line:
                class_node_stack.append((class_name, end_line))
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            
            # Add to functions dict
            if func_name not in symbols.functions:
                symbols.functions[func_name] = []
            symbols.functions[func_name].append(node_id)
            symbols.node_to_name[node_id] = func_name
            
            # Track class membership
            if current_class and current_class in symbols.class_methods:
                symbols.class_methods[current_class].append(node_id)
    
    logger.debug(f"Built symbol table: {len(symbols.functions)} functions, "
                f"{len(symbols.classes)} classes")
    
    return symbols


# ============================================================================
# Call graph construction
# ============================================================================

def build_call_graph(md: ASTMetadata) -> CallGraph:
    """
    Build a lightweight call graph by analyzing ast.Call nodes.
    
    This identifies which functions call which other functions (by name).
    Note: This is imprecise due to Python's dynamic nature, but good enough
    for localization boosting.
    """
    call_graph = CallGraph()
    
    # Find the enclosing function for each node
    def find_enclosing_function(node_id: int) -> Optional[int]:
        """Walk up parent chain to find enclosing function."""
        current = md.parent.get(node_id)
        while current is not None:
            node = md.node_index.get(current)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = md.parent.get(current)
        return None
    
    # Scan all nodes for Call expressions
    for node_id, node in md.node_index.items():
        if isinstance(node, ast.Call):
            # Find the enclosing function
            caller_id = find_enclosing_function(node_id)
            if caller_id is None:
                continue
            
            # Initialize caller's call set
            if caller_id not in call_graph.calls:
                call_graph.calls[caller_id] = set()
            
            # Extract callee name
            callee_name = _extract_callee_name(node)
            if callee_name:
                call_graph.calls[caller_id].add(callee_name)
                
                # Update reverse mapping
                if callee_name not in call_graph.called_by:
                    call_graph.called_by[callee_name] = set()
                call_graph.called_by[callee_name].add(caller_id)
    
    logger.debug(f"Built call graph: {len(call_graph.calls)} callers, "
                f"{len(call_graph.called_by)} unique callees")
    
    return call_graph


def _extract_callee_name(call_node: ast.Call) -> Optional[str]:
    """
    Extract the callee name from a Call node.
    
    Handles:
        - func()           -> "func"
        - obj.method()     -> "method"
        - module.func()    -> "func"
        - Class()          -> "Class"
    """
    func = call_node.func
    
    if isinstance(func, ast.Name):
        # Simple function call: func()
        return func.id
    
    elif isinstance(func, ast.Attribute):
        # Method/attribute call: obj.method()
        return func.attr
    
    # Other cases (subscript calls, etc.) - skip
    return None


# ============================================================================
# Structural scoring
# ============================================================================

def compute_structural_boost(
    md: ASTMetadata,
    source_code: str,
    issue_text: str,
) -> Dict[int, float]:
    """
    Compute structural boost scores based on identifier matching and call graph.
    
    This is the main entry point for structural boosting in the localization pipeline.
    
    Args:
        md: AST metadata with node_index
        source_code: Source code of the file (unused, for API consistency)
        issue_text: The bug report / issue description
    
    Returns:
        Dict mapping node_id -> structural boost score
        Empty dict if no matches found.
    """
    if not issue_text:
        return {}
    
    # Extract identifiers from issue
    identifiers = extract_identifiers_from_issue(issue_text)
    if not identifiers['all']:
        logger.debug("No identifiers extracted from issue for structural boost")
        return {}
    
    # Build symbol table and call graph
    symbols = build_symbol_table(md)
    call_graph = build_call_graph(md)
    
    # Compute scores
    scores: Dict[int, float] = {}
    matched_functions: Set[int] = set()
    matched_classes: Set[str] = set()
    
    # --- Direct function matches ---
    for func_name in identifiers['functions']:
        # Exact match
        if func_name in symbols.functions:
            for node_id in symbols.functions[func_name]:
                scores[node_id] = max(scores.get(node_id, 0), DIRECT_MATCH_BOOST)
                matched_functions.add(node_id)
                logger.debug(f"Direct function match: {func_name} -> node {node_id}")
        
        # Partial match (substring)
        for sym_name, node_ids in symbols.functions.items():
            if func_name != sym_name and (func_name in sym_name or sym_name in func_name):
                for node_id in node_ids:
                    if node_id not in matched_functions:
                        scores[node_id] = max(scores.get(node_id, 0), PARTIAL_MATCH_BOOST)
                        logger.debug(f"Partial function match: {func_name} ~ {sym_name}")
    
    # --- Direct class matches ---
    for class_name in identifiers['classes']:
        if class_name in symbols.classes:
            class_node_id = symbols.classes[class_name]
            scores[class_node_id] = max(scores.get(class_node_id, 0), DIRECT_MATCH_BOOST)
            matched_classes.add(class_name)
            logger.debug(f"Direct class match: {class_name} -> node {class_node_id}")
    
    # --- Boost methods of matched classes ---
    for class_name in matched_classes:
        if class_name in symbols.class_methods:
            for method_id in symbols.class_methods[class_name]:
                if method_id not in matched_functions:
                    scores[method_id] = max(scores.get(method_id, 0), CLASS_MEMBER_BOOST)
                    logger.debug(f"Class member boost: {class_name}.{symbols.node_to_name.get(method_id)}")
    
    # --- 1-hop call graph neighbors ---
    for func_id in list(matched_functions):
        func_name = symbols.node_to_name.get(func_id)
        if not func_name:
            continue
        
        # Boost callers of matched function
        if func_name in call_graph.called_by:
            for caller_id in call_graph.called_by[func_name]:
                if caller_id not in matched_functions:
                    scores[caller_id] = max(scores.get(caller_id, 0), CALLER_BOOST)
                    caller_name = symbols.node_to_name.get(caller_id, f"node_{caller_id}")
                    logger.debug(f"Caller boost: {caller_name} calls {func_name}")
        
        # Boost callees of matched function
        if func_id in call_graph.calls:
            for callee_name in call_graph.calls[func_id]:
                if callee_name in symbols.functions:
                    for callee_id in symbols.functions[callee_name]:
                        if callee_id not in matched_functions:
                            scores[callee_id] = max(scores.get(callee_id, 0), CALLEE_BOOST)
                            logger.debug(f"Callee boost: {func_name} calls {callee_name}")
    
    # --- Error type handling ---
    # If error types are mentioned, boost exception handlers and raise statements
    if identifiers['errors']:
        for node_id, node in md.node_index.items():
            if isinstance(node, ast.Raise):
                # Check if this raises a matched error type
                if node.exc and isinstance(node.exc, ast.Call):
                    error_name = _extract_callee_name(node.exc)
                    if error_name in identifiers['errors']:
                        scores[node_id] = max(scores.get(node_id, 0), DIRECT_MATCH_BOOST)
                        logger.debug(f"Error raise boost: raises {error_name}")
            
            elif isinstance(node, ast.ExceptHandler):
                # Check if this catches a matched error type
                if node.type:
                    if isinstance(node.type, ast.Name) and node.type.id in identifiers['errors']:
                        scores[node_id] = max(scores.get(node_id, 0), PARTIAL_MATCH_BOOST)
                        logger.debug(f"Error handler boost: catches {node.type.id}")
    
    if scores:
        logger.info(f"Structural boost: {len(scores)} nodes boosted, "
                   f"{len(matched_functions)} direct matches")
    
    return scores


# ============================================================================
# Convenience wrapper for localize.py integration
# ============================================================================

def structural_boost_scores(
    md: ASTMetadata,
    source_code: str,
    issue_text: str,
) -> Dict[int, float]:
    """
    Simplified wrapper for use in localize_fault().
    
    Returns node_id -> structural score mapping, or empty dict on error.
    """
    try:
        return compute_structural_boost(md, source_code, issue_text)
    except Exception as e:
        logger.warning(f"Structural boost failed: {e}")
        return {}


# ============================================================================
# Extended identifier extraction for better coverage
# ============================================================================

def extract_identifiers_extended(
    issue_text: str,
    include_numbers: bool = False
) -> Set[str]:
    """
    Extended identifier extraction with more aggressive matching.
    
    Use this for cases where standard extraction misses identifiers.
    """
    identifiers = set()
    
    # All code-like tokens in backticks
    for match in re.finditer(r'`([^`]+)`', issue_text):
        token = match.group(1).strip()
        # Split on dots and collect parts
        parts = token.split('.')
        for part in parts:
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                identifiers.add(part)
    
    # All snake_case identifiers
    for match in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', issue_text):
        identifiers.add(match.group(1))
    
    # All PascalCase identifiers
    for match in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', issue_text):
        identifiers.add(match.group(1))
    
    # Filter out common words
    identifiers = {i for i in identifiers if not _is_common_word(i.lower())}
    
    return identifiers

