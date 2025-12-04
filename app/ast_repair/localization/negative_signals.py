"""
negative_signals.py

Negative localization signals - features that indicate a node is UNLIKELY to be buggy.

These signals are valuable for LightGBM training because the model can learn
"this feature being high means stay away from this location."

Signals implemented:
    1. pass_only_execution: Node executed by passing tests only (never by failing)
    2. is_boilerplate: Constructor, getter/setter, factory patterns
    3. too_simple: Trivial functions with no control flow
    4. too_complex: Very large functions (hard to pinpoint)
    5. in_except_handler: Inside exception handler (might mask real bug)
    6. is_test_debug_code: Test scaffolding or debug code patterns
    7. docstring_quality: Well-documented code tends to be more stable
    8. safe_code_patterns: Defensive programming patterns

Usage:
    negative_scores = compute_negative_signals(md, source_code, sbfl_data)
    # Returns: Dict[int, NegativeSignals] mapping node_id -> signal bundle
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from app.ast_repair.metadata import ASTMetadata


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class NegativeSignals:
    """
    Bundle of negative signals for a single node.
    
    All values are in [0, 1] where higher = more likely NOT the bug.
    """
    pass_only_execution: float = 0.0   # Executed by passing tests only
    is_boilerplate: float = 0.0        # Constructor, getter, etc.
    too_simple: float = 0.0            # Trivial function
    too_complex: float = 0.0           # Very large function
    in_except_handler: float = 0.0     # Inside except block
    is_test_debug: float = 0.0         # Test/debug code patterns
    docstring_quality: float = 0.0     # Well-documented = stable
    safe_patterns: float = 0.0         # Defensive coding patterns
    
    def total_penalty(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted sum of all negative signals.
        
        Default weights are tuned for SWE-bench-lite.
        """
        if weights is None:
            weights = DEFAULT_NEGATIVE_WEIGHTS
        
        return (
            weights.get("pass_only", 1.0) * self.pass_only_execution
            + weights.get("boilerplate", 0.8) * self.is_boilerplate
            + weights.get("too_simple", 0.6) * self.too_simple
            + weights.get("too_complex", 0.3) * self.too_complex
            + weights.get("except_handler", 0.5) * self.in_except_handler
            + weights.get("test_debug", 1.0) * self.is_test_debug
            + weights.get("docstring", 0.3) * self.docstring_quality
            + weights.get("safe_patterns", 0.4) * self.safe_patterns
        )


# Default weights for negative signals
DEFAULT_NEGATIVE_WEIGHTS = {
    "pass_only": 1.0,       # Strong: if only passing tests hit it, not buggy
    "boilerplate": 0.8,     # Medium-strong: __init__, getters usually safe
    "too_simple": 0.6,      # Medium: trivial code rarely buggy
    "too_complex": 0.3,     # Weak: complex code IS buggy, just hard to localize
    "except_handler": 0.5,  # Medium: except blocks mask real bugs
    "test_debug": 1.0,      # Strong: test code shouldn't be bug location
    "docstring": 0.3,       # Weak: docs correlate with stability
    "safe_patterns": 0.4,   # Medium: defensive code is safer
}


# ============================================================================
# Boilerplate / scaffolding patterns
# ============================================================================

BOILERPLATE_PATTERNS = [
    # Initialization (controversial - sometimes buggy, but usually not)
    r'^__init__$',
    r'^__new__$',
    r'^__del__$',
    r'^__copy__$',
    r'^__deepcopy__$',
    
    # Context managers
    r'^__enter__$',
    r'^__exit__$',
    
    # Simple property accessors (getters/setters)
    r'^get_[a-z_]+$',
    r'^set_[a-z_]+$',
    r'^_get_[a-z_]+$',
    r'^_set_[a-z_]+$',
    r'^has_[a-z_]+$',
    r'^is_[a-z_]+$',          # is_valid, is_empty, etc.
    
    # Factory / builder patterns
    r'^create_[a-z_]+$',
    r'^build_[a-z_]+$',
    r'^make_[a-z_]+$',
    r'^from_[a-z_]+$',        # from_dict, from_json, etc.
    r'^to_[a-z_]+$',          # to_dict, to_json (some overlap with utility)
    
    # Registration / setup
    r'^register_[a-z_]+$',
    r'^setup$',
    r'^teardown$',
    r'^configure$',
    r'^initialize$',
    
    # Deprecation wrappers
    r'^deprecated_.*$',
    r'^_deprecated_.*$',
    r'^_compat_.*$',
    
    # Mixin / interface methods (often just pass or raise NotImplementedError)
    r'^_abstract_.*$',
]

_BOILERPLATE_COMPILED = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


def is_boilerplate_method(name: str) -> float:
    """
    Check if method name matches boilerplate patterns.
    
    Returns score in [0, 1].
    """
    for pattern in _BOILERPLATE_COMPILED:
        if pattern.match(name):
            return 0.8  # High confidence boilerplate
    
    # Partial matches for common patterns
    name_lower = name.lower()
    
    if name_lower.startswith('_') and name_lower.endswith('_helper'):
        return 0.5
    
    if 'callback' in name_lower or 'handler' in name_lower:
        return 0.3  # Event handlers are sometimes buggy
    
    return 0.0


# ============================================================================
# Test / debug code detection
# ============================================================================

TEST_DEBUG_NAME_PATTERNS = [
    r'^test_',
    r'_test$',
    r'^_test_',
    r'^debug_',
    r'_debug$',
    r'^_debug_',
    r'^dump_',
    r'^log_',
    r'^trace_',
    r'^mock_',
    r'^fake_',
    r'^stub_',
    r'^example_',
    r'^demo_',
    r'^sample_',
    r'^benchmark_',
]

_TEST_DEBUG_COMPILED = [re.compile(p, re.IGNORECASE) for p in TEST_DEBUG_NAME_PATTERNS]


def is_test_or_debug_code(
    node: ast.AST,
    source_lines: List[str]
) -> float:
    """
    Detect test scaffolding or debug code patterns.
    
    Returns score in [0, 1].
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return 0.0
    
    score = 0.0
    name = node.name if hasattr(node, 'name') else ""
    
    # Check name patterns
    for pattern in _TEST_DEBUG_COMPILED:
        if pattern.match(name):
            score += 0.6
            break
    
    # Check for debug statements in body (for functions)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        start_line = getattr(node, 'lineno', 0)
        end_line = getattr(node, 'end_lineno', start_line)
        
        for line_num in range(start_line, min(end_line + 1, len(source_lines) + 1)):
            if line_num <= 0 or line_num > len(source_lines):
                continue
            line = source_lines[line_num - 1]
            
            # Debug statements
            if 'print(' in line and 'sprint' not in line.lower():
                score += 0.1
            if 'pdb.' in line or 'breakpoint()' in line or 'ipdb' in line:
                score += 0.3
            if 'logging.debug' in line:
                score += 0.05
            
            # Debug comments
            if '# DEBUG' in line or '# TODO' in line or '# FIXME' in line:
                score += 0.1
            if '# HACK' in line or '# XXX' in line:
                score += 0.1
    
    return min(score, 1.0)


# ============================================================================
# Code complexity analysis
# ============================================================================

def compute_complexity_signals(node: ast.AST) -> Tuple[float, float]:
    """
    Compute too_simple and too_complex scores.
    
    Returns (too_simple, too_complex) both in [0, 1].
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (0.0, 0.0)
    
    # Calculate metrics
    start_line = getattr(node, 'lineno', 0)
    end_line = getattr(node, 'end_lineno', start_line)
    line_count = max(1, end_line - start_line + 1)
    
    # Count control flow structures
    control_flow_count = 0
    statement_count = 0
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                              ast.Match, ast.ExceptHandler)):
            control_flow_count += 1
        if isinstance(child, ast.stmt):
            statement_count += 1
    
    # --- Too Simple ---
    too_simple = 0.0
    
    # Single return statement
    if statement_count <= 2 and control_flow_count == 0:
        too_simple = 0.9
    # Very short with no control flow
    elif line_count <= 3 and control_flow_count == 0:
        too_simple = 0.7
    elif line_count <= 5 and control_flow_count == 0:
        too_simple = 0.4
    elif line_count <= 7 and control_flow_count <= 1:
        too_simple = 0.2
    
    # Check if it's just a pass or raise NotImplementedError
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = node.body
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                too_simple = 1.0
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Just a docstring
                too_simple = 0.95
            elif isinstance(stmt, ast.Raise):
                # raise NotImplementedError
                too_simple = 0.9
    
    # --- Too Complex ---
    too_complex = 0.0
    
    # Very long functions are hard to localize precisely
    if line_count > 200:
        too_complex = 0.7
    elif line_count > 100:
        too_complex = 0.5
    elif line_count > 50:
        too_complex = 0.3
    elif line_count > 30:
        too_complex = 0.1
    
    # High cyclomatic complexity
    if control_flow_count > 20:
        too_complex = max(too_complex, 0.5)
    elif control_flow_count > 10:
        too_complex = max(too_complex, 0.3)
    
    return (too_simple, too_complex)


# ============================================================================
# Exception handler detection
# ============================================================================

def is_in_except_handler(node_id: int, md: ASTMetadata) -> float:
    """
    Check if node is inside an exception handler.
    
    Exception handlers often mask the real bug location - the bug is
    usually where the exception was raised, not where it's caught.
    
    Returns 1.0 if inside except block, 0.0 otherwise.
    """
    current = node_id
    
    while current is not None:
        node = md.node_index.get(current)
        if node is None:
            break
        
        if isinstance(node, ast.ExceptHandler):
            return 1.0
        
        # Also check for finally blocks (less suspicious but still)
        # We detect this by checking if we're in a Try node's finalbody
        parent_id = md.parent.get(current)
        if parent_id is not None:
            parent_node = md.node_index.get(parent_id)
            if isinstance(parent_node, ast.Try):
                # Check if current node is in finalbody
                for final_stmt in parent_node.finalbody:
                    if id(final_stmt) == id(node):
                        return 0.5  # Finally blocks are less suspicious
        
        current = md.parent.get(current)
    
    return 0.0


# ============================================================================
# Docstring quality analysis
# ============================================================================

def compute_docstring_quality(node: ast.AST) -> float:
    """
    Compute documentation quality score.
    
    Well-documented code tends to be more mature and stable.
    Higher score = better documented = LESS likely to be buggy.
    
    Returns score in [0, 1].
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return 0.0
    
    docstring = ast.get_docstring(node)
    
    if not docstring:
        return 0.0  # No docs = neutral
    
    score = 0.0
    
    # Base score for having a docstring
    score += 0.1
    
    # Length-based scoring
    doc_len = len(docstring)
    if doc_len > 500:
        score += 0.2
    elif doc_len > 200:
        score += 0.15
    elif doc_len > 100:
        score += 0.1
    elif doc_len > 50:
        score += 0.05
    
    # Has parameter documentation
    param_patterns = [
        r':param\s+\w+:',           # Sphinx style
        r'Args:\s*\n',              # Google style
        r'Parameters\s*\n',         # NumPy style
        r'@param\s+',               # JSDoc style
    ]
    for pattern in param_patterns:
        if re.search(pattern, docstring):
            score += 0.15
            break
    
    # Has return documentation
    return_patterns = [
        r':returns?:',
        r'Returns:\s*\n',
        r'@returns?\s+',
    ]
    for pattern in return_patterns:
        if re.search(pattern, docstring):
            score += 0.1
            break
    
    # Has type hints in docs
    if re.search(r':type\s+\w+:', docstring) or re.search(r'\(\w+\)', docstring):
        score += 0.1
    
    # Has examples (strong signal of maturity)
    if '>>>' in docstring:  # Doctest
        score += 0.2
    elif re.search(r'Example[s]?:', docstring, re.IGNORECASE):
        score += 0.15
    
    # Has raises documentation
    if re.search(r':raises?\s+\w+:', docstring) or re.search(r'Raises:\s*\n', docstring):
        score += 0.1
    
    return min(score, 1.0)


# ============================================================================
# Safe / defensive code patterns
# ============================================================================

SAFE_CODE_PATTERNS = [
    # Type checking before operations
    (r'if\s+isinstance\s*\(', 0.15),
    (r'if\s+type\s*\(', 0.1),
    
    # None checks
    (r'if\s+\w+\s+is\s+None', 0.1),
    (r'if\s+\w+\s+is\s+not\s+None', 0.1),
    (r'if\s+not\s+\w+:', 0.05),
    
    # Bounds checking
    (r'if\s+\w+\s*[<>]=?\s*\d+', 0.1),
    (r'if\s+len\s*\(', 0.1),
    
    # Assertions (developer awareness)
    (r'^\s*assert\s+', 0.15),
    
    # Try/except for specific errors
    (r'except\s+\w+Error', 0.1),
    
    # Default values
    (r'\w+\s*=\s*\w+\s+or\s+', 0.05),  # x = val or default
    (r'\.get\s*\(\s*[^,]+\s*,', 0.1),  # dict.get(key, default)
    
    # Explicit type conversion
    (r'int\s*\(', 0.05),
    (r'str\s*\(', 0.05),
    (r'float\s*\(', 0.05),
    (r'bool\s*\(', 0.05),
]


def compute_safe_patterns_score(
    node: ast.AST,
    source_lines: List[str]
) -> float:
    """
    Detect defensive programming patterns.
    
    Code with many safety checks is less likely to be buggy.
    Returns score in [0, 1].
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0.0
    
    start_line = getattr(node, 'lineno', 0)
    end_line = getattr(node, 'end_lineno', start_line)
    
    if start_line <= 0 or end_line <= 0:
        return 0.0
    
    score = 0.0
    line_count = end_line - start_line + 1
    
    for line_num in range(start_line, min(end_line + 1, len(source_lines) + 1)):
        if line_num <= 0 or line_num > len(source_lines):
            continue
        line = source_lines[line_num - 1]
        
        for pattern, weight in SAFE_CODE_PATTERNS:
            if re.search(pattern, line):
                score += weight
    
    # Normalize by function size (larger functions naturally have more patterns)
    if line_count > 10:
        score = score / (line_count / 10)
    
    return min(score, 1.0)


# ============================================================================
# SBFL pass-only execution signal
# ============================================================================

def compute_pass_only_signal(
    node_id: int,
    md: ASTMetadata,
    sbfl_line_stats: Optional[Dict[int, Tuple[int, int]]] = None
) -> float:
    """
    Check if node is executed only by passing tests.
    
    Args:
        node_id: The AST node to check
        md: AST metadata
        sbfl_line_stats: Dict mapping line_num -> (pass_count, fail_count)
    
    Returns:
        1.0 if executed by passing tests only (strong negative)
        0.5 if never executed
        0.0 if executed by at least one failing test
    """
    if sbfl_line_stats is None:
        return 0.0  # No SBFL data available
    
    start_line, end_line = md.line_map.get(node_id, (None, None))
    if start_line is None or end_line is None:
        return 0.0
    
    total_pass = 0
    total_fail = 0
    lines_covered = 0
    
    for line_num in range(start_line, end_line + 1):
        if line_num in sbfl_line_stats:
            pass_count, fail_count = sbfl_line_stats[line_num]
            total_pass += pass_count
            total_fail += fail_count
            lines_covered += 1
    
    if lines_covered == 0:
        return 0.5  # Never executed - mildly suspicious (could be dead code)
    
    if total_fail == 0:
        if total_pass > 0:
            return 1.0  # Only passing tests - strong negative signal
        return 0.5  # No coverage data
    
    return 0.0  # At least one failing test hit this code


# ============================================================================
# Main API: Compute all negative signals for all nodes
# ============================================================================

def compute_negative_signals(
    md: ASTMetadata,
    source_code: str,
    sbfl_line_stats: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, NegativeSignals]:
    """
    Compute all negative signals for all function/method nodes.
    
    Args:
        md: AST metadata with node_index
        source_code: Source code of the file
        sbfl_line_stats: Optional SBFL data {line -> (pass_count, fail_count)}
    
    Returns:
        Dict mapping node_id -> NegativeSignals bundle
    """
    source_lines = source_code.split('\n') if source_code else []
    results: Dict[int, NegativeSignals] = {}
    
    for node_id, node in md.node_index.items():
        # Only compute for function-like nodes (main localization targets)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        
        signals = NegativeSignals()
        
        # 1. Pass-only execution
        signals.pass_only_execution = compute_pass_only_signal(node_id, md, sbfl_line_stats)
        
        # 2. Boilerplate detection
        if hasattr(node, 'name'):
            signals.is_boilerplate = is_boilerplate_method(node.name)
        
        # 3. Complexity analysis
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signals.too_simple, signals.too_complex = compute_complexity_signals(node)
        
        # 4. Exception handler
        signals.in_except_handler = is_in_except_handler(node_id, md)
        
        # 5. Test/debug code
        signals.is_test_debug = is_test_or_debug_code(node, source_lines)
        
        # 6. Docstring quality
        signals.docstring_quality = compute_docstring_quality(node)
        
        # 7. Safe patterns
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signals.safe_patterns = compute_safe_patterns_score(node, source_lines)
        
        results[node_id] = signals
    
    if results:
        # Log summary
        high_penalty_count = sum(1 for s in results.values() if s.total_penalty() > 0.5)
        logger.debug(f"Computed negative signals for {len(results)} nodes, "
                    f"{high_penalty_count} have penalty > 0.5")
    
    return results


def compute_total_negative_score(
    md: ASTMetadata,
    source_code: str,
    sbfl_line_stats: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, float]:
    """
    Convenience function returning just the total penalty score per node.
    
    This is what gets integrated into the main scoring pipeline.
    
    Returns:
        Dict mapping node_id -> total_penalty (higher = less likely buggy)
    """
    all_signals = compute_negative_signals(md, source_code, sbfl_line_stats)
    return {nid: signals.total_penalty() for nid, signals in all_signals.items()}


# ============================================================================
# Wrapper for localize.py integration
# ============================================================================

def negative_signal_scores(
    md: ASTMetadata,
    source_code: str,
    sbfl_line_stats: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Dict[int, float]:
    """
    Simplified wrapper for use in localize_fault().
    
    Returns node_id -> penalty score (SUBTRACT this from total score).
    """
    try:
        return compute_total_negative_score(md, source_code, sbfl_line_stats)
    except Exception as e:
        logger.warning(f"Negative signal computation failed: {e}")
        return {}

