#!/usr/bin/env python3
"""
Quick diagnostic to visualize what each localization signal produces.

Run on a single task to sanity-check that signals are firing sensibly
before training LightGBM.

Usage:
    python scripts/diagnose_signals.py <source_file> <issue_text_file>
    
Example:
    python scripts/diagnose_signals.py /path/to/buggy.py /path/to/issue.txt

Output:
    - Shows top-5 nodes per signal
    - Flags any signals that are all zeros
    - Shows signal distributions
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ast_repair.parser import parse_file_to_ast
from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.localization.sbfl_project import sbfl_project
from app.ast_repair.localization.stacktrace_anchor import stacktrace_anchor
from app.ast_repair.localization.backward_slice import backward_slice
from app.ast_repair.localization.semantic_retrieval import semantic_retrieval_scores
from app.ast_repair.localization.structural_boost import structural_boost_scores
from app.ast_repair.localization.negative_signals import negative_signal_scores, compute_negative_signals
from app.ast_repair.localization.issue_boost import extract_method_names_from_issue, extract_class_names_from_issue


def get_node_description(node: ast.AST, node_id: int, md: ASTMetadata) -> str:
    """Get a human-readable description of a node."""
    start, end = md.line_map.get(node_id, (None, None))
    line_info = f"L{start}-{end}" if start else "L?"
    
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}() [{line_info}]"
    elif isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}() [{line_info}]"
    elif isinstance(node, ast.ClassDef):
        return f"class {node.name} [{line_info}]"
    elif isinstance(node, ast.Assign):
        return f"assignment [{line_info}]"
    elif isinstance(node, ast.If):
        return f"if statement [{line_info}]"
    elif isinstance(node, ast.For):
        return f"for loop [{line_info}]"
    elif isinstance(node, ast.Return):
        return f"return [{line_info}]"
    else:
        return f"{type(node).__name__} [{line_info}]"


def print_signal_summary(
    signal_name: str,
    scores: Dict[int, float],
    md: ASTMetadata,
    top_n: int = 5
):
    """Print summary of a signal's scores."""
    print(f"\n{'='*60}")
    print(f"📊 {signal_name.upper()}")
    print(f"{'='*60}")
    
    if not scores:
        print("  ⚠️  NO SCORES (signal returned empty dict)")
        return
    
    values = list(scores.values())
    non_zero = [v for v in values if v > 0]
    
    print(f"  Nodes scored: {len(scores)}")
    print(f"  Non-zero scores: {len(non_zero)} ({len(non_zero)/len(scores)*100:.1f}%)")
    
    if non_zero:
        print(f"  Min: {min(values):.4f}")
        print(f"  Max: {max(values):.4f}")
        print(f"  Mean: {sum(values)/len(values):.4f}")
    
    # Top scoring nodes
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top {top_n} nodes:")
    for node_id, score in sorted_scores[:top_n]:
        node = md.node_index.get(node_id)
        if node:
            desc = get_node_description(node, node_id, md)
            print(f"    {score:.4f}  {desc}")
    
    # Health check
    if len(non_zero) == 0:
        print(f"\n  ❌ ISSUE: All zeros - signal not firing!")
    elif len(non_zero) < 3:
        print(f"\n  ⚠️  WARNING: Very sparse - only {len(non_zero)} non-zero values")
    else:
        print(f"\n  ✅ Signal appears healthy")


def print_negative_signals_detail(
    md: ASTMetadata,
    source_code: str,
    top_n: int = 5
):
    """Print detailed breakdown of negative signals."""
    print(f"\n{'='*60}")
    print(f"📊 NEGATIVE SIGNALS (detailed breakdown)")
    print(f"{'='*60}")
    
    all_signals = compute_negative_signals(md, source_code)
    
    if not all_signals:
        print("  ⚠️  NO SIGNALS COMPUTED")
        return
    
    # Aggregate by signal type
    signal_types = [
        "pass_only_execution", "is_boilerplate", "too_simple", "too_complex",
        "in_except_handler", "is_test_debug", "docstring_quality", "safe_patterns"
    ]
    
    for sig_type in signal_types:
        values = [getattr(s, sig_type) for s in all_signals.values()]
        non_zero = [v for v in values if v > 0]
        
        if non_zero:
            print(f"\n  {sig_type}:")
            print(f"    Non-zero: {len(non_zero)}/{len(values)}")
            print(f"    Max: {max(values):.3f}")
            
            # Show top penalized node
            top_node = max(all_signals.items(), key=lambda x: getattr(x[1], sig_type))
            node = md.node_index.get(top_node[0])
            if node:
                desc = get_node_description(node, top_node[0], md)
                print(f"    Top: {desc} ({getattr(top_node[1], sig_type):.3f})")
        else:
            print(f"\n  {sig_type}: all zeros")


def main():
    if len(sys.argv) < 3:
        print("Usage: python diagnose_signals.py <source_file> <issue_text_file>")
        print("\nAlternatively, provide issue text directly:")
        print("  python diagnose_signals.py <source_file> --issue 'description of the bug'")
        sys.exit(1)
    
    source_file = sys.argv[1]
    
    # Get issue text
    if sys.argv[2] == "--issue":
        issue_text = " ".join(sys.argv[3:])
    else:
        issue_file = sys.argv[2]
        issue_text = Path(issue_file).read_text()
    
    # Load source
    print(f"\n🔍 SIGNAL DIAGNOSTIC")
    print(f"   Source: {source_file}")
    print(f"   Issue: {issue_text[:100]}..." if len(issue_text) > 100 else f"   Issue: {issue_text}")
    
    source_code = Path(source_file).read_text()
    
    # Parse AST
    try:
        root_ast, metadata = parse_file_to_ast(source_file)
    except Exception as e:
        print(f"❌ Failed to parse file: {e}")
        sys.exit(1)
    
    print(f"\n   Parsed {len(metadata.node_index)} AST nodes")
    
    # Count function nodes (main localization targets)
    func_count = sum(1 for n in metadata.node_index.values() 
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    print(f"   Found {func_count} functions/methods")
    
    # === Issue Analysis ===
    print(f"\n{'='*60}")
    print("📝 ISSUE ANALYSIS")
    print(f"{'='*60}")
    
    methods = extract_method_names_from_issue(issue_text)
    classes = extract_class_names_from_issue(issue_text)
    
    print(f"  Extracted method names: {methods if methods else '(none)'}")
    print(f"  Extracted class names: {classes if classes else '(none)'}")
    
    # === Run Each Signal ===
    
    # 1. SBFL (will be empty without test data, that's OK)
    sbfl_line_scores: Dict[int, float] = {}  # Empty - no SBFL data in diagnostic
    sbfl_scores = sbfl_project(sbfl_line_scores, root_ast, metadata)
    print_signal_summary("SBFL (empty - no test data)", sbfl_scores, metadata)
    
    # 2. Semantic Retrieval
    try:
        semantic_scores = semantic_retrieval_scores(metadata, source_code, issue_text)
        print_signal_summary("Semantic Retrieval", semantic_scores, metadata)
    except Exception as e:
        print(f"\n⚠️  Semantic retrieval failed: {e}")
        print("   (This is OK if transformers not installed)")
    
    # 3. Structural Boost
    structural_scores = structural_boost_scores(metadata, source_code, issue_text)
    print_signal_summary("Structural Boost", structural_scores, metadata)
    
    # 4. Negative Signals
    neg_scores = negative_signal_scores(metadata, source_code)
    print_signal_summary("Negative Signals (total penalty)", neg_scores, metadata)
    
    # Detailed negative signal breakdown
    print_negative_signals_detail(metadata, source_code)
    
    # === Summary ===
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    signals_status = []
    
    if semantic_scores:
        non_zero = sum(1 for v in semantic_scores.values() if v > 0)
        status = "✅" if non_zero > 0 else "❌"
        signals_status.append(f"  {status} Semantic: {non_zero} non-zero")
    
    if structural_scores:
        non_zero = sum(1 for v in structural_scores.values() if v > 0)
        status = "✅" if non_zero > 0 else "⚠️"
        signals_status.append(f"  {status} Structural: {non_zero} non-zero")
    else:
        signals_status.append(f"  ⚠️  Structural: no matches (normal if issue doesn't mention code)")
    
    if neg_scores:
        non_zero = sum(1 for v in neg_scores.values() if v > 0)
        status = "✅" if non_zero > 0 else "⚠️"
        signals_status.append(f"  {status} Negative: {non_zero} penalized")
    
    for s in signals_status:
        print(s)
    
    print(f"\n💡 Note: SBFL requires running with actual test coverage data.")
    print(f"   Use evaluate_localization.py for full pipeline evaluation.")


if __name__ == "__main__":
    main()

