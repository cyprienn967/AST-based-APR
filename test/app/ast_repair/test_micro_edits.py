"""
test_micro_edits.py

Unit tests for the micro-edit fast path routing.
"""

import ast
import pytest
from pathlib import Path

from app.ast_repair.micro_edits import (
    is_micro_editable,
    generate_micro_edits,
    MicroEdit,
    TestOutcome,
)
from app.ast_repair.parser import parse_file_to_ast
from app.ast_repair.metadata import ASTMetadata


def test_is_micro_editable():
    """Test that micro-editable nodes are correctly identified."""
    # Test Compare node (==, !=, etc.)
    compare_node = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.Eq()],
        comparators=[ast.Constant(value=5)]
    )
    assert is_micro_editable(compare_node)
    
    # Test Return node
    return_node = ast.Return(value=ast.Constant(value=True))
    assert is_micro_editable(return_node)
    
    # Test UnaryOp node
    unary_node = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x"))
    assert is_micro_editable(unary_node)
    
    # Test If node
    if_node = ast.If(
        test=ast.Name(id="condition"),
        body=[],
        orelse=[]
    )
    assert is_micro_editable(if_node)
    
    # Test non-micro-editable node (e.g., ClassDef)
    class_node = ast.ClassDef(
        name="MyClass",
        bases=[],
        keywords=[],
        body=[],
        decorator_list=[]
    )
    assert not is_micro_editable(class_node)


def test_generate_micro_edits_compare():
    """Test micro-edit generation for comparison operators."""
    # Create a metadata object
    metadata = ASTMetadata()
    
    # Test == to != transformation
    compare_eq = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.Eq()],
        comparators=[ast.Constant(value=5)]
    )
    node_id = metadata.register_node(compare_eq, None)
    
    edits = generate_micro_edits(compare_eq, node_id, metadata)
    assert len(edits) > 0
    assert any("==" in edit.description and "!=" in edit.description for edit in edits)
    
    # Test != to == transformation
    compare_neq = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.NotEq()],
        comparators=[ast.Constant(value=5)]
    )
    node_id2 = metadata.register_node(compare_neq, None)
    
    edits = generate_micro_edits(compare_neq, node_id2, metadata)
    assert len(edits) > 0
    assert any("!=" in edit.description and "==" in edit.description for edit in edits)


def test_generate_micro_edits_return():
    """Test micro-edit generation for return statements."""
    metadata = ASTMetadata()
    
    # Test boolean return
    return_bool = ast.Return(value=ast.Constant(value=True))
    node_id = metadata.register_node(return_bool, None)
    
    edits = generate_micro_edits(return_bool, node_id, metadata)
    assert len(edits) > 0
    assert any("True" in edit.description and "False" in edit.description for edit in edits)
    
    # Test integer return
    return_int = ast.Return(value=ast.Constant(value=5))
    node_id2 = metadata.register_node(return_int, None)
    
    edits = generate_micro_edits(return_int, node_id2, metadata)
    assert len(edits) > 0
    # Should suggest 6, 4, 0, 1
    assert len(edits) <= 3  # Max 3 edits per node


def test_generate_micro_edits_unary():
    """Test micro-edit generation for unary operators."""
    metadata = ASTMetadata()
    
    # Test 'not' removal
    unary_not = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x"))
    node_id = metadata.register_node(unary_not, None)
    
    edits = generate_micro_edits(unary_not, node_id, metadata)
    assert len(edits) > 0
    assert any("not" in edit.description.lower() for edit in edits)


def test_generate_micro_edits_if():
    """Test micro-edit generation for if statements."""
    metadata = ASTMetadata()
    
    # Test condition negation
    if_stmt = ast.If(
        test=ast.Name(id="condition"),
        body=[ast.Pass()],
        orelse=[]
    )
    node_id = metadata.register_node(if_stmt, None)
    
    edits = generate_micro_edits(if_stmt, node_id, metadata)
    assert len(edits) > 0
    assert any("negate" in edit.description.lower() for edit in edits)


def test_micro_edit_transform():
    """Test that micro-edit transformations produce valid AST nodes."""
    metadata = ASTMetadata()
    
    # Create a comparison node
    compare = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.Eq()],
        comparators=[ast.Constant(value=5)]
    )
    node_id = metadata.register_node(compare, None)
    
    edits = generate_micro_edits(compare, node_id, metadata)
    assert len(edits) > 0
    
    # Apply the transformation
    first_edit = edits[0]
    transformed = first_edit.transform(compare)
    
    # Verify it's still a Compare node
    assert isinstance(transformed, ast.Compare)
    # Verify the operator changed
    assert not isinstance(transformed.ops[0], ast.Eq)


def test_max_edits_per_node():
    """Test that we don't generate more than 3 edits per node."""
    metadata = ASTMetadata()
    
    # Create a node that could generate many edits
    return_int = ast.Return(value=ast.Constant(value=5))
    node_id = metadata.register_node(return_int, None)
    
    edits = generate_micro_edits(return_int, node_id, metadata)
    assert len(edits) <= 3


def test_get_ranked_node_ids():
    """Test extraction of ranked node IDs from BugLocation objects."""
    from app.ast_repair.localize import BugLocation, get_ranked_node_ids
    
    # Create mock BugLocation objects
    bug_locs = [
        BugLocation(node_id=10, subtree=ast.Pass(), expansion_chain=[10, 20]),
        BugLocation(node_id=15, subtree=ast.Pass(), expansion_chain=[15, 25]),
        BugLocation(node_id=20, subtree=ast.Pass(), expansion_chain=[20, 30]),
    ]
    
    node_ids = get_ranked_node_ids(bug_locs)
    assert node_ids == [10, 15, 20]


def test_replace_node_in_ast():
    """Test that replace_node_in_ast correctly replaces nodes."""
    from app.ast_repair.apply_edits import replace_node_in_ast
    
    # Create a simple AST: x == 5
    original_compare = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.Eq()],
        comparators=[ast.Constant(value=5)]
    )
    
    # Wrap in a module
    module = ast.Module(body=[ast.Expr(value=original_compare)], type_ignores=[])
    
    # Create metadata
    metadata = ASTMetadata()
    module_id = metadata.register_node(module, None)
    expr_id = metadata.register_node(module.body[0], module_id)
    metadata.add_child(module_id, expr_id)
    compare_id = metadata.register_node(original_compare, expr_id)
    metadata.add_child(expr_id, compare_id)
    
    # Create a new comparison: x != 5
    new_compare = ast.Compare(
        left=ast.Name(id="x"),
        ops=[ast.NotEq()],
        comparators=[ast.Constant(value=5)]
    )
    
    # Replace the node
    replace_node_in_ast(module, compare_id, new_compare, metadata)
    
    # Verify the replacement
    replaced_compare = module.body[0].value
    assert isinstance(replaced_compare, ast.Compare)
    assert isinstance(replaced_compare.ops[0], ast.NotEq)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

