#todo
"""
apply_edits.py

Given:
    - root_ast:  the Python AST for a file (from parser.parse_file_to_ast)
    - metadata:  an ASTMetadata instance (node_id → node, parent/children, lines)
    - edits:     a list of ASTEdit operations (from edit_schema.py)

This module applies a sequence of *op-based* AST edits of the form:

    {
      "op": "replace_expr" | "replace_stmt" | "insert_before" | "insert_after" | "delete",
      "target": {"node_id": <int>},
      "new_code": "<small Python code snippet>"  # only for some ops
    }

Design (v0):
------------
• We use ASTMetadata only for:
    - mapping node_id → AST node
    - mapping node_id → parent_id

• We DO NOT update metadata incrementally as we mutate the tree.
  For v0, metadata is only required to locate targets. After all edits,
  callers should re-run parser.py to rebuild fresh metadata if they need it.

• We:
    - Parse new_code into AST nodes (expr or list[stmt] depending on op)
    - Locate the parent container of the target node via metadata + AST
    - Modify the tree in-place (replace, insert, delete)

Supported ops (v0 semantics):
-----------------------------
    "replace_expr":
        - target node is used in expression position
        - new_code must be a valid Python expression
        - we replace the target node with the new expression node

    "replace_stmt":
        - target node is a statement in some list field (e.g., body)
        - new_code must be one or more statements
        - we replace the target statement with the new statements

    "insert_before":
        - target node is a statement in a list field
        - new_code must be one or more statements
        - we insert new statements immediately before the target

    "insert_after":
        - same as insert_before but we insert after the target

    "delete":
        - target node is a statement in a list field
        - we remove it from the list

Limitations (v0):
-----------------
    - For "replace_expr", we search for the target node in:
        • any attribute field of the parent that is an AST node
        • any list field of the parent containing AST nodes
      If we cannot find a matching field, we raise an error.

    - For statement-based ops, we expect the target node to appear
      inside some list of AST nodes in the parent (e.g., body, orelse).
      If not found, we raise an error.

    - We do not currently support:
        • editing multiple disjoint occurrences of the same node
        • deep, semantic transformations (that belongs to higher layers)

If any edit fails (bad op, incompatible context, parse error, etc.),
we raise ASTEditApplicationError so the caller can decide whether to
fallback or discard the entire patch.
"""

from __future__ import annotations

import ast
from typing import List, Tuple, Optional

from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.edit_schema import ASTEdit, SUPPORTED_OPS


class ASTEditApplicationError(Exception):
    """Raised when an AST edit cannot be applied."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_edits(root_ast: ast.AST, metadata: ASTMetadata, edits: List[ASTEdit]) -> ast.AST:
    """
    Apply a sequence of ASTEdit operations to root_ast in-place.

    Parameters:
        root_ast: ast.AST
            The root of the Python AST (Module node from parser.py).

        metadata: ASTMetadata
            Metadata produced by parser.py. Used only to:
                • resolve node_id → node
                • resolve parent_id → parent node

        edits: List[ASTEdit]
            List of structured edits conforming to edit_schema.py.

    Returns:
        The mutated root_ast (same object, modified in-place).

    Notes:
        - For v0, metadata is NOT updated after edits.
          If you need fresh metadata, re-run parser.parse_file_to_ast
          on the serialized source.

        - Edits are applied in the order provided. If later edits refer
          to node_ids that have been deleted or replaced, behavior is
          undefined and may raise ASTEditApplicationError.
    """
    for edit in edits:
        _apply_single_edit(root_ast, metadata, edit)

    return root_ast


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_single_edit(root_ast: ast.AST, metadata: ASTMetadata, edit: ASTEdit) -> None:
    """Apply a single ASTEdit to the AST in-place."""
    if edit.op not in SUPPORTED_OPS:
        raise ASTEditApplicationError(f"Unsupported op '{edit.op}'")

    try:
        target_node = metadata.get_node_by_id(edit.target_node_id)
    except KeyError as e:
        raise ASTEditApplicationError(
            f"Unknown target_node_id {edit.target_node_id}"
        ) from e

    parent_id = metadata.get_parent(edit.target_node_id)
    parent_node: Optional[ast.AST] = (
        metadata.get_node_by_id(parent_id) if parent_id is not None else None
    )

    # Dispatch by op
    if edit.op == "replace_expr":
        if edit.new_code is None:
            raise ASTEditApplicationError("replace_expr requires new_code")
        new_expr = _parse_expr(edit.new_code)
        _replace_expr(parent_node, target_node, new_expr)

    elif edit.op == "replace_stmt":
        if edit.new_code is None:
            raise ASTEditApplicationError("replace_stmt requires new_code")
        new_stmts = _parse_statements(edit.new_code)
        _replace_stmt(parent_node, target_node, new_stmts)

    elif edit.op == "insert_before":
        if edit.new_code is None:
            raise ASTEditApplicationError("insert_before requires new_code")
        new_stmts = _parse_statements(edit.new_code)
        _insert_relative(parent_node, target_node, new_stmts, before=True)

    elif edit.op == "insert_after":
        if edit.new_code is None:
            raise ASTEditApplicationError("insert_after requires new_code")
        new_stmts = _parse_statements(edit.new_code)
        _insert_relative(parent_node, target_node, new_stmts, before=False)

    elif edit.op == "delete":
        _delete_node(parent_node, target_node)

    else:
        # Should be unreachable due to SUPPORTED_OPS check.
        raise ASTEditApplicationError(f"Unhandled op '{edit.op}'")


# ---------------------------------------------------------------------------
# Parsing new_code into AST
# ---------------------------------------------------------------------------

def _parse_expr(code: str) -> ast.expr:
    """Parse code as a Python expression and return the AST node."""
    try:
        expr_ast = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise ASTEditApplicationError(f"Failed to parse new_code as expr: {e}") from e

    if not isinstance(expr_ast.body, ast.expr):
        raise ASTEditApplicationError("Parsed expression is not an ast.expr node")

    return expr_ast.body


def _parse_statements(code: str) -> List[ast.stmt]:
    """
    Parse code as one or more Python statements and return the list of stmt nodes.
    """
    try:
        mod = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ASTEditApplicationError(f"Failed to parse new_code as stmt(s): {e}") from e

    # mod.body is List[ast.stmt]
    return list(mod.body)


# ---------------------------------------------------------------------------
# Tree manipulation helpers
# ---------------------------------------------------------------------------

def _replace_expr(parent: Optional[ast.AST], target: ast.AST, new_expr: ast.expr) -> None:
    """
    Replace an expression node under parent with new_expr.

    We search through all fields of parent:
        - if a field value is exactly the target node, replace it with new_expr
        - if a field is a list, replace any entries equal to target with new_expr

    If parent is None or no matching field is found, we error.
    """
    if parent is None:
        raise ASTEditApplicationError("replace_expr target has no parent (cannot replace root)")

    replaced = False

    for field_name, value in ast.iter_fields(parent):
        # Single child
        if value is target:
            setattr(parent, field_name, new_expr)
            replaced = True

        # List of children
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if item is target:
                    new_list.append(new_expr)
                    replaced = True
                else:
                    new_list.append(item)
            if replaced:
                setattr(parent, field_name, new_list)

    if not replaced:
        raise ASTEditApplicationError(
            "replace_expr: could not find target node in any parent field"
        )


def _replace_stmt(parent: Optional[ast.AST], target: ast.AST, new_stmts: List[ast.stmt]) -> None:
    """
    Replace a statement node under parent with one or more new statements.

    We look for any list-valued field of parent that contains the target
    node, and replace that single element with the contents of new_stmts.
    """
    if parent is None:
        raise ASTEditApplicationError("replace_stmt target has no parent (cannot replace root)")

    replaced = False

    for field_name, value in ast.iter_fields(parent):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            # Find index of target
            for idx, item in enumerate(value):
                if item is target:
                    # Replace target with new_stmts
                    new_list = value[:idx] + new_stmts + value[idx + 1 :]
                    setattr(parent, field_name, new_list)
                    replaced = True
                    break

        if replaced:
            break

    if not replaced:
        raise ASTEditApplicationError(
            "replace_stmt: could not find target stmt in any parent stmt list"
        )


def _insert_relative(
    parent: Optional[ast.AST],
    target: ast.AST,
    new_stmts: List[ast.stmt],
    before: bool,
) -> None:
    """
    Insert new_stmts before or after a target statement in a parent's stmt list.

    before=True  → insert before target
    before=False → insert after target
    """
    if parent is None:
        raise ASTEditApplicationError(
            "insert_before/after target has no parent (cannot insert at root)"
        )

    inserted = False

    for field_name, value in ast.iter_fields(parent):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            for idx, item in enumerate(value):
                if item is target:
                    if before:
                        new_list = value[:idx] + new_stmts + value[idx:]
                    else:
                        new_list = value[:idx + 1] + new_stmts + value[idx + 1:]
                    setattr(parent, field_name, new_list)
                    inserted = True
                    break

        if inserted:
            break

    if not inserted:
        raise ASTEditApplicationError(
            "insert_before/after: could not find target stmt in any parent stmt list"
        )


def _delete_node(parent: Optional[ast.AST], target: ast.AST) -> None:
    """
    Delete a statement node from its parent's stmt list.

    We search all list-valued fields of parent that contain stmt nodes,
    and remove the target node if found.
    """
    if parent is None:
        raise ASTEditApplicationError("delete target has no parent (cannot delete root)")

    deleted = False

    for field_name, value in ast.iter_fields(parent):
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            new_list = [item for item in value if item is not target]
            if len(new_list) != len(value):
                setattr(parent, field_name, new_list)
                deleted = True
                break

    if not deleted:
        raise ASTEditApplicationError(
            "delete: could not find target stmt in any parent stmt list"
        )
