# serialize.py
"""
serialize.py

This module converts a mutated Python AST (after apply_edits) back into
source code using Python's built-in `ast.unparse`.

Responsibilities:
    • Accept a root `ast.AST` object
    • Return a fully formatted Python source code string
    • Optionally perform post-processing cleanup

Usage:
    from app.ast_repair.serialize import ast_to_source
    new_source = ast_to_source(root_ast)

Notes:
    - In Python 3.9+, `ast.unparse` is stable and preserves correct syntax.
    - We can add formatting normalization later if desired.
"""

from __future__ import annotations
import ast


def ast_to_source(root_ast: ast.AST) -> str:
    """
    Convert an AST back to source code using `ast.unparse`.

    Parameters:
        root_ast: ast.AST

    Returns:
        A Python source code string.
    """
    try:
        code = ast.unparse(root_ast)
    except Exception as e:
        raise RuntimeError(f"Failed to unparse AST: {e}") from e

    # Ensure code ends with a newline
    if not code.endswith("\n"):
        code += "\n"

    return code

