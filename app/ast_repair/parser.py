"""
parser.py produces root_ast (the actual AST) and then metadata:
root_ast: the full Python AST for the file
metadata: an ASTMetadata instance populated with:
    node_id → AST node
    parent/children relationships
    line spans

how to call parser.py:
    from app.ast_repair.parser import parse_file_to_ast
    ast_root, metadata = parse_file_to_ast(filepath)
"""

from __future__ import annotations
import ast
from typing import Tuple

from app.ast_repair.metadata import ASTMetadata


def parse_file_to_ast(file_path: str) -> Tuple[ast.AST, ASTMetadata]:
    """
    build AST + metadat then return both (as a tuple)

    This function guarantees:
        • root_ast is a valid Python ast.AST
        • metadata.node_index contains all nodes
        • metadata.parent/children reflect the true tree
        • line_map contains start/end line information where possible
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse file → raw Python AST
    root = ast.parse(source, filename=file_path)

    # Metadata structure
    metadata = ASTMetadata()

    # DFS traversal to assign IDs and record structure
    def visit(node: ast.AST, parent_id: int | None):
        # Assign unique ID
        node_id = metadata.register_node(node, parent_id)

        # Visit children
        for child in ast.iter_child_nodes(node):
            child_id = visit(child, node_id)
            metadata.add_child(node_id, child_id)

        return node_id

    # Build full tree and metadata
    visit(root, parent_id=None)

    return root, metadata