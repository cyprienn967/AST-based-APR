"""
parser.py

Produces:
    root_ast: full Python AST for the file
    metadata: ASTMetadata instance with:
        • node_id → AST node
        • parent and children relationships
        • line spans (start/end)
"""

from __future__ import annotations
import ast
from typing import Tuple

from app.ast_repair.metadata import ASTMetadata


def parse_file_to_ast(file_path: str) -> Tuple[ast.AST, ASTMetadata]:
    """
    Parse a Python file and build:
        • AST tree (root)
        • ASTMetadata with structural information

    Guarantees:
        • metadata.node_index: all nodes included
        • parent/children relationships correct
        • line_map spans populated when available
    """

    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Parse file → raw Python AST
    root = ast.parse(source, filename=file_path)

    # Metadata accumulator
    metadata = ASTMetadata()

    # DFS traversal to assign IDs and record structure
    def visit(node: ast.AST, parent_id: int | None) -> int:
        node_id = metadata.register_node(node, parent_id)

        for child in ast.iter_child_nodes(node):
            child_id = visit(child, node_id)
            metadata.add_child(node_id, child_id)

        return node_id

    # Build full tree and metadata
    visit(root, parent_id=None)

    return root, metadata
