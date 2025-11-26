"""
ASTMetadata: Stores all structural information for AST-based edits and localization.

Currently included (sufficient for AST edits, diffing, subtree extraction, SBFL→AST, etc.):

- node_index: node_id → AST node object
- parent: node_id → parent_node_id
- children: node_id → list of child_node_ids
- line_map: node_id → (start_line, end_line)
- start_lineno / end_lineno: convenience views for localization modules

Later (planned):
- symbol tables
- reverse call graph
- variable def-use chains
- cross-file import resolution
- type metadata
- CFG hints
- semantic validators
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ASTMetadata:
    """
    Metadata produced by parser.py.

    node_index: node_id → AST node object
    parent: node_id → parent node_id (or None)
    children: node_id → list of child node_ids
    line_map: node_id → (start_line, end_line)
    """

    # Maps node_id → Python AST node
    node_index: Dict[int, ast.AST] = field(default_factory=dict)

    # Maps node_id → parent node_id
    parent: Dict[int, Optional[int]] = field(default_factory=dict)

    # Maps node_id → list of child node_ids
    children: Dict[int, List[int]] = field(default_factory=dict)

    # Maps node_id → (start_line, end_line) (None means not available)
    line_map: Dict[int, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)

    # Internal ID counter
    _next_id: int = 1

    # -------------------------------------------------------------
    # Node registration
    # -------------------------------------------------------------

    def new_id(self) -> int:
        """Generate a new globally unique node_id."""
        nid = self._next_id
        self._next_id += 1
        return nid

    def register_node(self, node: ast.AST, parent_id: Optional[int]) -> int:
        """
        Register a node in metadata:
            • Assign a unique node_id
            • Store node object
            • Record parent and child list
            • Record line span (lineno, end_lineno)
        """
        node_id = self.new_id()

        self.node_index[node_id] = node
        self.parent[node_id] = parent_id
        self.children[node_id] = []

        # Extract line information if present
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        if start is not None and end is not None:
            self.line_map[node_id] = (start, end)
        else:
            self.line_map[node_id] = (None, None)

        return node_id

    def add_child(self, parent_id: int, child_id: int):
        """Add a child relationship."""
        self.children[parent_id].append(child_id)

    # -------------------------------------------------------------
    # Convenience properties for localization modules
    # -------------------------------------------------------------

    @property
    def start_lineno(self) -> Dict[int, Optional[int]]:
        """Return a lightweight view: node_id → start_line."""
        return {nid: span[0] for nid, span in self.line_map.items()}

    @property
    def end_lineno(self) -> Dict[int, Optional[int]]:
        """Return a lightweight view: node_id → end_line."""
        return {nid: span[1] for nid, span in self.line_map.items()}

    # -------------------------------------------------------------
    # Helper getters
    # -------------------------------------------------------------

    def get_node_by_id(self, node_id: int) -> ast.AST:
        return self.node_index[node_id]

    def get_parent(self, node_id: int) -> Optional[int]:
        return self.parent[node_id]

    def get_children(self, node_id: int) -> List[int]:
        return self.children[node_id]

    def get_line_span(self, node_id: int) -> Tuple[Optional[int], Optional[int]]:
        return self.line_map[node_id]
