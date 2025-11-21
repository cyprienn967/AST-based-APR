"""
CURRENT METADATA INCLUDED:

for now we only store info needed for:
- Basic AST navigation
- Mapping node_id → AST node
- Mapping parent/children relationships
- Mapping node → (start_line, end_line)
- Supporting subtree extraction
- Supporting apply_edits.py
- Supporting serialization and diff generation

for later (when we do validation + our own localization):
- Full symbol table (function defs, variable defs, class defs)
- Reverse call graph (function_def → call sites)
- Variable use-def chains
- Cross-file import resolution
- Control-flow graph (CFG) hints
- Type and annotation metadata
- Semantic impact propagation for validators
- AST-based precise fault localization logic

"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class ASTMetadata:
    """
node_index: node_id → AST node object
parent: node_id → parent_node_id
children: node_id → list of child_node_ids
line_map: node_id → (start_line, end_line)
this metadata is gonna be produced by parser.py
    """

    # Maps node_id → Python AST node
    node_index: Dict[int, ast.AST] = field(default_factory=dict)

    # Maps node_id → parent_node_id
    parent: Dict[int, Optional[int]] = field(default_factory=dict)

    # Maps node_id → list of child_node_ids
    children: Dict[int, List[int]] = field(default_factory=dict)

    # Maps node_id → (start_line, end_line) (None allowed when not available)
    line_map: Dict[int, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)

    # Internal counter for assigning unique node IDs
    _next_id: int = 1

    def new_id(self) -> int:
        """Generate a new globally unique node_id."""
        nid = self._next_id
        self._next_id += 1
        return nid

    def register_node(self, node: ast.AST, parent_id: Optional[int]) -> int:
        """
        Assign a new node_id to this AST node, register:
            • node_index
            • parent
            • initialize children list
            • line_map (if available)

        Returns the assigned node_id.
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
        """Add a child relationship to metadata."""
        self.children[parent_id].append(child_id)

    # -------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------

    def get_node_by_id(self, node_id: int) -> ast.AST:
        return self.node_index[node_id]

    def get_parent(self, node_id: int) -> Optional[int]:
        return self.parent[node_id]

    def get_children(self, node_id: int) -> List[int]:
        return self.children[node_id]

    def get_line_span(self, node_id: int) -> Tuple[Optional[int], Optional[int]]:
        return self.line_map[node_id]

   