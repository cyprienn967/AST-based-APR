"""
edit_schema.py

Op-based AST edit JSON with small code snippets.

This module defines:
    - The *schema* for AST edit operations that the LLM must produce.
    - A small dataclass `ASTEdit` representing a single edit.
    - Utilities to parse and validate LLM output into `ASTEdit` objects.
    - A human-readable schema description string you can embed in prompts.

Core design (v0):
-----------------
We use an op-based edit format where each edit is a JSON object of the form:

    {
      "op": "replace_expr",
      "target": {"node_id": 184},
      "new_code": "return None if value is None else value"
    }

Key points:

    • "op" encodes the operation type and expected context:
        - "replace_expr"   : replace an expression node
        - "replace_stmt"   : replace a statement node (or small stmt block)
        - "insert_before"  : insert statement(s) before a target statement node
        - "insert_after"   : insert statement(s) after a target statement node
        - "delete"         : delete the target node (usually a statement)

    • "target" is an object with:
        - "node_id": integer ID assigned by parser.py / ASTMetadata

    • "new_code" is a small Python code snippet:
        - required for ops that introduce or replace code
          ("replace_expr", "replace_stmt", "insert_before", "insert_after")
        - forbidden / ignored for "delete"

    • We rely on the AST layer to:
        - parse `new_code` into AST (expression or statements)
        - enforce syntactic invariants
        - apply the edit structurally using `node_id` and metadata

This gives:
    - Strong structural guarantees via AST-based application.
    - Short, human-readable edit descriptions for the LLM.
    - Minimal token bloat vs full JSON AST.
    - A clear separation between WHERE we edit (op + target.node_id)
      and WHAT code is inserted (new_code snippet).

For now, we intentionally keep the op set small (5 ops) because:
    • SWE-bench-lite patches are usually small (3–4 lines).
    • These ops cover the majority of realistic local repairs.
    • We can add more ops later (e.g., wrap_in_try_except, rename_var, etc.)

Top-level LLM output format:
----------------------------
We accept a few flexible shapes, all normalized to a list of edits:

    1) Single edit object:
        {
          "op": "...",
          "target": {"node_id": 123},
          "new_code": "..."
        }

    2) List of edit objects:
        [
          {...},
          {...}
        ]

    3) Object with "edits" key:
        {
          "edits": [
            {...},
            {...}
          ]
        }

        

TO NOTE: 


IF LIST OF EDITS (I.E MULTIPLE) FOR LITE <=> WRONG EDIT (propagation check type)


In all cases, we parse and validate into `List[ASTEdit]`.

Future (Phase 2) extensions:
----------------------------
    • Add more specialized ops:
        - "wrap_in_try_except"
        - "rename_identifier"
        - "change_call_arguments"
        - etc.
    • Attach additional metadata (e.g., "kind": "expr"/"stmt"/"block")
      if needed for finer-grained validation.
    • Add semantic flags for validators (e.g., "may_change_control_flow": true).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal
import json


# -----------------------------
# Supported operations (v0)
# -----------------------------

# We keep this as a simple list/set so it's easy to extend later.
SUPPORTED_OPS = {
    "replace_expr",   # replace an expression node
    "replace_stmt",   # replace a statement node / small block
    "insert_before",  # insert stmt(s) before target stmt
    "insert_after",   # insert stmt(s) after target stmt
    "delete",         # delete target node
}


class EditSchemaError(Exception):
    """Raised when LLM output does not conform to the expected edit schema."""


@dataclass
class ASTEdit:
    """
    Internal representation of a single AST edit.

    Fields:
        op: str
            One of SUPPORTED_OPS.

        target_node_id: int
            The node_id as assigned in ASTMetadata by parser.py.

        new_code: Optional[str]
            Small Python code snippet to be parsed and inserted/applied.
            Required for ops that add/replace code:
                - replace_expr
                - replace_stmt
                - insert_before
                - insert_after
            Forbidden/ignored for:
                - delete
    """

    op: str
    target_node_id: int
    new_code: Optional[str] = None


# ----------------------------------------------------------------------
# Schema description for LLM prompting
# ----------------------------------------------------------------------

def schema_description() -> str:
    """
    Return a human-readable description of the edit schema, suitable
    for including in LLM prompts.

    Person B can call this and embed the string directly in the system / tool
    prompt so the model knows exactly how to format its output.
    """
    return (
        "You must output JSON describing a list of AST edit operations. "
        "Each edit has the form:\n\n"
        "  {\n"
        "    \"op\": \"<operation>\",\n"
        "    \"target\": {\"node_id\": <int>},\n"
        "    \"new_code\": \"<small Python snippet>\"  // only for some ops\n"
        "  }\n\n"
        "Supported ops:\n"
        "  - \"replace_expr\": replace an expression node with a new expression.\n"
        "  - \"replace_stmt\": replace a statement node (or very small block).\n"
        "  - \"insert_before\": insert one or more statements before the target statement.\n"
        "  - \"insert_after\": insert one or more statements after the target statement.\n"
        "  - \"delete\": delete the target node (usually a statement).\n\n"
        "Rules:\n"
        "  - \"target.node_id\" must be an integer referencing an existing AST node.\n"
        "  - \"new_code\" is REQUIRED for: replace_expr, replace_stmt, insert_before, insert_after.\n"
        "  - \"new_code\" is FORBIDDEN (or will be ignored) for: delete.\n"
        "  - \"new_code\" must be valid Python code for the intended context (expr or stmt).\n"
        "  - Output either:\n"
        "        {\"edits\": [ { ... }, { ... } ]}\n"
        "    or  [ { ... }, { ... } ]\n"
        "    or  a single object { ... } for one edit.\n"
    )


# ----------------------------------------------------------------------
# Parsing / validation helpers
# ----------------------------------------------------------------------

def _normalize_top_level(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize various allowed top-level JSON shapes from the LLM into
    a list of raw edit dicts.

    Accepted forms:
        1) Single edit dict:
            { ... }

        2) List of edit dicts:
            [ { ... }, { ... } ]

        3) Object with "edits" key:
            { "edits": [ { ... }, { ... } ] }
    """
    if isinstance(obj, dict):
        if "edits" in obj and isinstance(obj["edits"], list):
            return obj["edits"]
        # treat a bare dict as a single edit
        return [obj]
    elif isinstance(obj, list):
        return obj
    else:
        raise EditSchemaError(
            f"Top-level JSON must be an object or list; got {type(obj).__name__}"
        )


def _parse_single_edit(raw: Dict[str, Any]) -> ASTEdit:
    """
    Parse and validate a single raw edit dict into an ASTEdit.
    Raises EditSchemaError if fields are missing or invalid.
    """
    # op
    op = raw.get("op")
    if not isinstance(op, str):
        raise EditSchemaError("Each edit must have a string field 'op'.")
    if op not in SUPPORTED_OPS:
        raise EditSchemaError(f"Unsupported op '{op}'. Supported: {sorted(SUPPORTED_OPS)}")

    # target
    target = raw.get("target")
    if not isinstance(target, dict):
        raise EditSchemaError("Each edit must have a 'target' object.")
    node_id = target.get("node_id")
    if not isinstance(node_id, int):
        raise EditSchemaError("'target.node_id' must be an integer.")

    # new_code
    new_code = raw.get("new_code", None)

    if op == "delete":
        # new_code should not be used
        if new_code not in (None, ""):
            # We don't need to be super strict: we can ignore extra new_code,
            # but it's good to log or at least be aware.
            # Here we just ignore it.
            new_code = None
    else:
        # new_code is required
        if not isinstance(new_code, str) or not new_code.strip():
            raise EditSchemaError(
                f"Edit with op '{op}' requires a non-empty 'new_code' string."
            )

    return ASTEdit(op=op, target_node_id=node_id, new_code=new_code)


def parse_edits_from_json_str(raw_json: str) -> List[ASTEdit]:
    """
    Parse a JSON string returned by the LLM into a list of ASTEdit objects.

    This:
        • Parses JSON
        • Normalizes top-level structure (single/edit, list, or {\"edits\": [...]})
        • Validates each edit against the schema
        • Returns a list[ASTEdit]

    Raises EditSchemaError if:
        • JSON is invalid
        • Required fields are missing or wrong type
        • Ops are unsupported
    """
    try:
        obj = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise EditSchemaError(f"Invalid JSON from LLM: {e}") from e

    raw_edits = _normalize_top_level(obj)
    edits: List[ASTEdit] = []
    for idx, raw_edit in enumerate(raw_edits):
        if not isinstance(raw_edit, dict):
            raise EditSchemaError(
                f"Edit at index {idx} must be an object; got {type(raw_edit).__name__}"
            )
        edits.append(_parse_single_edit(raw_edit))

    return edits
