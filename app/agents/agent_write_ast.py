# agent_write_ast.py

import json
from dataclasses import dataclass
from typing import List

from app.model.common import SELECTED_MODEL
from app.ast_repair.edit_schema import (
    schema_description,
    parse_edits_from_json_str,
    EditSchemaError,
    ASTEdit,
)


SYSTEM_PROMPT = """You are AutoCodeRover-AST, an agent that produces AST edit actions
instead of code patches. You MUST output JSON conforming exactly to the AST edit schema.

If no change is needed, output [] and nothing else.
"""


@dataclass
class ASTGenerationResult:
    edits: List[ASTEdit]
    prompt: str
    raw_response: str | None


def _format_prompt(
    issue: str,
    file_path: str,
    snippet: str,
    ast_dump: str,
    intended_behavior: str,
) -> str:
    return f"""
<issue>
{issue}
</issue>

<file>
{file_path}
</file>

<buggy_code>
{snippet}
</buggy_code>

<intended_behavior>
{intended_behavior}
</intended_behavior>

<ast>
{ast_dump}
</ast>

### STRICT EDIT SCHEMA ###
{schema_description()}

Respond ONLY with JSON. Return a single edit object or [].
"""


def _extract_json_region(s: str) -> str:
    """
    Extracts the JSON array/object region from a messy LLM output.
    """
    first_brace = s.find("{")
    first_bracket = s.find("[")

    candidates = [x for x in [first_brace, first_bracket] if x != -1]
    if not candidates:
        raise ValueError("No JSON object or array start found.")

    start = min(candidates)

    last_brace = s.rfind("}")
    last_bracket = s.rfind("]")

    end = max(last_brace, last_bracket)
    if end == -1:
        raise ValueError("No JSON end found.")

    return s[start : end + 1]


def generate_ast_edits(
    issue_context: str,
    file_path: str,
    code_snippet: str,
    intended_behavior: str,
    ast_dump: str,
    allow_multiple: bool = False,
) -> ASTGenerationResult:
    """
    Core LLM call: returns List[ASTEdit].

    Enforcement:
      - Only valid JSON survives
      - Only schema-valid edits survive
      - Optional enforcement of one-edit-only (SWE-bench lite)
    """

    user_prompt = _format_prompt(
        issue_context,
        file_path,
        code_snippet,
        ast_dump,
        intended_behavior,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content = None

    try:
        content, _, _, _ = SELECTED_MODEL.call(
            messages=messages,
            response_format="json_object",
        )
    except Exception:
        return ASTGenerationResult([], user_prompt, None)

    if not content:
        return ASTGenerationResult([], user_prompt, None)

    content = content.strip()

    try:
        json_text = _extract_json_region(content)
    except Exception:
        return ASTGenerationResult([], user_prompt, content)

    try:
        edits = parse_edits_from_json_str(json_text)
    except EditSchemaError:
        return ASTGenerationResult([], user_prompt, content)

    if not allow_multiple and len(edits) > 1:
        return ASTGenerationResult([], user_prompt, content)

    return ASTGenerationResult(edits, user_prompt, content)
