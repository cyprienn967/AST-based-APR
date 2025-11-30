# agent_write_ast.py

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger

from app.agents.agent_common import extract_json_from_response
from app.model import common as model_common
from app.ast_repair.edit_schema import (
    schema_description,
    parse_edits_from_json_str,
    EditSchemaError,
    ASTEdit,
)
from app.ast_repair.serialize import ast_to_source
from app.ast_repair.localize import localize_fault, annotate_code_with_node_ids
from app.ast_repair.parser import parse_file_to_ast


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
    annotated_code: str,
    intended_behavior: str,
    target_node_id: int,
    test_failure_output: str = "",
) -> str:
    """
    Format the prompt for the LLM using annotated code instead of AST dump.
    
    This reduces token count by ~40x (from 20K to 500 tokens) by showing
    node_ids inline with the code rather than as a massive AST dump.
    
    Args:
        issue: The issue description
        file_path: Path to the file being fixed
        annotated_code: Code with node_id annotations
        intended_behavior: Expected behavior after fix
        target_node_id: The node_id to target for editing
        test_failure_output: Stderr/traceback from failing test (optional but highly recommended)
    """
    # Create an example output for format alignment
    example_output = {
        "op": "replace_expr",
        "target": {"node_id": target_node_id},
        "new_code": "your_fixed_code_here"
    }
    example_json = json.dumps(example_output, indent=2)
    
    # Build the test failure section if available
    test_failure_section = ""
    if test_failure_output and test_failure_output.strip():
        # Truncate if too long (keep last 2000 chars which usually has the actual error)
        truncated_output = test_failure_output[-2000:] if len(test_failure_output) > 2000 else test_failure_output
        test_failure_section = f"""
<test_failure>
The following test failure was observed when running the failing test:

{truncated_output}

This shows the EXACT symptom of the bug. Use this to understand what's going wrong.
</test_failure>

"""
    
    return f"""
<issue>
{issue}
</issue>

<file>
{file_path}
</file>

<buggy_code>
{annotated_code}
</buggy_code>

{test_failure_section}<intended_behavior>
{intended_behavior}
</intended_behavior>

<instructions>
Fix the code by producing an AST edit targeting node_id: {target_node_id}

Use this STRICT schema:
{schema_description()}

Example output format:
{example_json}

Respond ONLY with valid JSON. Return a single edit object or [] if no fix needed.
</instructions>
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
    annotated_code: str,
    target_node_id: int,
    allow_multiple: bool = False,
    test_failure_output: str = "",
) -> ASTGenerationResult:
    """
    Core LLM call: returns List[ASTEdit].

    Enforcement:
      - Only valid JSON survives
      - Only schema-valid edits survive
      - Optional enforcement of one-edit-only (SWE-bench lite)
    
    Args:
        issue_context: The issue description
        file_path: Path to file being edited
        code_snippet: Original code snippet (unused in current implementation)
        intended_behavior: Expected behavior after fix
        annotated_code: Code with node_id annotations
        target_node_id: Target node for edit
        allow_multiple: Whether to allow multiple edits
        test_failure_output: Stderr/traceback from failing test
    """

    user_prompt = _format_prompt(
        issue_context,
        file_path,
        annotated_code,
        intended_behavior,
        target_node_id,
        test_failure_output,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    content = None

    # Debug: Check if SELECTED_MODEL is None
    if model_common.SELECTED_MODEL is None:
        logger.error("🚨 [AST AGENT DEBUG] SELECTED_MODEL is None! Cannot make LLM call.")
        logger.error("🚨 [AST AGENT DEBUG] This means the model was not initialized in the subprocess.")
        return ASTGenerationResult([], user_prompt, None)
    
    logger.debug(f"🔍 [AST AGENT DEBUG] SELECTED_MODEL: {model_common.SELECTED_MODEL.name if model_common.SELECTED_MODEL else 'None'}")

    try:
        content, *_ = model_common.SELECTED_MODEL.call(
            messages=messages,
            # Removed response_format="json_object" for 30% speed improvement
            # Prompt already requests JSON, and _extract_json_region() handles mixed output
        )
    except AttributeError as e:
        if "'NoneType' object has no attribute 'call'" in str(e):
            logger.error("🚨 [AST AGENT DEBUG] SELECTED_MODEL.call() failed because SELECTED_MODEL is None")
            logger.error(f"🚨 [AST AGENT DEBUG] Exception: {e}")
        logger.debug("LLM call failed: {}", e)
        return ASTGenerationResult([], user_prompt, None)
    except Exception as e:
        logger.debug("LLM call failed: {}", e)
        return ASTGenerationResult([], user_prompt, None)

    if not content:
        logger.debug("LLM returned empty content")
        return ASTGenerationResult([], user_prompt, None)

    content = content.strip()

    try:
        json_text = extract_json_from_response(content)
    except Exception as e:
        logger.debug("Failed to extract JSON from response: {}", e)
        return ASTGenerationResult([], user_prompt, content)

    try:
        edits = parse_edits_from_json_str(json_text)
    except EditSchemaError as e:
        logger.debug("Failed to parse edits from JSON: {}", e)
        return ASTGenerationResult([], user_prompt, content)

    if not allow_multiple and len(edits) > 1:
        return ASTGenerationResult([], user_prompt, content)

    return ASTGenerationResult(edits, user_prompt, content)


def write_ast_edits_for_file(
    file_path: str,
    issue_context: str,
    intended_behavior: str,
    sbfl_line_scores: Optional[Dict[int, float]] = None,
    traceback_text: str = "",
    failing_line: Optional[int] = None,
    top_k: int = 3,
    allow_multiple: bool = False,
) -> ASTGenerationResult:
    """
    Main entry point: Localize faults + Generate AST edits.
    
    This is the unified interface that combines:
      1. localize_fault() - finds suspicious nodes
      2. LLM generation - produces edits for each suspicious node
    
    Args:
        file_path: absolute path to the file to edit
        issue_context: the issue description
        intended_behavior: expected behavior after fix
        sbfl_line_scores: SBFL scores (line_num -> score). If None, starts empty
        traceback_text: stderr from failing test (optional)
        failing_line: deepest failing line number (optional)
        top_k: how many suspicious nodes to localize
        allow_multiple: whether to allow multiple edits per location
    
    Returns:
        ASTGenerationResult with all edits and prompts
    """
    
    # Parse the file
    try:
        root_ast, metadata = parse_file_to_ast(file_path)
    except Exception as exc:
        return ASTGenerationResult([], f"Failed to parse {file_path}: {exc}", None)
    
    # Get SBFL scores if not provided
    if sbfl_line_scores is None:
        sbfl_line_scores = {}
    
    # Localize faults
    try:
        bug_locations = localize_fault(
            root_ast,
            metadata,
            sbfl_line_scores,
            traceback_text,
            failing_line,
            top_k=top_k,
        )
    except Exception as exc:
        return ASTGenerationResult([], f"Localization failed: {exc}", None)
    
    if not bug_locations:
        return ASTGenerationResult([], "No suspicious nodes found by localization", None)
    
    # Get relative path for the prompt
    rel_path = Path(file_path).name  # Just use filename for now
    
    # Generate edits from each bug location
    all_edits: List[ASTEdit] = []
    all_prompts: List[str] = []
    last_response = None
    
    for bug_loc in bug_locations:
        # Extract subtree source
        try:
            subtree_source = ast_to_source(bug_loc.subtree)
        except Exception:
            continue
        
        # Check size before proceeding (skip if too large)
        source_lines = subtree_source.count('\n') + 1
        if source_lines > 30:
            logger.debug("Skipping oversized subtree ({} lines) at node {}", 
                        source_lines, bug_loc.node_id)
            continue
        
        logger.debug("Processing bug location {} with {} lines of code", 
                    bug_loc.node_id, source_lines)
        
        # Create annotated code instead of AST dump (40x token reduction!)
        try:
            annotated_code = annotate_code_with_node_ids(
                subtree_source,
                metadata,
                bug_loc.node_id,
                max_annotations=15
            )
        except Exception as e:
            logger.debug("Failed to annotate code: {}", e)
            # Fallback to unannotated code
            annotated_code = subtree_source
        
        # Format prompt with annotated code and test failure info
        user_prompt = _format_prompt(
            issue_context,
            rel_path,
            annotated_code,
            intended_behavior,
            bug_loc.node_id,
            traceback_text,  # Pass test failure output to LLM
        )
        all_prompts.append(user_prompt)
        
        # Call LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        # Debug: Check if SELECTED_MODEL is None before calling
        if model_common.SELECTED_MODEL is None:
            logger.error("🚨 [AST AGENT DEBUG] SELECTED_MODEL is None at node {}! Skipping.", bug_loc.node_id)
            continue
        
        logger.debug(f"🔍 [AST AGENT DEBUG] Calling LLM for node {bug_loc.node_id} with model: {model_common.SELECTED_MODEL.name}")
        
        try:
            content, *_ = model_common.SELECTED_MODEL.call(
                messages=messages,
                # Removed response_format="json_object" for 30% speed improvement
                # Prompt already requests JSON, and _extract_json_region() handles mixed output
            )
        except AttributeError as e:
            if "'NoneType' object has no attribute 'call'" in str(e):
                logger.error("🚨 [AST AGENT DEBUG] SELECTED_MODEL.call() failed for node {} because SELECTED_MODEL is None", bug_loc.node_id)
                logger.error(f"🚨 [AST AGENT DEBUG] Exception: {e}")
            logger.debug("LLM call failed for node {}: {}", bug_loc.node_id, e)
            continue
        except Exception as e:
            logger.debug("LLM call failed for node {}: {}", bug_loc.node_id, e)
            continue
        
        if not content:
            logger.debug("LLM returned empty content for node {}", bug_loc.node_id)
            continue
        
        content = content.strip()
        last_response = content
        
        # Extract and parse JSON
        try:
            json_text = extract_json_from_response(content)
            edits = parse_edits_from_json_str(json_text)
        except Exception as e:
            logger.debug("Failed to parse edits for node {}: {}", bug_loc.node_id, e)
            continue
        
        # Enforce single edit if needed
        if not allow_multiple and len(edits) > 1:
            continue
        
        all_edits.extend(edits)
    
    combined_prompt = "\n---\n".join(all_prompts) if all_prompts else ""
    return ASTGenerationResult(all_edits, combined_prompt, last_response)


# =============================================================================
# ASTAgent: New orchestrator that replaces PatchAgent
# =============================================================================

class ASTAgent:
    """
    Orchestrator for AST-based patch generation.
    
    This replaces the old PatchAgent which used search APIs.
    
    New flow:
      1. Localize faults using localize_fault()
      2. Generate AST edits using write_ast_edits_for_file()
      3. Apply edits to generate patches
    """
    
    def __init__(
        self,
        task,
        issue_stmt: str,
        output_dir: str,
    ):
        """
        Args:
            task: Task object with project metadata
            issue_stmt: Issue description from the task
            output_dir: Directory to save outputs
        """
        self.task = task
        self.issue_stmt = issue_stmt
        self.output_dir = output_dir
        
        self._request_idx = -1
        self._responses: dict[str, str] = {}
        self._diffs: dict[str, str] = {}
    
    def write_patch_for_file(
        self,
        file_path: str,
        sbfl_line_scores: Optional[Dict[int, float]] = None,
        traceback_text: str = "",
        failing_line: Optional[int] = None,
        top_k: int = 3,
        enable_fast_path: bool = True,
    ) -> tuple[bool, str, str]:
        """
        Generate a patch for a file using AST-based localization and editing.
        
        Args:
            file_path: Absolute path to file to edit
            sbfl_line_scores: SBFL suspiciousness scores (optional)
            traceback_text: Stderr from failing test (optional)
            failing_line: Deepest failing line (optional)
            top_k: Number of suspicious nodes to localize
            enable_fast_path: Whether to try micro-edit fast path before LLM
        
        Returns:
            (success: bool, patch_response: str, diff_content: str)
        """
        from app.ast_repair.apply_edits import apply_edits, ASTEditApplicationError
        from app.ast_repair.diff import unified_diff_str
        
        self._request_idx += 1
        
        # Get original source
        try:
            original_source = Path(file_path).read_text()
        except FileNotFoundError:
            logger.warning("File not found: {}", file_path)
            return False, "File not found", ""
        
        # Parse AST
        try:
            root_ast, metadata = parse_file_to_ast(file_path)
        except Exception as exc:
            logger.warning("Failed to parse {}: {}", file_path, exc)
            return False, f"Parse failed: {exc}", ""
        
        # NEW: Try micro-edit fast path first (if enabled and test_cmd available)
        if enable_fast_path and hasattr(self.task, 'test_cmd'):
            from app.ast_repair.micro_edits import try_micro_edit_fast_path
            from app.ast_repair.localize import localize_fault, get_ranked_node_ids
            
            # Run localization to get ranked nodes
            try:
                if sbfl_line_scores is None:
                    sbfl_line_scores = {}
                
                bug_locations = localize_fault(
                    root_ast,
                    metadata,
                    sbfl_line_scores,
                    traceback_text,
                    failing_line,
                    top_k=top_k,
                )
                
                if bug_locations:
                    # Extract ranked node IDs
                    ranked_node_ids = get_ranked_node_ids(bug_locations)
                    
                    # Try fast path
                    fast_path_result = try_micro_edit_fast_path(
                        root_ast=root_ast,
                        metadata=metadata,
                        sbfl_ranked_nodes=ranked_node_ids,
                        file_path=file_path,
                        task=self.task,
                        max_nodes=5
                    )
                    
                    if fast_path_result:
                        patch_content, description = fast_path_result
                        logger.info("Fast path succeeded with: {}", description)
                        
                        # Store and return
                        handle = f"micro_edit_patch_{self._request_idx}"
                        self._responses[handle] = description
                        self._diffs[handle] = patch_content
                        
                        return True, description, patch_content
            except Exception as e:
                logger.debug("Fast path failed or skipped: {}", e)
        
        # Fast path didn't work - continue with LLM approach (ORIGINAL rankings preserved!)
        # Generate edits using new method
        try:
            generation = write_ast_edits_for_file(
                file_path,
                self.issue_stmt,
                getattr(self.task, "intended_behavior", "Fix the issue"),
                sbfl_line_scores=sbfl_line_scores,
                traceback_text=traceback_text,
                failing_line=failing_line,
                top_k=top_k,
                allow_multiple=False,
            )
        except Exception as exc:
            logger.warning("Edit generation failed: {}", exc)
            return False, f"Edit generation failed: {exc}", ""
        
        if not generation.edits:
            logger.info("No edits generated for {}", file_path)
            return False, "No edits generated", ""
        
        # Apply edits
        try:
            apply_edits(root_ast, metadata, generation.edits)
        except ASTEditApplicationError as exc:
            logger.warning("Failed to apply edits: {}", exc)
            return False, f"Apply failed: {exc}", ""
        
        # Serialize back to source
        try:
            new_source = ast_to_source(root_ast)
        except Exception as exc:
            logger.warning("Serialization failed: {}", exc)
            return False, f"Serialization failed: {exc}", ""
        
        if new_source == original_source:
            logger.info("Edits produced no changes to {}", file_path)
            return False, "No changes", ""
        
        # Generate diff
        rel_path = Path(file_path).relative_to(self.task.repo_path).as_posix() if hasattr(self.task, 'repo_path') else Path(file_path).name
        diff_content = unified_diff_str(original_source, new_source, rel_path)
        
        if not diff_content.strip():
            logger.info("Diff is empty for {}", file_path)
            return False, "Empty diff", ""
        
        # Store results
        handle = f"ast_patch_{self._request_idx}"
        self._responses[handle] = generation.raw_response or json.dumps([e.to_dict() for e in generation.edits])
        self._diffs[handle] = diff_content
        
        return True, self._responses[handle], diff_content
