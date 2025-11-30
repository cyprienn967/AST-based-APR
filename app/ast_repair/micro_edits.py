"""
micro_edits.py

Fast path routing using rule-based micro-edits for simple bug detection.

Strategy:
  1. Generate trivial syntactic edits (operator swaps, constant tweaks)
  2. Test each edit quickly (only failing test, 2s timeout)
  3. If any edit passes all tests → EARLY EXIT (fast path)
  4. If all fail → Fall back to LLM repair (slow path)

Key: This is ROUTING, not scoring. Complex bugs keep their rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import ast
import copy
import subprocess
import shlex

from loguru import logger

from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.serialize import ast_to_source


class TestOutcome(Enum):
    ALL_TESTS_PASS = "all_pass"      # Micro-edit fixes the bug! ✅
    PARTIAL = "partial"               # Fixed one test, broke others
    STILL_FAILS = "still_fails"       # Didn't help
    INVALID = "invalid"               # Edit couldn't be applied
    TIMEOUT = "timeout"               # Test hung
    ERROR = "error"                   # Exception during testing


@dataclass
class MicroEdit:
    node_id: int
    description: str
    transform: Callable[[ast.AST], ast.AST]  # Function to transform the node


def is_micro_editable(node: ast.AST) -> bool:
    """Check if node is a good candidate for micro-edit testing."""
    return isinstance(node, (
        ast.Compare,      # x == y, x < y
        ast.BinOp,        # x + 1, x * 2
        ast.Return,       # return x
        ast.Assign,       # x = value
        ast.If,           # if condition:
        ast.UnaryOp,      # not x, -x
        ast.AugAssign,    # x += 1
    ))


def generate_micro_edits(
    node: ast.AST,
    node_id: int,
    metadata: ASTMetadata
) -> List[MicroEdit]:
    """Generate 2-3 rule-based micro-edits for a node."""
    edits = []
    
    # COMPARE NODES (==, !=, <, >, etc.)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = node.ops[0]
        
        if isinstance(op, ast.Eq):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Change == to !=",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left), 
                    ops=[ast.NotEq()], 
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
        elif isinstance(op, ast.NotEq):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Change != to ==",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left), 
                    ops=[ast.Eq()], 
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
        elif isinstance(op, ast.Lt):
            edits.append(MicroEdit(
                node_id=node_id, 
                description="Change < to <=",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left), 
                    ops=[ast.LtE()], 
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
            edits.append(MicroEdit(
                node_id=node_id, 
                description="Change < to >",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left), 
                    ops=[ast.Gt()], 
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
        elif isinstance(op, ast.Gt):
            edits.append(MicroEdit(
                node_id=node_id, 
                description="Change > to >=",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left), 
                    ops=[ast.GtE()], 
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
        elif isinstance(op, ast.LtE):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Change <= to <",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left),
                    ops=[ast.Lt()],
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
        elif isinstance(op, ast.GtE):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Change >= to >",
                transform=lambda n: ast.Compare(
                    left=copy.deepcopy(n.left),
                    ops=[ast.Gt()],
                    comparators=[copy.deepcopy(c) for c in n.comparators]
                )
            ))
    
    # RETURN STATEMENTS
    elif isinstance(node, ast.Return):
        if node.value and isinstance(node.value, ast.Constant):
            val = node.value.value
            
            if isinstance(val, bool):
                # Flip boolean returns
                edits.append(MicroEdit(
                    node_id=node_id,
                    description=f"Change return {val} to {not val}",
                    transform=lambda n, v=not val: ast.Return(
                        value=ast.Constant(value=v)
                    )
                ))
            elif isinstance(val, int):
                # Try common alternatives
                for new_val in [val + 1, val - 1, 0, 1]:
                    if new_val != val:
                        edits.append(MicroEdit(
                            node_id=node_id,
                            description=f"Change return {val} to {new_val}",
                            transform=lambda n, v=new_val: ast.Return(
                                value=ast.Constant(value=v)
                            )
                        ))
        elif node.value is None:
            # Try returning True/False/0/1 instead of None
            for val in [True, False, 0]:
                edits.append(MicroEdit(
                    node_id=node_id,
                    description=f"Change return None to return {val}",
                    transform=lambda n, v=val: ast.Return(
                        value=ast.Constant(value=v)
                    )
                ))
    
    # UNARY OPERATIONS (not x, -x)
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Remove 'not' operator",
                transform=lambda n: copy.deepcopy(n.operand)
            ))
        elif isinstance(node.op, ast.USub):
            edits.append(MicroEdit(
                node_id=node_id,
                description="Remove '-' operator",
                transform=lambda n: copy.deepcopy(n.operand)
            ))
    
    # BINARY OPERATIONS (x + 1, x * 2)
    elif isinstance(node, ast.BinOp):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
            val = node.right.value
            for new_val in [val + 1, val - 1]:
                if new_val != val:
                    edits.append(MicroEdit(
                        node_id=node_id,
                        description=f"Change constant {val} to {new_val}",
                        transform=lambda n, v=new_val: ast.BinOp(
                            left=copy.deepcopy(n.left), 
                            op=copy.deepcopy(n.op),
                            right=ast.Constant(value=v)
                        )
                    ))
    
    # IF STATEMENTS
    elif isinstance(node, ast.If):
        edits.append(MicroEdit(
            node_id=node_id,
            description="Negate if condition",
            transform=lambda n: ast.If(
                test=ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(n.test)),
                body=[copy.deepcopy(s) for s in n.body],
                orelse=[copy.deepcopy(s) for s in n.orelse]
            )
        ))
    
    return edits[:3]  # Max 3 edits per node


def test_micro_edit(
    root_ast: ast.AST,
    metadata: ASTMetadata,
    edit: MicroEdit,
    file_path: str,
    failing_test_cmd: str,
    full_test_cmd: str,
    project_path: str
) -> TestOutcome:
    """
    Apply micro-edit and test it.
    
    Returns outcome without modifying original file permanently.
    """
    from app.ast_repair.apply_edits import replace_node_in_ast
    
    # Deep copy AST
    modified_ast = copy.deepcopy(root_ast)
    
    # Deep copy metadata (for node replacement)
    # Note: We need to rebuild the index with copied nodes
    copied_metadata = copy.deepcopy(metadata)
    
    # Apply edit
    try:
        target_node = copied_metadata.node_index.get(edit.node_id)
        if not target_node:
            return TestOutcome.INVALID
        
        # Transform the node
        modified_node = edit.transform(target_node)
        
        # Replace node in AST
        replace_node_in_ast(modified_ast, edit.node_id, modified_node, copied_metadata)
        
    except Exception as e:
        logger.debug(f"Failed to apply micro-edit: {e}")
        return TestOutcome.INVALID
    
    # Serialize
    try:
        modified_code = ast_to_source(modified_ast)
    except Exception:
        return TestOutcome.INVALID
    
    # Quick syntax check
    try:
        compile(modified_code, '<string>', 'exec')
    except SyntaxError:
        return TestOutcome.INVALID
    
    # Write to file temporarily
    file_path_obj = Path(file_path)
    try:
        backup = file_path_obj.read_text()
    except Exception:
        return TestOutcome.ERROR
    
    try:
        file_path_obj.write_text(modified_code)
        
        # Run ONLY failing test (fast!)
        # Use shlex to properly split command
        test_cmd_parts = shlex.split(failing_test_cmd)
        result = subprocess.run(
            test_cmd_parts,
            cwd=project_path,
            capture_output=True,
            timeout=2,
            text=True
        )
        
        if result.returncode == 0:
            # Failing test now passes! Run full test suite
            full_cmd_parts = shlex.split(full_test_cmd)
            result_full = subprocess.run(
                full_cmd_parts,
                cwd=project_path,
                capture_output=True,
                timeout=30,
                text=True
            )
            
            if result_full.returncode == 0:
                return TestOutcome.ALL_TESTS_PASS
            else:
                return TestOutcome.PARTIAL
        else:
            return TestOutcome.STILL_FAILS
    
    except subprocess.TimeoutExpired:
        return TestOutcome.TIMEOUT
    except Exception as e:
        logger.debug(f"Error testing micro-edit: {e}")
        return TestOutcome.ERROR
    finally:
        # Always restore
        try:
            file_path_obj.write_text(backup)
        except Exception:
            pass


def try_micro_edit_fast_path(
    root_ast: ast.AST,
    metadata: ASTMetadata,
    sbfl_ranked_nodes: List[int],
    file_path: str,
    task: Any,
    max_nodes: int = 5
) -> Optional[tuple[str, str]]:
    """
    Fast path: Test micro-edits on top nodes.
    
    Returns:
        (patch_content, description) if successful, None otherwise
    """
    logger.info("=" * 60)
    logger.info("FAST PATH: Testing micro-edits on top {} nodes", max_nodes)
    logger.info("=" * 60)
    
    # Filter to micro-editable nodes
    candidates = [
        node_id for node_id in sbfl_ranked_nodes[:max_nodes]
        if node_id in metadata.node_index and is_micro_editable(metadata.node_index[node_id])
    ]
    
    logger.info(f"Found {len(candidates)} micro-editable candidates")
    
    if not candidates:
        logger.info("No micro-editable nodes. Skipping fast path.")
        return None
    
    # Extract test commands from task
    # Try to get failing test specifically, fall back to full test
    failing_test_cmd = getattr(task, 'test_cmd', 'pytest')
    full_test_cmd = getattr(task, 'test_cmd', 'pytest')
    project_path = getattr(task, 'project_path', str(Path(file_path).parent))
    
    tested_count = 0
    for node_id in candidates:
        node = metadata.node_index[node_id]
        
        # Generate micro-edits
        micro_edits = generate_micro_edits(node, node_id, metadata)
        
        if not micro_edits:
            continue
        
        logger.debug(f"Node {node_id}: Testing {len(micro_edits)} micro-edits")
        
        for edit in micro_edits:
            tested_count += 1
            logger.debug(f"  [{tested_count}] {edit.description}")
            
            outcome = test_micro_edit(
                root_ast, metadata, edit, file_path,
                failing_test_cmd, full_test_cmd, project_path
            )
            
            if outcome == TestOutcome.ALL_TESTS_PASS:
                logger.info("=" * 60)
                logger.info("✅ FAST PATH SUCCESS!")
                logger.info(f"   Node: {node_id}")
                logger.info(f"   Fix: {edit.description}")
                logger.info(f"   Tests performed: {tested_count}")
                logger.info("=" * 60)
                
                # Generate patch
                try:
                    modified_ast = copy.deepcopy(root_ast)
                    modified_metadata = copy.deepcopy(metadata)
                    
                    # Apply the successful edit
                    from app.ast_repair.apply_edits import replace_node_in_ast
                    target_node = modified_metadata.node_index[edit.node_id]
                    modified_node = edit.transform(target_node)
                    replace_node_in_ast(modified_ast, edit.node_id, modified_node, modified_metadata)
                    
                    modified_code = ast_to_source(modified_ast)
                    original_code = ast_to_source(root_ast)
                    
                    from app.ast_repair.diff import unified_diff_str
                    patch = unified_diff_str(original_code, modified_code, file_path)
                    
                    return (patch, edit.description)
                except Exception as e:
                    logger.warning(f"Failed to generate patch for successful micro-edit: {e}")
                    continue
            
            # Log other outcomes for debugging
            logger.debug(f"    → {outcome.value}")
    
    logger.info(f"Fast path tested {tested_count} micro-edits. None worked.")
    logger.info("Proceeding to slow path (LLM repair)...")
    return None

