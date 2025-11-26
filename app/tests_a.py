
#     python test_ast_repair/tests_personA.py


'''

Run in root (AST-based-APR/)

python -m app.tests_a

if correct, will print 'All Person A tests passed'

'''

import os
import ast
import tempfile
from app.ast_repair.parser import parse_file_to_ast
from app.ast_repair.metadata import ASTMetadata
from app.ast_repair.edit_schema import ASTEdit, parse_edits_from_json_str
from app.ast_repair.apply_edits import apply_edits
from app.ast_repair.serialize import ast_to_source
from app.ast_repair.diff import unified_diff_str

# ---------------------------------------------------------
# Utility to write temp files
# ---------------------------------------------------------
def write_tempfile(contents: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(contents)
    return path

# ---------------------------------------------------------
# Test 1 — replace_expr
# ---------------------------------------------------------
def test_replace_expr():
    src = """
def foo(x):
    return x + 1
"""
    path = write_tempfile(src)

    root, md = parse_file_to_ast(path)

    # Find node_id of expression (x + 1)
    target_id = None
    for nid, node in md.node_index.items():
        if isinstance(node, ast.BinOp):
            target_id = nid
            break
    assert target_id is not None

    edit = ASTEdit(
        op="replace_expr",
        target_node_id=target_id,
        new_code="x + 2",
    )

    apply_edits(root, md, [edit])
    new_src = ast_to_source(root)

    assert "x + 2" in new_src

# ---------------------------------------------------------
# Test 2 — replace_stmt
# ---------------------------------------------------------
def test_replace_stmt():
    src = """
def foo():
    a = 1
    b = 2
    return a + b
"""
    path = write_tempfile(src)
    root, md = parse_file_to_ast(path)

    # Replace b = 2 with b = 99
    target_id = None
    for nid, node in md.node_index.items():
        if isinstance(node, ast.Assign) and node.targets[0].id == "b":
            target_id = nid
            break
    assert target_id is not None

    edit = ASTEdit(
        op="replace_stmt",
        target_node_id=target_id,
        new_code="b = 99",
    )

    apply_edits(root, md, [edit])
    new_src = ast_to_source(root)
    assert "b = 99" in new_src

# ---------------------------------------------------------
# Test 3 — insert_before
# ---------------------------------------------------------
def test_insert_before():
    src = """
def foo():
    x = 1
    return x
"""
    path = write_tempfile(src)
    root, md = parse_file_to_ast(path)

    # Insert before: print("debug") before x = 1
    target_id = None
    for nid, node in md.node_index.items():
        if isinstance(node, ast.Assign):
            target_id = nid
            break
    assert target_id is not None

    edit = ASTEdit(
        op="insert_before",
        target_node_id=target_id,
        new_code="print('debug')",
    )

    apply_edits(root, md, [edit])
    new_src = ast_to_source(root)
    assert "print('debug')" in new_src.splitlines()[1]

# ---------------------------------------------------------
# Test 4 — insert_after
# ---------------------------------------------------------
def test_insert_after():
    src = """
def foo():
    x = 1
    return x
"""
    path = write_tempfile(src)
    root, md = parse_file_to_ast(path)

    target_id = None
    for nid, node in md.node_index.items():
        if isinstance(node, ast.Assign):
            target_id = nid
            break
    assert target_id is not None

    edit = ASTEdit(
        op="insert_after",
        target_node_id=target_id,
        new_code="print('done')",
    )

    apply_edits(root, md, [edit])
    new_src = ast_to_source(root)
    assert "print('done')" in new_src

# ---------------------------------------------------------
# Test 5 — delete
# ---------------------------------------------------------
def test_delete():
    src = """
def foo():
    x = 1
    y = 2
    return x + y
"""
    path = write_tempfile(src)
    root, md = parse_file_to_ast(path)

    to_delete = None
    for nid, node in md.node_index.items():
        if isinstance(node, ast.Assign) and node.targets[0].id == "y":
            to_delete = nid
            break
    assert to_delete is not None

    edit = ASTEdit(op="delete", target_node_id=to_delete)

    apply_edits(root, md, [edit])
    new_src = ast_to_source(root)
    assert "y = 2" not in new_src

# ---------------------------------------------------------
# Test 6 — diff
# ---------------------------------------------------------
def test_diff():
    old = "x = 1\n"
    new = "x = 2\n"
    diff = unified_diff_str(old, new, filename="file.py")
    assert "-x = 1" in diff and "+x = 2" in diff

# ---------------------------------------------------------
# Test 7 — schema parser
# ---------------------------------------------------------
def test_schema_parser():
    json_str = '{"op":"replace_expr","target":{"node_id":5},"new_code":"x+1"}'
    edits = parse_edits_from_json_str(json_str)
    assert len(edits) == 1
    assert edits[0].op == "replace_expr"
    assert edits[0].target_node_id == 5
    assert edits[0].new_code == "x+1"


if __name__ == "__main__":
    # Run manually without pytest
    test_replace_expr()
    test_replace_stmt()
    test_insert_before()
    test_insert_after()
    test_delete()
    test_diff()
    test_schema_parser()
    print("All Person A tests passed.")
