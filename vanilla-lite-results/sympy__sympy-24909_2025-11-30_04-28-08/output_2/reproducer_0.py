import sys

def print_stacktrace(e: Exception):
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    print("Traceback (most recent call last):", file=sys.stderr)
    for frame in tb:
        line_number = frame.lineno
        code_context = frame.line.strip() if frame.line else "Unknown"
        print(f'  File "{frame.filename}"', file=sys.stderr)
        print(f"    {line_number}: {code_context}", file=sys.stderr)
    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

def test_milli_prefix_issue():
    from sympy.physics.units import milli, W
    
    try:
        assert milli*W != 1, "Bug: milli*W evaluates to 1"
        assert (milli*W).name == "mW", "Bug: milli*W does not evaluate to milli watts (mW)"
    except AssertionError as e:
        print_stacktrace(e)
        raise

if __name__ == "__main__":
    try:
        test_milli_prefix_issue()
        print("No issues detected.")
    except AssertionError:
        sys.exit(1)
    sys.exit(0)
