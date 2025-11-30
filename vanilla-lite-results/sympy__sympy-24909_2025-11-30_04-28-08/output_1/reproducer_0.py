import sys
from sympy.physics.units import milli, W

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

try:
    result = milli * W
    expected = 'milli*watt'
    assert str(result) == expected, f"Expected {expected}, but got {result}"

except AssertionError as e:
    print_stacktrace(e)
    raise
except Exception as e:
    print_stacktrace(e)
    raise
else:
    print("No assertion error occurred.")
