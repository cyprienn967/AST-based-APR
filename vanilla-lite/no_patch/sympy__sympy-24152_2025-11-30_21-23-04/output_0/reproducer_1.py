def print_stacktrace(e: Exception):
    import traceback
    import sys
    tb = traceback.extract_tb(e.__traceback__)
    print("Traceback (most recent call last):", file=sys.stderr)
    for frame in tb:
        line_number = frame.lineno
        code_context = frame.line.strip() if frame.line else "Unknown"
        print(f'  File "{frame.filename}"', file=sys.stderr)
        print(f"    {line_number}: {code_context}", file=sys.stderr)
    print(f"{e.__class__.__name__}: {e}", file=sys.stderr)

from sympy import *
from sympy.physics.quantum import *

try:
    # Issue reproduction
    U = Operator('U')
    V = Operator('V')
    P = TensorProduct(2*U - V, U + V)

    expected_result = 2*TensorProduct(U, U) + 2*TensorProduct(U, V) - TensorProduct(V, U) - TensorProduct(V, V)
    
    # The expansion should match this result
    expanded_P = P.expand(tensorproduct=True)

    # Use assert to check the expected behavior
    assert expanded_P == expected_result, "TensorProduct Expansion Issue: The expanded form is incomplete or incorrect. Got: {}".format(expanded_P)

except AssertionError as e:
    print("AssertionError occurred.")
    print_stacktrace(e)
    raise e

except Exception as e:
    print("An unexpected error occurred.")
    print_stacktrace(e)
    raise e

print("Test completed successfully. No issues found in TensorProduct expansion.")
