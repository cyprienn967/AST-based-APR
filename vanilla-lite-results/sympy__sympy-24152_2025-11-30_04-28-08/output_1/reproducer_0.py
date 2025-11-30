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

if __name__ == "__main__":
    from sympy import *
    from sympy.physics.quantum import *
    
    try:
        U = Operator('U')
        V = Operator('V')
        P = TensorProduct(2*U - V, U + V)
        expanded_P = P.expand(tensorproduct=True)

        # Check if the expansion is correct
        expected_result = TensorProduct(2*U, U) + TensorProduct(2*U, V) - TensorProduct(V, U) - TensorProduct(V, V)

        assert expanded_P == expected_result, "TensorProduct expansion is incorrect!"

    except AssertionError as e:
        print_stacktrace(e)
        raise
    except Exception as e:
        print_stacktrace(e)
        raise

    print("TensorProduct expansion works correctly.")
