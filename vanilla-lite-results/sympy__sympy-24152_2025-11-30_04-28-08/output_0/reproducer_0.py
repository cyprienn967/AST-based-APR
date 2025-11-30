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

try:
    from sympy import symbols
    from sympy.physics.quantum import Operator, TensorProduct
    
    # Define operators
    U = Operator('U')
    V = Operator('V')
    
    # Create a TensorProduct
    P = TensorProduct(2*U - V, U + V)
    
    expected_expansion = 2*TensorProduct(U, U) + 2*TensorProduct(U, V) - TensorProduct(V, U) - TensorProduct(V, V)
    
    # Perform the expansion
    expanded_P = P.expand(tensorproduct=True)
    
    assert expanded_P == expected_expansion, f"Expansion is incomplete or incorrect: {expanded_P}"
    
    print("Test passed. TensorProduct expanded correctly.")
    sys.exit(0)
    
except AssertionError as e:
    print_stacktrace(e)
    raise
except Exception as e:
    print_stacktrace(e)
    raise
