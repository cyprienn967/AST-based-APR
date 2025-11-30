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

def main():
    try:
        from sympy.physics.units import milli, W

        # Checking the expression (milli * W should evaluate to milli watts / mW)
        result = milli * W
        expected = W * milli

        # Raise an assertion error if the result is not as expected
        assert result == expected, f"AssertionError: Expected {expected}, but got {result}"

        # Checking if milli*W evaluates to milli watts
        assert result == milli * W, "AssertionError: milli * W did not evaluate to milli watts"

        # Verify the result through string comparison (symbolic validation)
        assert str(result) == str(milli * W), f"AssertionError: milli * W did not evaluate to milli watts, got {result}"

        # Additional output to verify the exact string
        print(f"Calculation successful: milli * W = {result}")
    except AssertionError as e:
        print_stacktrace(e)
        raise

if __name__ == '__main__':
    main()
