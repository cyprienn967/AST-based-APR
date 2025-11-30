Here is the standalone Python file `reproducer.py`:

```python
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

def reproduce_issue():
    from sympy.physics.units import milli, W

    try:
        assert not (milli * W == 1), "Bug exists: milli*W evaluates to 1"
        assert milli * W == milli * W, "Bug exists: milli*W does not evaluate correctly"
    except AssertionError as e:
        print_stacktrace(e)
        raise

if __name__ == "__main__":
    reproduce_issue()
    print("No issues found.")
```

If the issue is present, the script will raise an `AssertionError`, print the stack trace, and exit with a non-zero code. If the issue is not present, the script will print "No issues found." and exit with code 0. 

To run the reproduction script, save it as `reproducer.py` and execute it from the terminal using `python3 reproducer.py`.