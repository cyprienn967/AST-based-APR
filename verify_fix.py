#!/usr/bin/env python3
"""
Quick verification script to test if the subprocess model initialization fix works.
This simulates what happens when tasks run in subprocesses.
"""

import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_registration():
    """Test that models can be registered and used"""
    from app.model import common
    from app.model.register import register_all_models
    
    print("=" * 60)
    print("VERIFICATION TEST: Model Registration in Subprocess")
    print("=" * 60)
    
    # Step 1: Check initial state
    print("\n1. Initial state (should be empty/None):")
    print(f"   MODEL_HUB keys: {list(common.MODEL_HUB.keys())}")
    print(f"   SELECTED_MODEL: {common.SELECTED_MODEL}")
    
    # Step 2: Register models
    print("\n2. Registering models...")
    register_all_models()
    print(f"   MODEL_HUB keys: {list(common.MODEL_HUB.keys())[:5]}... ({len(common.MODEL_HUB)} total)")
    print(f"   SELECTED_MODEL: {common.SELECTED_MODEL}")
    
    # Step 3: Set a model
    print("\n3. Setting model to gpt-4o-2024-05-13...")
    try:
        common.set_model("gpt-4o-2024-05-13")
        print(f"   ✓ SELECTED_MODEL set successfully: {common.SELECTED_MODEL}")
        print(f"   ✓ Model name: {common.SELECTED_MODEL.name}")
    except Exception as e:
        print(f"   ✗ Failed to set model: {e}")
        return False
    
    # Step 4: Verify model has call method
    print("\n4. Verifying model has call method...")
    if common.SELECTED_MODEL is None:
        print("   ✗ SELECTED_MODEL is None!")
        return False
    elif not hasattr(common.SELECTED_MODEL, 'call'):
        print("   ✗ SELECTED_MODEL does not have 'call' method!")
        return False
    else:
        print("   ✓ SELECTED_MODEL has 'call' method")
    
    # Step 5: Check model.setup() was called
    print("\n5. Checking if model client is initialized...")
    if hasattr(common.SELECTED_MODEL, 'client'):
        print(f"   Model has 'client' attribute: {common.SELECTED_MODEL.client}")
        if common.SELECTED_MODEL.client is not None:
            print("   ✓ Model client is initialized")
        else:
            print("   ! Model client is None (needs API key)")
    else:
        print("   Model does not use 'client' attribute (might be LiteLLM)")
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION PASSED: Models can be registered and used")
    print("=" * 60)
    return True


def test_subprocess_simulation():
    """Simulate what happens in a subprocess"""
    from multiprocessing import Process, Queue
    from app.model import common
    from app.model.register import register_all_models
    
    print("\n\n" + "=" * 60)
    print("SUBPROCESS SIMULATION TEST")
    print("=" * 60)
    
    def subprocess_task(queue):
        """This runs in a subprocess"""
        from app.model import common
        from app.model.register import register_all_models
        
        # Check initial state in subprocess
        initial_hub_len = len(common.MODEL_HUB)
        initial_model = common.SELECTED_MODEL
        
        # Register models in subprocess
        register_all_models()
        
        # Set a model
        common.set_model("gpt-4o-2024-05-13")
        
        # Check final state
        final_hub_len = len(common.MODEL_HUB)
        final_model = common.SELECTED_MODEL
        
        results = {
            'initial_hub_len': initial_hub_len,
            'initial_model': str(initial_model),
            'final_hub_len': final_hub_len,
            'final_model_name': final_model.name if final_model else 'None',
            'has_call': hasattr(final_model, 'call') if final_model else False,
        }
        queue.put(results)
    
    # Run in subprocess
    queue = Queue()
    process = Process(target=subprocess_task, args=(queue,))
    process.start()
    process.join()
    
    if not queue.empty():
        results = queue.get()
        print(f"\nSubprocess Results:")
        print(f"  Initial MODEL_HUB length: {results['initial_hub_len']}")
        print(f"  Initial SELECTED_MODEL: {results['initial_model']}")
        print(f"  Final MODEL_HUB length: {results['final_hub_len']}")
        print(f"  Final SELECTED_MODEL: {results['final_model_name']}")
        print(f"  Has 'call' method: {results['has_call']}")
        
        if results['final_hub_len'] > 0 and results['has_call']:
            print("\n✓ SUBPROCESS TEST PASSED: Models work in subprocess")
            return True
        else:
            print("\n✗ SUBPROCESS TEST FAILED: Models not properly initialized in subprocess")
            return False
    else:
        print("\n✗ SUBPROCESS TEST FAILED: No results from subprocess")
        return False


if __name__ == "__main__":
    print("Running verification tests...\n")
    
    # Check if OPENAI_KEY is set (optional, won't actually call API)
    if not os.getenv("OPENAI_KEY"):
        print("⚠ WARNING: OPENAI_KEY not set. This is OK for verification,")
        print("           but you'll need it to actually run experiments.\n")
    
    # Run tests
    test1_passed = test_model_registration()
    test2_passed = test_subprocess_simulation()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Model Registration Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Subprocess Simulation Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 60)
    
    if test1_passed and test2_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe fix should work! You can now run the actual experiment.")
        sys.exit(0)
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nThe fix may not work correctly. Check the output above.")
        sys.exit(1)

