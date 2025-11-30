# Testing Instructions

## Quick Test (Single Task)

Test with just one task to verify the fix works:

```bash
cd C:\Users\cypri\Everything\AST-based-APR

# Create a single-task test file
echo "sympy__sympy-24152" > test_single.txt

# Run with debug output
python scripts/run.py conf/vanilla-lite.conf --selected_tasks test_single.txt
```

## Full Test (All 3 Tasks)

Once the single task works, test all 3:

```bash
python scripts/run.py conf/vanilla-lite.conf --selected_tasks vanilla-lite/three_tasks.txt
```

## What to Look For in Logs

### Success Indicators (in order)

1. **Subprocess Initialization** (should appear near start of each task):
   ```
   🔍 [SUBPROCESS DEBUG] Registering models in subprocess...
   🔍 [SUBPROCESS DEBUG] MODEL_HUB before registration: []
   🔍 [SUBPROCESS DEBUG] SELECTED_MODEL before registration: None
   🔍 [SUBPROCESS DEBUG] MODEL_HUB after registration: ['gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13', ...]
   🔍 [SUBPROCESS DEBUG] SELECTED_MODEL after registration: <model object>
   🔍 [SUBPROCESS DEBUG] Setting model to: gpt-4o-2024-05-13
   🔍 [SUBPROCESS DEBUG] SELECTED_MODEL after set_model: <model object>
   ```

2. **AST Agent Model Check** (when generating patches):
   ```
   🔍 [AST AGENT DEBUG] SELECTED_MODEL: gpt-4o-2024-05-13
   🔍 [AST AGENT DEBUG] Calling LLM for node <X> with model: gpt-4o-2024-05-13
   ```

3. **Patch Generation** (should now succeed):
   ```
   INFO | Model (gpt-4o-2024-05-13) API request cost info: input_tokens=XXX, output_tokens=XXX
   INFO | Generated and cached reproducer for future retries
   INFO | Start generating patches with reviewer
   ```

4. **Output Directories**:
   - Should have patches in `applicable_patch/` directory
   - Should NOT have all tasks in `no_patch/` directory

### Failure Indicators (if still broken)

1. **Old Error (means fix didn't apply)**:
   ```
   DEBUG | LLM call failed for node <X>: 'NoneType' object has no attribute 'call'
   ```

2. **Model Registration Failed**:
   ```
   🚨 [SUBPROCESS DEBUG] MODEL_HUB after registration: []
   ```

3. **Model Not Set**:
   ```
   🔍 [SUBPROCESS DEBUG] SELECTED_MODEL after set_model: None
   ```

4. **AST Agent Detects None**:
   ```
   🚨 [AST AGENT DEBUG] SELECTED_MODEL is None at node <X>! Skipping.
   ```

## Interpreting Results

### Complete Success
- All 3 tasks complete without "no patch" errors
- Patches appear in `applicable_patch/` or similar directories
- No 🚨 emoji errors in logs

### Partial Success
- Some tasks generate patches, others don't
- Check individual task logs for specific failure reasons
- May indicate task-specific issues rather than infrastructure problems

### Complete Failure (Same as Before)
- All tasks still end up in `no_patch/`
- 🚨 emoji errors still appear
- Means the fix didn't work or another issue exists

## Verifying the Fix Applied

Check that the code changes are present:

```bash
# Check main.py has the debug logging
grep -A 5 "SUBPROCESS DEBUG" app/main.py

# Check agent_write_ast.py has the debug logging
grep -A 5 "AST AGENT DEBUG" app/agents/agent_write_ast.py
```

## Next Steps Based on Results

### If Test Succeeds
1. Remove or reduce debug logging if desired (keep 🚨 errors)
2. Run full SWE-bench-lite experiment
3. Evaluate results

### If Test Still Fails
1. Check the 🚨 error messages to identify new issues
2. Verify OPENAI_KEY is set correctly
3. Check if MODEL_HUB population is working
4. May need additional debugging

### If Partially Works
1. Compare successful vs failed tasks
2. Look for task-specific patterns
3. May indicate issues with specific types of bugs

## Log File Locations

After running, check these files for detailed logs:
```
vanilla-lite/
├── no_patch/                           # Tasks that failed (should be empty now)
├── applicable_patch/                   # Tasks with valid patches (should have all 3)
├── sympy__sympy-24152_<timestamp>/
│   └── info.log                        # Main log file - check for debug emojis
├── sympy__sympy-24213_<timestamp>/
│   └── info.log
└── sympy__sympy-24909_<timestamp>/
    └── info.log
```

## Expected Timeline

- Single task test: ~3-5 minutes
- All 3 tasks: ~10-15 minutes
- Full SWE-bench-lite (300 tasks): Several hours

## Troubleshooting

### Debug logs not appearing
- Logs use loguru, check log level is DEBUG
- Verify changes were saved to files
- Try restarting Python interpreter

### Models still not registered
- Check if `register_all_models()` is actually being called
- Add print statements if needed
- Verify no exceptions during registration

### API Key Issues
```bash
# Verify key is set
echo $OPENAI_KEY

# Should show: sk-...
# If empty, set it again:
export OPENAI_KEY=your-key-here
```

