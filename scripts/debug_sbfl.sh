#!/bin/bash
# Debug script to test SBFL on a single task
# Usage: ./scripts/debug_sbfl.sh sympy__sympy-24152

TASK_ID=${1:-"sympy__sympy-24152"}

echo "Debugging SBFL for task: $TASK_ID"
echo "========================================"

# Source conda
source /root/miniconda3/etc/profile.d/conda.sh

# Determine the environment name from task ID
# Format: repo__project-version -> setup_repo__project__version
ENV_NAME="setup_${TASK_ID/__/__}"
# Need to extract version differently
REPO=$(echo $TASK_ID | cut -d'-' -f1)
ENV_PREFIX="setup_${REPO}"

echo "Looking for environment starting with: $ENV_PREFIX"

# Find matching environment
ENV_NAME=$(conda env list | grep "$ENV_PREFIX" | head -1 | awk '{print $1}')

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Could not find conda environment for $TASK_ID"
    echo "Available environments:"
    conda env list | grep setup_
    exit 1
fi

echo "Using environment: $ENV_NAME"
echo ""

# Activate environment
conda activate "$ENV_NAME"

# Check if pytest-cov is installed
echo "Checking pytest-cov installation..."
if pip show pytest-cov > /dev/null 2>&1; then
    echo "  ✅ pytest-cov is installed"
    pip show pytest-cov | grep Version
else
    echo "  ❌ pytest-cov is NOT installed!"
    echo "  Installing now..."
    pip install pytest-cov coverage
fi

echo ""
echo "Checking coverage installation..."
if pip show coverage > /dev/null 2>&1; then
    echo "  ✅ coverage is installed"
    pip show coverage | grep Version
else
    echo "  ❌ coverage is NOT installed!"
fi

echo ""
echo "========================================"
echo "Running test with coverage..."
echo "========================================"

# Go to project directory (need to find it based on task setup)
# For SWE-bench, projects are usually in /opt/SWE-bench/testbed/
PROJECT_DIR="/opt/SWE-bench/testbed/${REPO}"

if [ ! -d "$PROJECT_DIR" ]; then
    # Try alternative locations
    PROJECT_DIR=$(find /opt/SWE-bench -name "${REPO}*" -type d 2>/dev/null | head -1)
fi

if [ -z "$PROJECT_DIR" ] || [ ! -d "$PROJECT_DIR" ]; then
    echo "WARNING: Could not find project directory for $REPO"
    echo "You may need to run setup first."
    echo ""
    echo "Trying to run pytest anyway to see error..."
    python -m pytest --version
    python -m pytest --cov --help | head -5
else
    echo "Project directory: $PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Try running a simple test with coverage
    echo ""
    echo "Running: python -m pytest --cov --cov-context=test --version"
    python -m pytest --version
    
    # Check if .coverage file gets created
    rm -f .coverage
    echo ""
    echo "Running minimal coverage test..."
    python -c "import coverage; print(f'Coverage version: {coverage.__version__}')"
fi

conda deactivate

echo ""
echo "========================================"
echo "Debug complete"
echo "========================================"

