#!/bin/bash
# Script to install pytest-cov in all SWE-bench conda environments
# Run this in Docker before running evaluate_localization.py

set -e

echo "Installing pytest-cov in all setup_* conda environments..."

# Source conda
source /root/miniconda3/etc/profile.d/conda.sh

# Get all setup_* environments
envs=$(conda env list | grep "setup_" | awk '{print $1}')

if [ -z "$envs" ]; then
    echo "No setup_* environments found!"
    exit 1
fi

echo "Found environments:"
echo "$envs"
echo ""

# Install pytest-cov in each environment
for env in $envs; do
    echo "========================================"
    echo "Installing pytest-cov in: $env"
    echo "========================================"
    
    conda activate "$env"
    
    # Check if pytest-cov is already installed
    if pip show pytest-cov > /dev/null 2>&1; then
        echo "  pytest-cov already installed, skipping..."
    else
        pip install pytest-cov coverage --quiet
        echo "  Installed pytest-cov"
    fi
    
    conda deactivate
done

echo ""
echo "========================================"
echo "Done! pytest-cov installed in all environments."
echo "You can now run the localization evaluation."
echo "========================================"

