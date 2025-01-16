#!/bin/bash

# Ask the user for the memory limit in GB
echo "Enter the memory limit in GB (e.g., 4 for 4GB):"
read MEMORY_GB

# Convert GB to KB (1GB = 1048576 KB)
MEMORY_KB=$((MEMORY_GB * 1048576))

# Limit the virtual memory to MEMORY_KB (KB) using ulimit
ulimit -v $MEMORY_KB

# Run the Python training script
python hyperparam.py
