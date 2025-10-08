#!/bin/bash
# Quick start script for PDE optimization examples
#
# Usage:
#   ./run.sh           # Runs all examples
#   ./run.sh 3.1       # Runs specific example

# Activate virtual environment
source venv/bin/activate

# Run examples
cd modular
python run_examples.py "$@"