#!/bin/bash
# Kura Application Runner
# This script ensures the application runs with the correct Python virtual environment

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$DIR/.venv/bin/activate"

# Run the main application
cd "$DIR"
python3 macos/main.py "$@"

