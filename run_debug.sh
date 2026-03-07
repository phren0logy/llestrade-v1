#!/bin/bash
# Debug launch script for Llestrade
# This script enables various debugging options to help diagnose crashes

echo "🔍 Starting Llestrade in DEBUG MODE..."
echo "================================================"

# Enable Python fault handler
export PYTHONFAULTHANDLER=1

# Enable Python debug memory allocator
export PYTHONMALLOC=debug

# Enable Qt debug logging
export QT_LOGGING_RULES="*.debug=true"

# Enable application debug mode
export DEBUG=true
export DEBUG_LLM=true
export DEBUG_QT=true

# Disable malloc optimizations on macOS
export MallocNanoZone=0
export MALLOC_CHECK_=3

# Log the environment
echo "Debug environment variables set:"
echo "  PYTHONFAULTHANDLER=1"
echo "  PYTHONMALLOC=debug" 
echo "  QT_LOGGING_RULES=*.debug=true"
echo "  DEBUG=true"
echo "  MallocNanoZone=0 (macOS specific)"
echo "  MALLOC_CHECK_=3"
echo ""

# Run the application with uv
echo "Launching application..."
echo "================================================"
uv run main.py "$@"
echo ""
echo "Hint: use run_new_ui.sh if you want to launch with Phoenix env vars pre-set."
