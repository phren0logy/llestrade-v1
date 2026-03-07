#!/bin/bash
# Launch the dashboard UI with Phoenix observability env vars enabled.
echo "🆕 Starting Llestrade (Dashboard + Phoenix)..."
echo "================================================"
echo ""
echo "This launches the current dashboard interface with Phoenix observability."
echo ""
echo "🔍 Phoenix Observability: ENABLED"
echo "📊 Phoenix UI will be available at: http://localhost:6006"
echo "📁 Project: llestrade"
echo ""
echo "Starting application..."
echo ""

# Enable Phoenix observability
export PHOENIX_ENABLED=true
export PHOENIX_PORT=6006
export PHOENIX_PROJECT=llestrade
export PHOENIX_EXPORT_FIXTURES=false

# Run dashboard UI directly
uv run -m src.app
