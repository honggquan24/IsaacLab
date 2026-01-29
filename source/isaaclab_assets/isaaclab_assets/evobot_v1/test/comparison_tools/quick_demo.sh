#!/bin/bash
# Quick demo script - Generate and display RL vs PID comparison

echo "======================================================================"
echo "QUICK RL vs PID COMPARISON DEMO"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Generate fake RL tracking data"
echo "  2. Generate fake PID tracking data"
echo "  3. Create comparison plots"
echo "  4. Display results"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

# Navigate to project root if needed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"

cd "$PROJECT_ROOT"

# Run the demo
python source/isaaclab_assets/isaaclab_assets/evobot_v1/test/demo_comparison.py \
    --duration 30.0 \
    --step_duration 10.0 \
    --step_values "0.0,0.5,0.0,-0.5,0.0"

echo ""
echo "======================================================================"
echo "Demo complete! Plot window should be open."
echo "Close the plot window to exit."
echo "======================================================================"
