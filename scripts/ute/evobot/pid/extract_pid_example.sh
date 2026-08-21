#!/bin/bash
# Example script to extract PID from trained RL policy

# Path to your trained checkpoint (modify this!)
CHECKPOINT="logs/rsl_rl/evobot_v1_velocity/2026-01-19_10-30-40/model_800.pt"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo "Please update CHECKPOINT variable in this script to point to your trained model"
    exit 1
fi

echo "=================================="
echo "PID EXTRACTION FROM RL POLICY"
echo "=================================="
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "This will take ~5-10 minutes depending on your GPU..."
echo ""

# Run extraction with visualization
./isaaclab.sh -p scripts/ute/evobot/pid/extract_pid_from_rl.py \
    --checkpoint "$CHECKPOINT" \
    --num_envs 128 \
    --num_trajectories 200 \
    --trajectory_length 300 \
    --visualize \
    --alpha 0.1

echo ""
echo "=================================="
echo "EXTRACTION COMPLETED!"
echo "=================================="
echo ""
echo "Check output in: logs/pid_extraction/"
echo ""
echo "To test extracted PID gains, copy the command printed above and run it"
echo ""
