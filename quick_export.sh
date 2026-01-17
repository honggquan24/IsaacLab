#!/bin/bash
# Quick TensorBoard export script
# Usage: ./quick_export.sh [task_name]

set -e

TASK=${1:-""}
LOGDIR="logs/rsl_rl"
OUTPUT_DIR="tensorboard_exports"

echo "========================================"
echo "  Quick TensorBoard Export"
echo "========================================"
echo ""

# If task specified, find latest run
if [ -n "$TASK" ]; then
    echo "Task: $TASK"

    # Check if task directory exists
    if [ ! -d "$LOGDIR/$TASK" ]; then
        echo "Error: Task directory not found: $LOGDIR/$TASK"
        echo ""
        echo "Available tasks:"
        ls -1 "$LOGDIR/"
        exit 1
    fi

    # Find latest run
    LATEST_RUN=$(ls -t "$LOGDIR/$TASK/" | head -1)
    echo "Latest run: $LATEST_RUN"
    echo ""

    OUTPUT="$OUTPUT_DIR/${TASK}_latest"

    # Export
    echo "Exporting..."
    python export_tensorboard_advanced.py \
        --task "$TASK" \
        --run "$LATEST_RUN" \
        --output "$OUTPUT"

    echo ""
    echo "Generating HTML viewer..."
    python view_tensorboard_exports.py \
        --export-dir "$OUTPUT/$TASK/$LATEST_RUN"

    echo ""
    echo "========================================"
    echo "✓ Done!"
    echo ""
    echo "Open in browser:"
    echo "  file://$(pwd)/$OUTPUT/$TASK/$LATEST_RUN/index.html"
    echo "========================================"

else
    echo "Exporting all tasks..."
    echo ""

    # Export all
    python export_tensorboard_images.py \
        --logdir "$LOGDIR" \
        --output "$OUTPUT_DIR"

    echo ""
    echo "Generating HTML viewer..."
    python view_tensorboard_exports.py \
        --export-dir "$OUTPUT_DIR"

    echo ""
    echo "========================================"
    echo "✓ Done!"
    echo ""
    echo "Open in browser:"
    echo "  file://$(pwd)/$OUTPUT_DIR/index.html"
    echo "========================================"
fi
