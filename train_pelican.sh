#!/usr/bin/env bash
# Train a pelican recognizer on the reallybig call library.
#
# Usage:
#   ./train_pelican.sh pelican0-10
#   ./train_pelican.sh pelican0-10 --epochs 100   (pass extra flags after name)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <recognizer-name> [extra birdnet_analyzer.train flags]"
    exit 1
fi

NAME="$1"; shift

TRAIN_DATA="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig"
OUTPUT_DIR="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers"
VENV="$(dirname "$0")/.venv/bin/python"

for suffix in .tflite _Labels.txt _Params.csv _sample_counts.csv _validation_metrics.csv; do
    if [[ -e "$OUTPUT_DIR/$NAME$suffix" ]]; then
        echo "Error: $OUTPUT_DIR/$NAME$suffix already exists." >&2
        echo "Choose a new name to avoid overwriting an existing recognizer." >&2
        exit 1
    fi
done

echo "Training: $NAME"
echo "Data:     $TRAIN_DATA"
echo "Output:   $OUTPUT_DIR/$NAME"
echo ""

"$VENV" -m birdnet_analyzer.train "$TRAIN_DATA" \
    -o "$OUTPUT_DIR/$NAME" \
    --hidden_units 2048 \
    --dropout 0.25 \
    -b 32 \
    --learning_rate 0.0001 \
    --upsampling_mode repeat \
    --upsampling_ratio 0.4 \
    --focal-loss \
    --focal-loss-alpha 0.25 \
    --focal-loss-gamma 3.0 \
    --epochs 50 \
    "$@"
