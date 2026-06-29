#!/usr/bin/env bash
# Train a pelican recognizer on the reallybig call library.
#
# Usage:
#   ./train_pelican.sh pelican0-10
#   ./train_pelican.sh pelican0-10 --epochs 100        (pass extra flags after name)
#   ./train_pelican.sh pelican0-10 --report-helpers    (opt out: keep helpers as
#                                                        positive, reported classes)
#
# DEFAULT BEHAVIOUR: ALL Environment_*/Homo sapiens_* classes are trained as
# non-events (all-zero hard negatives that protect species without an output neuron,
# never reported), with no exceptions. This is baked in as:
#     --non_event_prefixes "Environment_,Homo sapiens_"
# See CLAUDE.md "Design fork" for the rationale.
#
# To turn the helpers back into ordinary modelled-and-reported positive classes
# (the old default), make the active choice:
#   ./train_pelican.sh pelican0-10 --report-helpers
#
# Add --keep-airplane-siren to keep the episodic anthropophony you still want reported
# (Airplane, Siren) as positive classes while the rest stay non-events:
#   ./train_pelican.sh pelican0-10 --keep-airplane-siren
# expands the keep-list to:  --keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren"
#
# --nonevent-helpers is still accepted as a no-op (it is now the default).
#
# For other splits, pass --non_event_prefixes / --keep_as_class directly instead;
# an explicit --non_event_prefixes overrides the baked-in default.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <recognizer-name> [--report-helpers] [--keep-airplane-siren] [extra birdnet_analyzer.train flags]"
    exit 1
fi

NAME="$1"; shift

# Helpers (Environment_*/Homo sapiens_*) are non-events BY DEFAULT. The user can
# opt out with --report-helpers (keep them as positive, reported classes), or
# carve out Airplane/Siren with --keep-airplane-siren. Everything else passes
# straight through; an explicit --non_event_prefixes suppresses the baked-in default.
HELPERS_AS_NONEVENTS=true
KEEP_AIRPLANE_SIREN=false
USER_SET_PREFIXES=false
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --report-helpers)
            HELPERS_AS_NONEVENTS=false
            ;;
        --nonevent-helpers)
            # Now the default; accepted as a no-op for backward compatibility.
            HELPERS_AS_NONEVENTS=true
            ;;
        --keep-airplane-siren)
            KEEP_AIRPLANE_SIREN=true
            ;;
        --non_event_prefixes)
            USER_SET_PREFIXES=true
            EXTRA_ARGS+=("$arg")
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# Inject the default non-event prefixes unless the user opted out or set them explicitly.
if $HELPERS_AS_NONEVENTS && ! $USER_SET_PREFIXES; then
    EXTRA_ARGS+=(--non_event_prefixes "Environment_,Homo sapiens_")
    if $KEEP_AIRPLANE_SIREN; then
        EXTRA_ARGS+=(--keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren")
    fi
fi

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
if $USER_SET_PREFIXES; then
    echo "Helpers:  per explicit --non_event_prefixes"
elif $HELPERS_AS_NONEVENTS; then
    if $KEEP_AIRPLANE_SIREN; then
        echo "Helpers:  Environment_*/Homo sapiens_* = non-events (default), Airplane/Siren reported"
    else
        echo "Helpers:  Environment_*/Homo sapiens_* = non-events (default)"
    fi
else
    echo "Helpers:  Environment_*/Homo sapiens_* = positive reported classes (--report-helpers)"
fi
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
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
