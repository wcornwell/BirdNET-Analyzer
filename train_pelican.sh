#!/usr/bin/env bash
# Train a pelican recognizer on the reallybig call library.
#
# BRANCH NOTE: this is the `refactor-to-main-trial` port of main's train_pelican.sh.
# It runs on the birdnet-library training loader (upstream refactor + #939 speedup),
# not main's TFLite pipeline. Flags and hyperparameters are identical to main's version
# so the two produce comparable models for the Phase 4 A/B (see CLAUDE.md testing plan).
# Embeddings are still extracted inline from the reallybig folder (no cache step); the
# refactor's cache path only triggers when INPUT is a cache file.
#
# Usage:
#   ./train_pelican.sh pelican0-10
#   ./train_pelican.sh pelican0-10 --epochs 100        (pass extra flags after name)
#   ./train_pelican.sh bodangora_birdnet0-1 --library /path/to/other_library
#                                                      (train on a library other than
#                                                       reallybig; same recipe either way)
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
    echo "Usage: $0 <recognizer-name> [--library DIR] [--report-helpers] [--keep-airplane-siren] [extra birdnet_analyzer.train flags]"
    exit 1
fi

NAME="$1"; shift

# Helpers (Environment_*/Homo sapiens_*) are non-events BY DEFAULT. The user can
# opt out with --report-helpers (keep them as positive, reported classes), or
# carve out Airplane/Siren with --keep-airplane-siren. Everything else passes
# straight through; an explicit --non_event_prefixes suppresses the baked-in default.
#
# --library DIR trains on a different call library (default: reallybig). The recipe below
# -- non-event prefixes, hyperparameters, output dir -- is unchanged by it: the library is
# an INPUT to this recipe, not a variant of it. The folder must have reallybig's shape
# (`Genus species_Common Name/` class dirs, plus the Environment_*/Homo sapiens_*/Noise
# helpers), because that shape is what the non-event defaults and the downstream
# scientific-name join both key on.
HELPERS_AS_NONEVENTS=true
KEEP_AIRPLANE_SIREN=false
USER_SET_PREFIXES=false
TRAIN_DATA="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig"
EXPECT_LIBRARY=false
SPECIES_LIST=""
UNLISTED="non_event"
EXPECT_SPECIES_LIST=false
EXPECT_UNLISTED=false
EXTRA_ARGS=()
for arg in "$@"; do
    # --library takes a value, so consume the token after it rather than passing it on.
    if $EXPECT_LIBRARY; then
        TRAIN_DATA="$arg"; EXPECT_LIBRARY=false; continue
    fi
    # --species_list / --unlisted are captured (not just passed through) so the banner
    # can report the site scoping, the same way --library is.
    if $EXPECT_SPECIES_LIST; then
        SPECIES_LIST="$arg"; EXPECT_SPECIES_LIST=false
        EXTRA_ARGS+=(--species_list "$arg"); continue
    fi
    if $EXPECT_UNLISTED; then
        UNLISTED="$arg"; EXPECT_UNLISTED=false
        EXTRA_ARGS+=(--unlisted "$arg"); continue
    fi
    case "$arg" in
        --library)
            EXPECT_LIBRARY=true
            ;;
        --species_list)
            EXPECT_SPECIES_LIST=true
            ;;
        --unlisted)
            EXPECT_UNLISTED=true
            ;;
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
$EXPECT_LIBRARY && { echo "Error: --library needs a directory." >&2; exit 1; }
$EXPECT_SPECIES_LIST && { echo "Error: --species_list needs a file." >&2; exit 1; }
$EXPECT_UNLISTED && { echo "Error: --unlisted needs non_event or drop." >&2; exit 1; }
[[ -d "$TRAIN_DATA" ]] || { echo "Error: training library not found: $TRAIN_DATA" >&2; exit 1; }
if [[ -n "$SPECIES_LIST" ]]; then
    [[ -f "$SPECIES_LIST" ]] || { echo "Error: species list not found: $SPECIES_LIST" >&2; exit 1; }
    case "$UNLISTED" in
        non_event|drop) ;;
        *) echo "Error: --unlisted must be non_event or drop (got: $UNLISTED)" >&2; exit 1 ;;
    esac
fi

# Inject the default non-event prefixes unless the user opted out or set them explicitly.
if $HELPERS_AS_NONEVENTS && ! $USER_SET_PREFIXES; then
    EXTRA_ARGS+=(--non_event_prefixes "Environment_,Homo sapiens_")
    if $KEEP_AIRPLANE_SIREN; then
        EXTRA_ARGS+=(--keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren")
    fi
fi

# focal-loss-gamma default is 2.0 (was 3.0): the Smiths Lake gamma sweep showed gamma=3
# over-focused and starved hard/faint positives; gamma=2 recovered recall on the labeled
# soundscape (R@1FP/hr 0.213 -> 0.226, and fewer false negatives). See configs/smithslake_gamma.yaml.
# TRAIN_DATA is set in the argument loop above (default reallybig, override with --library).
OUTPUT_DIR="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers"
VENV="$(dirname "$0")/.venv/bin/python"

# Upstream 2.x replaced <name>_Params.csv with <name>.birdnet.train-params.csv
# (one row per parameter); the preflight tracks the current name.
for suffix in .tflite _Labels.txt .birdnet.train-params.csv _sample_counts.csv _validation_metrics.csv; do
    if [[ -e "$OUTPUT_DIR/$NAME$suffix" ]]; then
        echo "Error: $OUTPUT_DIR/$NAME$suffix already exists." >&2
        echo "Choose a new name to avoid overwriting an existing recognizer." >&2
        exit 1
    fi
done

echo "Training: $NAME"
echo "Data:     $TRAIN_DATA"
echo "Output:   $OUTPUT_DIR/$NAME"
echo "Loader:   birdnet-library (refactor + #939)   [branch: refactor-to-main-trial]"
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
if [[ -n "$SPECIES_LIST" ]]; then
    N_SPECIES=$(grep -cvE '^\s*(#|$)' "$SPECIES_LIST" || true)
    if [[ "$UNLISTED" == "non_event" ]]; then
        echo "Species:  $(basename "$SPECIES_LIST") ($N_SPECIES entries); unlisted classes = non-events (hard negatives)"
    else
        echo "Species:  $(basename "$SPECIES_LIST") ($N_SPECIES entries); unlisted classes DROPPED (not trained on)"
    fi
else
    echo "Species:  all classes in the library (no --species_list)"
fi
echo ""

# Launched via train_tf_first.py (not `-m birdnet_analyzer.train`) so TensorFlow binds
# its absl before the birdnet-library loader imports PyArrow — without this the trainer
# deadlocks at epoch 1 on macOS (see train_tf_first.py / CLAUDE.md).
"$VENV" "$(dirname "$0")/train_tf_first.py" "$TRAIN_DATA" \
    -o "$OUTPUT_DIR/$NAME" \
    --hidden_units 2048 \
    --dropout 0.25 \
    -b 32 \
    --learning_rate 0.0001 \
    --upsampling_mode repeat \
    --upsampling_ratio 0.4 \
    --focal-loss \
    --focal-loss-alpha 0.25 \
    --focal-loss-gamma 2.0 \
    --epochs 50 \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
