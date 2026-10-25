import os
from typing import Literal, get_args

from birdnet.globals import (
    ACOUSTIC_MODEL_VERSIONS,
    GEO_MODEL_VERSIONS,
    MODEL_LANGUAGE_EN_US,
    MODEL_LANGUAGES,
    VALID_MODEL_LANGUAGES_V2_4,
    VALID_MODEL_LANGUAGES_V3_0,
)


def _newest_version(versions: tuple[str, ...]) -> str:
    """Return the highest ``"<major>.<minor>"`` version from the given tuple."""
    return max(versions, key=lambda v: tuple(int(part) for part in v.split(".")))


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
RANDOM_SEED: int = 42

# Default model versions, derived from what the installed birdnet ships so they track
# the newest. Geo is never a choice: its newest version always replaces the older ones.
DEFAULT_ACOUSTIC_MODEL_VERSION: str = _newest_version(get_args(ACOUSTIC_MODEL_VERSIONS))
DEFAULT_GEO_MODEL_VERSION: str = _newest_version(get_args(GEO_MODEL_VERSIONS))

# Languages each acoustic version can label predictions in. v2.4 and v3.0 differ, so
# the CLI/GUI offer the union; birdnet validates the (version, language) pair on load.
ACOUSTIC_MODEL_LANGUAGES: dict[str, list[str]] = {
    "2.4": list(VALID_MODEL_LANGUAGES_V2_4),
    "3.0": list(VALID_MODEL_LANGUAGES_V3_0),
}
ALL_MODEL_LANGUAGES: list[str] = sorted(
    set(VALID_MODEL_LANGUAGES_V2_4) | set(VALID_MODEL_LANGUAGES_V3_0)
)

# Languages the newest geo model labels its species list in (v3.0 language set).
GEO_MODEL_LANGUAGES: list[str] = list(VALID_MODEL_LANGUAGES_V3_0)

SCORE_FUNCTIONS = Literal["cosine", "euclidean", "dot"]
CROP_MODES = Literal["center", "first", "segments"]
CODES_FILE: str = os.path.join(SCRIPT_DIR, "eBird_taxonomy_codes_2025E.json")
ALLOWED_FILETYPES: list[str] = [
    "wav",
    "flac",
    "mp3",
    "ogg",
    "m4a",
    "wma",
    "aiff",
    "aif",
]
RESULT_TYPES = Literal["table", "audacity", "kaleidoscope", "csv", "parquet"]
ADDITIONAL_COLUMNS = Literal[
    "lat", "lon", "week", "overlap", "sensitivity", "min_conf", "species_list", "model"
]
OUTPUT_RAVEN_FILENAME: str = "BirdNET_SelectionTable.txt"
OUTPUT_KALEIDOSCOPE_FILENAME: str = "BirdNET_Kaleidoscope.csv"
KALEIDOSCOPE_COLUMNS: list[str] = [
    "INDIR",
    "FOLDER",
    "IN FILE",
    "OFFSET",
    "DURATION",
    "TOP1MATCH",
    "TOP1DIST",
]
OUTPUT_CSV_FILENAME: str = "BirdNET_CombinedTable.csv"
OUTPUT_AUDACITY_FILENAME: str = "BirdNET_AudacityLabels.txt"
OUTPUT_PARQUET_FILENAME: str = "BirdNET_CombinedTable.parquet"
ANALYSIS_PARAMS_FILENAME: str = "birdnet.analyze-params.csv"
TRAIN_PARAMS_SUFFIX: str = ".birdnet.train-params.csv"
LABEL_LANGUAGE: MODEL_LANGUAGES = MODEL_LANGUAGE_EN_US
SAMPLE_CROP_MODES = Literal["center", "first", "segments", "smart"]
NON_EVENT_CLASSES: list[str] = ["noise", "other", "background", "silence"]

# Additional class-name prefixes whose folders are treated as non-events (all-zero
# hard-negative label rows, like NON_EVENT_CLASSES) instead of positive, reportable
# classes. Empty by default; set via --non_event_prefixes. Used to convert helper
# classes (e.g. "Environment_", "Homo sapiens_") into hard negatives that protect
# species without getting an output neuron.
NON_EVENT_PREFIXES: list[str] = []

# Exact class names that stay positive, reportable classes even when they match a
# NON_EVENT_PREFIXES prefix. Set via --keep_as_class (e.g. episodic anthropophony
# like "Homo sapiens_Airplane" you still want reported).
NON_EVENT_KEEP_CLASSES: list[str] = []

# Scientific binomials (lowercase) restricting training to a subset of the library's
# classes -- a site-scoped recognizer trained off the shared library. Matched against
# each class folder's "Genus species" prefix. Empty by default (train on everything);
# set via --species_list. Helper/non-event folders are never filtered by this list.
TRAIN_SPECIES_LIST: list[str] = []

# Path of the file TRAIN_SPECIES_LIST was read from, recorded in the run's params file
# so a site recognizer's scoping is recoverable from its artifacts.
TRAIN_SPECIES_LIST_FILE: str = ""

# How to treat library classes absent from TRAIN_SPECIES_LIST. "non_event" keeps their
# audio as all-zero hard negatives with no output neuron (they still suppress the
# retained species); "drop" excludes them entirely. Only consulted when
# TRAIN_SPECIES_LIST is non-empty. Set via --unlisted.
UNLISTED_HANDLING: str = "non_event"

UPSAMPLING_MODES = Literal["repeat", "linear", "mean", "smote"]
TRAINED_MODEL_OUTPUT_FORMATS = Literal["tflite", "raven", "detached"]
TRAINED_MODEL_SAVE_MODES = Literal["replace", "append"]
AUTOTUNE_METRICS = Literal["val_AUPRC", "val_AUROC"]
