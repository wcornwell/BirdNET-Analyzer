"""Tests for site-scoped training (--species_list / --unlisted).

Covers the two decisions a species list drives: which classes get an output neuron
(train.utils.is_selected_species) and what happens to the ones it leaves out --
either all-zero hard negatives via is_non_event ("non_event") or exclusion ("drop").
Also covers _scope_cache_to_species_list, the path that turns one shared library
cache into a per-site recognizer without re-extracting embeddings.

The lists live in config and are set at train time, so each test sets and restores
them explicitly, matching test_non_events.py.
"""

from unittest.mock import patch

import numpy as np
import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.train.core import train
from birdnet_analyzer.train.utils import (
    _scope_cache_to_species_list,
    is_non_event,
    is_selected_species,
    load_species_list,
)

# A stand-in for reallybig's shape: two listed species, one unlisted, plus the helpers.
LISTED = "Manorina melanocephala_Noisy Miner"
ALSO_LISTED = "Acanthiza pusilla_Brown Thornbill"
UNLISTED = "Ardeotis australis_Australian Bustard"
SITE_LIST = ["manorina melanocephala", "acanthiza pusilla"]


@pytest.fixture(autouse=True)
def _reset_config():
    """Save/restore mutable train-time config so tests don't leak into each other."""
    prev = (
        cfg.NON_EVENT_PREFIXES,
        cfg.NON_EVENT_KEEP_CLASSES,
        cfg.TRAIN_SPECIES_LIST,
        cfg.UNLISTED_HANDLING,
    )
    cfg.NON_EVENT_PREFIXES = ["Environment_", "Homo sapiens_"]
    cfg.NON_EVENT_KEEP_CLASSES = []
    cfg.TRAIN_SPECIES_LIST = []
    cfg.UNLISTED_HANDLING = "non_event"
    yield
    (
        cfg.NON_EVENT_PREFIXES,
        cfg.NON_EVENT_KEEP_CLASSES,
        cfg.TRAIN_SPECIES_LIST,
        cfg.UNLISTED_HANDLING,
    ) = prev


# --- the filter is inert by default ------------------------------------------------


def test_no_species_list_selects_everything():
    """The default must be byte-for-byte the old behaviour: no list, no filtering."""
    assert is_selected_species(LISTED) is True
    assert is_selected_species(UNLISTED) is True
    assert is_non_event(UNLISTED) is False


# --- selection ---------------------------------------------------------------------


def test_listed_species_selected_and_unlisted_not():
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    assert is_selected_species(LISTED) is True
    assert is_selected_species(ALSO_LISTED) is True
    assert is_selected_species(UNLISTED) is False


def test_matching_is_case_insensitive_and_ignores_common_name():
    cfg.TRAIN_SPECIES_LIST = ["MANORINA MELANOCEPHALA".lower()]
    assert is_selected_species("Manorina melanocephala_Noisy Miner") is True
    # same binomial, different common name in the folder -> still selected
    assert is_selected_species("Manorina melanocephala_Mickey Bird") is True


def test_helpers_are_never_filtered_by_a_species_list():
    """Helpers have no binomial, so filtering them would delete the hard negatives."""
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    assert is_selected_species("Environment_Rain") is True
    assert is_selected_species("Homo sapiens_Music") is True
    assert is_selected_species("Noise") is True


def test_keep_as_class_helper_survives_a_species_list():
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    cfg.NON_EVENT_KEEP_CLASSES = ["Homo sapiens_Airplane"]
    assert is_selected_species("Homo sapiens_Airplane") is True
    # ... and stays a reported class rather than being demoted
    assert is_non_event("Homo sapiens_Airplane") is False


# --- what happens to the unlisted ---------------------------------------------------


def test_non_event_mode_demotes_unlisted_species_to_hard_negatives():
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    cfg.UNLISTED_HANDLING = "non_event"
    assert is_non_event(UNLISTED) is True
    assert is_non_event(LISTED) is False


def test_drop_mode_does_not_demote_unlisted_species():
    """In drop mode the folders are filtered out upstream, so nothing is demoted."""
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    cfg.UNLISTED_HANDLING = "drop"
    assert is_non_event(UNLISTED) is False
    assert is_non_event(LISTED) is False


def test_helpers_stay_non_events_under_a_species_list():
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    assert is_non_event("Environment_Rain") is True
    assert is_non_event("Noise") is True


# --- list file parsing --------------------------------------------------------------


def test_load_species_list_skips_comments_and_blanks_and_lowercases(tmp_path):
    path = tmp_path / "site.txt"
    path.write_text(
        "# Wild Deserts regional list\n"
        "\n"
        "Manorina melanocephala\n"
        "  Acanthiza Pusilla  \n"
        "   # indented comment\n"
    )
    assert load_species_list(str(path)) == [
        "manorina melanocephala",
        "acanthiza pusilla",
    ]


# --- the cache path: one library extraction -> many site recognizers ----------------


def _cache():
    """4 classes x 5 rows, plus one pre-existing all-zero helper (non-event) row."""
    labels = [LISTED, ALSO_LISTED, UNLISTED, "Corvus orru_Torresian Crow"]
    y = np.array(
        [
            [1, 0, 0, 0],  # listed
            [0, 1, 0, 0],  # listed
            [0, 0, 1, 0],  # unlisted -> orphaned by the slice
            [0, 0, 0, 1],  # unlisted -> orphaned by the slice
            [0, 0, 0, 0],  # helper non-event row, already all-zero
        ],
        dtype="float32",
    )
    x = np.arange(5 * 3, dtype="float32").reshape(5, 3)
    return x, y, np.array([]), np.array([]), labels


def test_cache_scoping_is_inert_without_a_species_list():
    x, y, xt, yt, labels = _cache()
    out = _scope_cache_to_species_list(x, y, xt, yt, labels, False, False)
    assert out[4] == labels
    assert out[1].shape == (5, 4)


def test_cache_non_event_mode_slices_columns_and_keeps_every_row():
    """Orphaned rows become all-zero, which *is* the non-event encoding -- keep them."""
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    cfg.UNLISTED_HANDLING = "non_event"
    x, y, xt, yt, labels = _cache()
    x_out, y_out, _, _, labels_out, is_binary, _ = _scope_cache_to_species_list(
        x, y, xt, yt, labels, False, False
    )
    assert labels_out == [LISTED, ALSO_LISTED]
    assert y_out.shape == (5, 2)
    assert len(x_out) == 5
    # the two unlisted rows and the helper row are all-zero hard negatives now
    assert y_out.any(axis=1).tolist() == [True, True, False, False, False]
    assert is_binary is False


def test_cache_drop_mode_removes_orphaned_rows_but_keeps_helper_rows():
    cfg.TRAIN_SPECIES_LIST = SITE_LIST
    cfg.UNLISTED_HANDLING = "drop"
    x, y, xt, yt, labels = _cache()
    x_out, y_out, _, _, labels_out, _, _ = _scope_cache_to_species_list(
        x, y, xt, yt, labels, False, False
    )
    assert labels_out == [LISTED, ALSO_LISTED]
    # 2 listed rows + the pre-existing all-zero helper row; the 2 orphans are gone
    assert y_out.shape == (3, 2)
    assert len(x_out) == 3
    assert y_out.any(axis=1).tolist() == [True, True, False]
    # the surviving helper row is the one that was already all-zero
    assert x_out[-1].tolist() == x[-1].tolist()


def test_cache_single_species_list_flags_binary():
    cfg.TRAIN_SPECIES_LIST = ["manorina melanocephala"]
    x, y, xt, yt, labels = _cache()
    *_, labels_out, is_binary, is_multi_label = _scope_cache_to_species_list(
        x, y, xt, yt, labels, False, True
    )
    assert labels_out == [LISTED]
    assert is_binary is True
    assert is_multi_label is False


def test_cache_species_list_matching_nothing_raises():
    cfg.TRAIN_SPECIES_LIST = ["Genus nonexistent".lower()]
    x, y, xt, yt, labels = _cache()
    with pytest.raises(ValueError, match="matched none"):
        _scope_cache_to_species_list(x, y, xt, yt, labels, False, False)


# --- CLI wiring ---------------------------------------------------------------------


@patch("birdnet_analyzer.train.utils.train_model")
def test_cli_flags_populate_config(mock_train_model, tmp_path):
    """--species_list / --unlisted must reach the config the loader reads."""
    path = tmp_path / "site.txt"
    path.write_text("Manorina melanocephala\nAcanthiza pusilla\n")

    args = train_parser().parse_args(
        [str(tmp_path), "--species_list", str(path), "--unlisted", "drop"]
    )
    kwargs = vars(args)
    kwargs.pop("load_params")
    train(**kwargs)

    mock_train_model.assert_called_once()
    assert cfg.TRAIN_SPECIES_LIST == SITE_LIST
    assert str(path) == cfg.TRAIN_SPECIES_LIST_FILE
    assert cfg.UNLISTED_HANDLING == "drop"


@patch("birdnet_analyzer.train.utils.train_model")
def test_empty_species_list_file_is_an_error(mock_train_model, tmp_path):
    """A list that parses to nothing would silently train on the whole library."""
    path = tmp_path / "empty.txt"
    path.write_text("# only a comment\n\n")

    args = train_parser().parse_args([str(tmp_path), "--species_list", str(path)])
    kwargs = vars(args)
    kwargs.pop("load_params")

    with pytest.raises(ValueError, match="empty"):
        train(**kwargs)

    mock_train_model.assert_not_called()
