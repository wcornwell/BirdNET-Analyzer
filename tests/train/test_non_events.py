"""Tests for helpers-as-non-events (--non_event_prefixes / --keep_as_class).

Covers train.utils.is_non_event, which decides whether a folder label is a
hard-negative non-event (all-zero label row, no output neuron) or a positive,
reportable class. The prefix/keep lists are read from config at train time, so each
test sets and restores them explicitly.
"""

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.train.utils import is_non_event


@pytest.fixture(autouse=True)
def _reset_non_event_config():
    """Save/restore the mutable non-event config so tests don't leak into each other."""
    prev_prefixes = cfg.NON_EVENT_PREFIXES
    prev_keep = cfg.NON_EVENT_KEEP_CLASSES
    cfg.NON_EVENT_PREFIXES = []
    cfg.NON_EVENT_KEEP_CLASSES = []
    yield
    cfg.NON_EVENT_PREFIXES = prev_prefixes
    cfg.NON_EVENT_KEEP_CLASSES = prev_keep


def test_builtin_non_event_classes_matched_case_insensitively():
    assert is_non_event("noise") is True
    assert is_non_event("Noise") is True
    assert is_non_event("Background") is True


def test_no_prefixes_means_helpers_are_positive_classes():
    assert is_non_event("Environment_Rain") is False
    assert is_non_event("Homo sapiens_Music") is False


def test_prefix_converts_helpers_to_non_events():
    cfg.NON_EVENT_PREFIXES = ["Environment_", "Homo sapiens_"]
    assert is_non_event("Environment_Rain") is True
    assert is_non_event("Homo sapiens_Music") is True
    # a real species is untouched
    assert is_non_event("Manorina melanocephala_Noisy Miner") is False


def test_keep_as_class_overrides_prefix():
    cfg.NON_EVENT_PREFIXES = ["Environment_", "Homo sapiens_"]
    cfg.NON_EVENT_KEEP_CLASSES = ["Homo sapiens_Airplane", "Homo sapiens_Siren"]
    assert is_non_event("Homo sapiens_Airplane") is False
    assert is_non_event("Homo sapiens_Siren") is False
    # non-kept helpers still convert
    assert is_non_event("Homo sapiens_Music") is True
    assert is_non_event("Environment_Wind") is True


def test_builtin_non_event_wins_even_if_kept():
    # A real NON_EVENT_CLASSES name is always a non-event, regardless of keep-list.
    cfg.NON_EVENT_KEEP_CLASSES = ["noise"]
    assert is_non_event("noise") is True
