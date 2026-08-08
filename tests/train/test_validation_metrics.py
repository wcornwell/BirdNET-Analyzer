"""Tests for compute_validation_metrics, behind <model>_validation_metrics.csv.

The metrics block partitions output columns into target species vs non_target helpers
and reports micro/macro P/R for each. It used to hand the label matrix to sklearn's
precision_recall_fscore_support, which infers a single-column (n, 1) matrix as *binary*
rather than multilabel — scoring the negative class as a second class, so a
single-species classifier (or any run with exactly one species or one non-target
column) got inflated summaries and per-class rows shifted off their labels. These tests
pin the multi-label framing at every width.

These exercise the pure function rather than training a real head: calling keras'
``model.fit`` under pytest deadlocks on macOS whenever an earlier test has already
imported PyArrow, whose statically-linked absl interposes TensorFlow's (the same hazard
``train_tf_first.py`` exists to avoid).
"""

import numpy as np
import pytest

from birdnet_analyzer.model import compute_validation_metrics

SPECIES = "Pelecanus conspicillatus_Call"
OTHER_SPECIES = "Corvus coronoides_Call"
HELPER = "Homo sapiens_Airplane"


def _random_indicators(n_classes, n_samples=200, seed=0):
    """A (y_true, y_pred) pair of independent random indicator matrices."""
    rng = np.random.default_rng(seed)
    return (
        (rng.random((n_samples, n_classes)) < 0.3).astype(int),
        (rng.random((n_samples, n_classes)) < 0.3).astype(int),
    )


@pytest.mark.parametrize("n_classes", [1, 2, 5, 430])
def test_matches_sklearn_reference(n_classes):
    """Agrees with sklearn's multilabel-indicator scoring at every width.

    sklearn is the reference only where it infers the target type correctly, so the
    one-column case is compared against an explicitly binary call on the positive
    class — which is what a one-column multilabel matrix means.
    """
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    reference = sklearn_metrics.precision_recall_fscore_support
    y_true, y_pred = _random_indicators(n_classes)

    per_class, summaries = compute_validation_metrics(y_true, y_pred)

    if n_classes == 1:
        expected_prec, expected_rec, *_ = reference(
            y_true.ravel(), y_pred.ravel(), average="binary", zero_division=0
        )
        expected_prec, expected_rec = [expected_prec], [expected_rec]
    else:
        expected_prec, expected_rec, *_ = reference(
            y_true, y_pred, average=None, zero_division=0
        )

    assert [p for *_, p, _ in per_class] == pytest.approx(list(expected_prec))
    assert [r for *_, r in per_class] == pytest.approx(list(expected_rec))

    # Micro/macro are derived from the same counts, so they follow per-class.
    assert summaries["overall_macro"] == pytest.approx(
        (np.mean(expected_prec), np.mean(expected_rec))
    )


@pytest.mark.parametrize(
    ("labels", "expected_types"),
    [
        pytest.param([SPECIES], ["species"], id="single_species"),
        pytest.param(
            [SPECIES, "Environment_Rain"], ["species", "non_target"], id="one_of_each"
        ),
        pytest.param(
            [SPECIES, OTHER_SPECIES, "Environment_Rain", HELPER],
            ["species", "species", "non_target", "non_target"],
            id="multi",
        ),
    ],
)
def test_per_class_rows_match_labels(labels, expected_types):
    """One row per output column, correctly named and tagged.

    The binary-inference bug produced two rows for a one-column matrix: the real label
    holding the *negative* class scores, plus a phantom ``class_1``.
    """
    y_true, y_pred = _random_indicators(len(labels))

    per_class, _ = compute_validation_metrics(y_true, y_pred, labels)

    assert [name for _, name, *_ in per_class] == labels
    assert [cls_type for cls_type, *_ in per_class] == expected_types


@pytest.mark.parametrize(
    ("labels", "partition"),
    [
        # One species column overall — the single-species classifier case.
        pytest.param([SPECIES], "species", id="lone_species"),
        # Multi-class model where only the *helper* partition is one column wide: the
        # per-class rows are fine here, but the subset summary took the binary path.
        pytest.param([SPECIES, OTHER_SPECIES, HELPER], "non_target", id="lone_helper"),
    ],
)
def test_single_column_partition_summaries(labels, partition):
    """A one-column partition's summaries equal that column's own P/R.

    Micro and macro over a single class are both just that class, so the summaries must
    reproduce the per-class row exactly. The old binary path averaged in the negative
    class, inflating both.
    """
    y_true, y_pred = _random_indicators(len(labels))

    per_class, summaries = compute_validation_metrics(y_true, y_pred, labels)
    (row,) = [r for r in per_class if r[0] == partition]

    for summary in (f"{partition}_micro", f"{partition}_macro"):
        assert summaries[summary] == pytest.approx(row[2:])


def test_empty_partition_reports_zero():
    """A run with no helper folders reports 0.0 non_target, not NaN."""
    y_true, y_pred = _random_indicators(1)

    _, summaries = compute_validation_metrics(y_true, y_pred, [SPECIES])

    assert summaries["non_target_micro"] == (0.0, 0.0)
    assert summaries["non_target_macro"] == (0.0, 0.0)


def test_zero_division_yields_zero_not_nan():
    """A class never predicted and never present scores 0, matching zero_division=0."""
    y_true = np.zeros((10, 1), dtype=int)
    y_pred = np.zeros((10, 1), dtype=int)

    per_class, summaries = compute_validation_metrics(y_true, y_pred, [SPECIES])

    assert per_class == [("species", SPECIES, 0.0, 0.0)]
    assert summaries["overall_micro"] == (0.0, 0.0)


def test_falls_back_to_indexed_names_when_labels_missing():
    """Unnamed or short label lists degrade to class_<i> rather than raising."""
    y_true, y_pred = _random_indicators(3)

    unnamed, _ = compute_validation_metrics(y_true, y_pred)
    short, _ = compute_validation_metrics(y_true, y_pred, [SPECIES])

    assert [name for _, name, *_ in unnamed] == ["class_0", "class_1", "class_2"]
    assert [name for _, name, *_ in short] == [SPECIES, "class_1", "class_2"]
