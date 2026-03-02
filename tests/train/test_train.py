import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

import pytest

import birdnet_analyzer.config as cfg
from birdnet_analyzer.cli import train_parser
from birdnet_analyzer.train.core import train


@pytest.fixture
def setup_test_environment():
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    classifier_output = os.path.join(output_dir, "classifier_output")

    # Store original config values
    original_config = {attr: getattr(cfg, attr) for attr in dir(cfg) if not attr.startswith("_") and not callable(getattr(cfg, attr))}

    yield {
        "test_dir": test_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "classifier_output": classifier_output,
    }

    # Clean up
    shutil.rmtree(test_dir)

    # Restore original config
    for attr, value in original_config.items():
        setattr(cfg, attr, value)


@patch("birdnet_analyzer.utils.ensure_model_exists")
@patch("birdnet_analyzer.train.utils.train_model")
def test_train_cli(mock_train_model, mock_ensure_model, setup_test_environment):
    env = setup_test_environment

    mock_ensure_model.return_value = True

    parser = train_parser()
    args = parser.parse_args([env["input_dir"], "--output", env["classifier_output"]])

    train(**vars(args))

    mock_ensure_model.assert_called_once()
    mock_train_model.assert_called_once_with()


def test_upsampling_per_class():
    """Verify that upsampling independently brings each class to min_samples.

    The original bug used a global len(y_temp) counter across all classes,
    so after padding the first small class the shared counter prevented
    subsequent classes from being upsampled.
    """
    from birdnet_analyzer.model import upsampling

    # Save and restore config
    original_binary = cfg.BINARY_CLASSIFICATION
    original_seed = cfg.RANDOM_SEED

    try:
        cfg.BINARY_CLASSIFICATION = False
        cfg.RANDOM_SEED = 42

        num_classes = 5
        # Class sizes: 100, 50, 20, 10, 5
        class_sizes = [100, 50, 20, 10, 5]
        total = sum(class_sizes)
        embed_dim = 8

        rng = np.random.default_rng(42)
        x = rng.standard_normal((total, embed_dim))
        y = np.zeros((total, num_classes), dtype=np.float32)

        offset = 0
        for cls_idx, size in enumerate(class_sizes):
            y[offset : offset + size, cls_idx] = 1.0
            offset += size

        # ratio=0.5 → min_samples = 100 * 0.5 = 50
        x_up, y_up = upsampling(x, y, ratio=0.5, mode="repeat")

        # Check per-class counts
        for cls_idx, original_size in enumerate(class_sizes):
            count = int(y_up[:, cls_idx].sum())
            assert count >= 50, f"Class {cls_idx} (original {original_size} samples) has only {count} after upsampling, expected >= 50"

        # The total should be more than the original (some classes needed padding)
        assert len(x_up) > total, f"Expected upsampled data ({len(x_up)}) to exceed original ({total})"

    finally:
        cfg.BINARY_CLASSIFICATION = original_binary
        cfg.RANDOM_SEED = original_seed
