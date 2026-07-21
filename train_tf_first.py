"""Launch birdnet_analyzer.train with TensorFlow initialised BEFORE PyArrow.

The refactor's birdnet-library training loader imports PyArrow (libarrow), whose
statically linked absl interposes TensorFlow's absl on macOS; if libarrow binds first,
TF's eager executor deadlocks on absl::Notification during model.fit (0% CPU hang at
epoch 1). Importing TensorFlow first makes it bind its own absl. Harmless off macOS.

This is why train_pelican.sh invokes the trainer through this launcher instead of
`python -m birdnet_analyzer.train` directly. All CLI args pass straight through
(argparse sees this file's argv[1:] via runpy's alter_sys).
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WRAPT_DISABLE_EXTENSIONS", "true")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf

_ = tf.constant(1.0) + 1.0  # force full eager/threadpool init now

import runpy  # noqa: E402

runpy.run_module("birdnet_analyzer.train", run_name="__main__", alter_sys=True)
