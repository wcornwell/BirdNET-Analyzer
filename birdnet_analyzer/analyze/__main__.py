import os

# Set TensorFlow environment variables before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from birdnet_analyzer.analyze.cli import main

main()
