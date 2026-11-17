import logging
import os

# The macOS XLA-init guard must be in place before anything imports TensorFlow. The
# cosmetic TF variables are left to utils.configure_runtime(), as upstream does, so a
# library import overrides nothing the host has already set.
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

from birdnet_analyzer.settings import (
    apply_matplotlib_config_directory,
    apply_model_directory,
)

# Library convention: emit records but leave the configuration to the host
# application. The shipped CLI and GUI configure handlers via logs.setup_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())

# The birdnet library reads BIRDNET_APP_DATA whenever it needs a model path.
apply_model_directory()
apply_matplotlib_config_directory()

from birdnet_analyzer.analyze import analyze  # noqa: E402
from birdnet_analyzer.embeddings import embeddings  # noqa: E402
from birdnet_analyzer.search import search  # noqa: E402
from birdnet_analyzer.segments import segments  # noqa: E402
from birdnet_analyzer.species import species  # noqa: E402
from birdnet_analyzer.train import train  # noqa: E402

__version__ = "3.0.0"
__all__ = ["analyze", "embeddings", "search", "segments", "species", "train"]
