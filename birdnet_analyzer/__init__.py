import os
import warnings

# Set TensorFlow environment variables before any other imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from birdnet_analyzer.analyze import analyze
from birdnet_analyzer.embeddings import embeddings
from birdnet_analyzer.search import search
from birdnet_analyzer.segments import segments
from birdnet_analyzer.species import species
from birdnet_analyzer.train import train

__version__ = "2.3.0"
__all__ = ["analyze", "embeddings", "search", "segments", "species", "train"]

warnings.filterwarnings("ignore")
