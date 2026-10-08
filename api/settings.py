"""API settings. Everything can be overridden with an environment variable."""
import os
from pathlib import Path

from ml import config as ml_config

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

# Folder with the fine-tuned sentiment model (later this can be a Hugging Face model id).
SENTIMENT_MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", str(MODELS_DIR / "sentiment_model"))

# Which churn model to serve. Default: whichever one the training run selected (models/churn_metrics.json).
CHURN_MODEL = os.getenv("CHURN_MODEL") or None

# "Today" for churn scoring. Default: the day after the last order in the data.
CHURN_AS_OF = os.getenv("CHURN_AS_OF") or None

# Same rule as training: only customers who ordered within this many days are scored.
ACTIVE_WITHIN_DAYS = ml_config.ACTIVE_WITHIN_DAYS or 365

# Sentiment scores in this band are reported as "uncertain" instead of being forced to a side.
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.35, 0.65

MAX_TEXT_CHARS = 2000
MAX_BATCH = 64
