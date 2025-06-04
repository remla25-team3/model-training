import pandas as pd
import sys
from pathlib import Path

# Add the project root (the directory containing model_training/) to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from pathlib import Path
from modeling.predict import SentimentPredictor
from model_training.config import EXTERNAL_DATA_DIR, MODELS_DIR

# Paths to data/model artifacts tracked by DVC
DATA_PATH = EXTERNAL_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
FEATURES_PATH = MODELS_DIR / "bow_sentiment_model.pkl"


@pytest.fixture(scope="session")
def df():
    """Fixture that loads the dataset, assumes it's been pulled via DVC."""
    if not DATA_PATH.exists():
        pytest.fail(f"Dataset file not found at {DATA_PATH}. Did you forget to run `dvc pull`?")
    df = pd.read_csv(DATA_PATH, sep="\t")
    return df


@pytest.fixture(scope="session")
def trained_model():
    """Fixture that loads a trained sentiment model and its feature transformer."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        pytest.fail(
            f"Model files not found. Run `dvc pull` or `dvc repro` first.\n"
            f"Expected: {MODEL_PATH} and {FEATURES_PATH}"
        )
    trained_model = SentimentPredictor(
        model_path=str(MODEL_PATH),
        features_path=str(FEATURES_PATH)
    )
    return trained_model
