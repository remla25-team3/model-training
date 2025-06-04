from pathlib import Path
import pickle
import sys
import time
import tracemalloc

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_training.config import MODELS_DIR
from model_training.dataset import preprocess_dataset

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
EXAMPLE_INPUTS = ["The food was delicious!", "I did not enjoy the service."]


def prepare_input(texts: list[str]) -> pd.DataFrame:
    # wrap input in a DataFrame with 'Review' column
    df = pd.DataFrame({'Review': texts})
    # apply preprocessing
    preprocessed = preprocess_dataset(df)  # list of strings
    # load vectorizer and transform
    cv = joblib.load(VECTORIZER_PATH)
    features = cv.transform(preprocessed).toarray()
    # return as DataFrame
    return pd.DataFrame(features, columns=cv.get_feature_names_out())


# TRAINING IS REPRODUCIBLE

@pytest.mark.infrastructure
def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}. Run `dvc pull` or check training pipeline."

@pytest.mark.infrastructure
def test_model_roundtrip_prediction_consistency(tmp_path):
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    orig_preds = model.predict(input_df)

    dump_path = tmp_path / "model_reloaded.pkl"
    with open(dump_path, "wb") as f:
        pickle.dump(model, f)

    with open(dump_path, "rb") as f:
        reloaded = pickle.load(f)

    new_preds = reloaded.predict(input_df)
    assert all(a == b for a, b in zip(orig_preds, new_preds)), "Predictions changed after reload."


# Non-functional requirements

@pytest.mark.infrastructure
def test_model_inference_latency():
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    start = time.time()
    _ = model.predict(input_df)
    duration = time.time() - start
    assert duration < 1.0, f"Inference latency too high: {duration:.3f}s"


@pytest.mark.infrastructure
def test_model_inference_memory_usage():
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    tracemalloc.start()
    _ = model.predict(input_df)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_MB = peak / (1024 * 1024)
    assert peak_MB < 100, f"Inference used too much memory: {peak_MB:.2f} MB"


@pytest.mark.infrastructure
def test_feature_extraction_latency():
    vectorizer = joblib.load(VECTORIZER_PATH)
    sample = ["This is a test review."] * 1000
    start = time.time()
    _ = vectorizer.transform(sample)
    duration = time.time() - start
    assert duration < 0.1, f"Feature extraction too slow: {duration:.3f}s"


@pytest.mark.infrastructure
def test_feature_vector_size_limit():
    """
    Ensure the number of features (dimensionality) remains bounded to control cost.
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    num_features = len(vectorizer.get_feature_names_out())
    assert num_features < 5000, f"Too many features ({num_features}); consider pruning low‐frequency tokens."