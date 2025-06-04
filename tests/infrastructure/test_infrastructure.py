from pathlib import Path
import pickle
import sys
import time
import tracemalloc

import joblib
import pandas as pd
import pytest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_training.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
from model_training.dataset import preprocess_dataset
from model_training.modeling.evaluate import main as evaluate_main
from model_training.modeling.train import train_model
from model_training import config as cfg

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
EXAMPLE_INPUTS = ["The food was delicious!", "I did not enjoy the service."]


def prepare_input(texts: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({'Review': texts, 'Label': [1]*len(texts)})  # dummy labels
    corpus, _ = preprocess_dataset(df)
    cv = joblib.load(VECTORIZER_PATH)
    features = cv.transform(corpus).toarray()
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

@pytest.mark.infrastructure
def test_evaluation_pipeline(tmp_path):

    vectorizer = joblib.load(VECTORIZER_PATH)

    # Carica il dataset pre-elaborato
    dataset_path = INTERIM_DATA_DIR / "data_interim.csv"
    df = pd.read_csv(dataset_path, header=None)
    corpus = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values

    # Trasforma e salva X_test e y_test
    X = vectorizer.transform(corpus).toarray()
    y = labels

    # Usa solo un sottoinsieme per il test
    X_test = X[:5]
    y_test = y[:5]

    x_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    pd.DataFrame(X_test).to_csv(x_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False, header=False)

    # Valuta
    output_metrics = tmp_path / "metrics.json"
    evaluate_main(  # chiama la funzione CLI come callable
        model_path=MODELS_DIR / "sentiment_model.pkl",
        output_path=output_metrics
    )

    # Verifica esistenza e contenuto
    assert output_metrics.exists(), "Evaluation did not produce a metrics file"
    with open(output_metrics) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "Missing accuracy in metrics"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy is outside valid range"


@pytest.mark.infrastructure
def test_train(tmp_path):
    # Simulated dataset
    reviews = [
        "Excellent service and great food!",
        "Awful experience, never coming back.",
        "Okay meal, decent staff.",
        "Absolutely loved it.",
        "Not worth the price."
    ]
    labels = [1, 0, 1, 1, 0]
    dataset_path = tmp_path / "data_interim.csv"
    pd.DataFrame({"Review": reviews, "Label": labels}).to_csv(dataset_path, index=False, header=False)

    # Define file paths
    features_path = tmp_path / "features.pkl"
    model_path = tmp_path / "model.pkl"
    x_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"
    metrics_path = tmp_path / "metrics.json"

    # Train the model
    acc = train_model(
        features_path=features_path,
        dataset_path=dataset_path,
        model_path=model_path,
        X_test_path=x_test_path,
        y_test_path=y_test_path
    )
    assert 0.0 <= acc <= 1.0, "Returned accuracy must be in [0, 1]"
