"""
Infrastructure-level tests to ensure the ML pipeline is reproducible, efficient, and complete.

Covers:
- Model artifacts: file existence, reload consistency, size constraints
- End-to-end training and evaluation workflows: correctness and output validation
- Feature extraction: dimensionality bounds and transformation speed
- Non-functional requirements: inference latency and memory usage
- Dataset ingestion and preprocessing: full pipeline execution via dataset.main()
"""


import json
from pathlib import Path
import sys
import time
import tracemalloc

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_training.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from model_training.dataset import main as dataset_main
from model_training.dataset import preprocess_dataset
from model_training.modeling.evaluate import main as evaluate_main
from model_training.modeling.train import train_model

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
EXAMPLE_INPUTS = ["The food was delicious!", "I did not enjoy the service."]


def prepare_input(texts: list[str]) -> pd.DataFrame:
    """
    Preprocess input texts into a feature matrix compatible with the trained model.
    """
    df = pd.DataFrame({'Review': texts, 'Label': [1] * len(texts)})  # dummy labels
    corpus, _ = preprocess_dataset(df)
    cv = joblib.load(VECTORIZER_PATH)
    features = cv.transform(corpus).toarray()
    return pd.DataFrame(features, columns=cv.get_feature_names_out())


# TRAINING IS REPRODUCIBLE

@pytest.mark.infrastructure
def test_model_artifact_exists():
    """
    Check if the trained model file exists on disk.
    """
    assert MODEL_PATH.exists(), (
        f"Model file not found: {MODEL_PATH}. "
        "Run `dvc pull` or check training pipeline."
    )


@pytest.mark.infrastructure
def test_model_roundtrip_prediction_consistency(tmp_path):
    """
    Ensure that saving and reloading the model does not alter predictions.
    """
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    orig_preds = model.predict(input_df)

    dump_path = tmp_path / "model_reloaded.pkl"

    joblib.dump(model, dump_path)
    reloaded = joblib.load(dump_path)

    new_preds = reloaded.predict(input_df)
    assert all(a == b for a, b in zip(orig_preds, new_preds)), "Predictions changed after reload."


# Non-functional requirements

@pytest.mark.infrastructure
def test_model_inference_latency():
    """
    Ensure model inference runs within acceptable time bounds.
    """
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    start = time.time()
    _ = model.predict(input_df)
    duration = time.time() - start
    assert duration < 1.0, f"Inference latency too high: {duration: .3f}s"


@pytest.mark.infrastructure
def test_model_inference_memory_usage():
    """
    Measure and verify peak memory usage during inference.
    """
    model = joblib.load(MODEL_PATH)
    input_df = prepare_input(EXAMPLE_INPUTS)
    tracemalloc.start()
    _ = model.predict(input_df)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_MB = peak / (1024 * 1024)
    assert peak_MB < 100, f"Inference used too much memory: {peak_MB: .2f} MB"


@pytest.mark.infrastructure
def test_feature_extraction_latency():
    """
    Check that transforming inputs into feature vectors is fast.
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    sample = ["This is a test review."] * 1000
    start = time.time()
    _ = vectorizer.transform(sample)
    duration = time.time() - start
    assert duration < 0.1, f"Feature extraction too slow: {duration: .3f}s"


@pytest.mark.infrastructure
def test_feature_vector_size_limit():
    """
    Ensure the number of features (dimensionality) remains bounded to control cost.
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    num_features = len(vectorizer.get_feature_names_out())
    assert num_features < 5000, f"Too many features ({num_features}), consider pruning low‐frequency tokens."


@pytest.mark.infrastructure
def test_evaluation_pipeline(tmp_path):
    """
    Run the full evaluation pipeline and verify the output metrics file.
    """
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

    feature_names = vectorizer.get_feature_names_out()

    x_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    pd.DataFrame(X_test, columns=feature_names).to_csv(x_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False, header=False)

    # Valuta
    output_metrics = tmp_path / "metrics.json"
    evaluate_main(  # chiama la funzione CLI come callable
        model_path=MODELS_DIR / "sentiment_model.pkl",
        output_path=output_metrics
    )

    # Verifica esistenza e contenuto
    assert output_metrics.exists(), "Evaluation did not produce a metrics file"
    with open(output_metrics, encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics, "Missing accuracy in metrics"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy is outside valid range"


@pytest.mark.infrastructure
def test_train(tmp_path):
    """
    Train the model from scratch on a small test set and check output accuracy.
    """
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
    df = pd.DataFrame({"Review": reviews, "Label": labels})
    df.to_csv(dataset_path, index=False, header=False)

    # Feature extraction (simulate featurize stage)
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(df["Review"]).toarray()
    features_path = tmp_path / "features.csv"
    pd.DataFrame(X, columns=cv.get_feature_names_out()).to_csv(features_path, index=False)

    # Define output file paths
    model_path = tmp_path / "model.pkl"
    x_test_path = tmp_path / "X_test.csv"
    y_test_path = tmp_path / "y_test.csv"

    # Train the model
    acc = train_model(
        features_path=features_path,
        dataset_path=dataset_path,
        model_path=model_path,
        X_test_path=x_test_path,
        y_test_path=y_test_path
    )

    assert 0.0 <= acc <= 1.0, "Returned accuracy must be in [0, 1]"


@pytest.mark.infrastructure
def test_dataset_main_pipeline(tmp_path):
    """
    Run the dataset main() pipeline end-to-end.
    Verifies that the raw dataset is downloaded, preprocessed, and stored.
    """
    input_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "processed.csv"

    # Run the full CLI-like pipeline
    dataset_main(input_path=input_path, output_path=output_path)

    # Check that files were created
    assert input_path.exists(), f"Raw dataset was not saved to {input_path}"
    assert output_path.exists(), f"Processed dataset was not written to {output_path}"

    # Validate processed output format
    df = pd.read_csv(output_path, header=None)
    assert df.shape[1] == 2, f"Processed dataset should have 2 columns (Review, Label), got {df.shape[1]}"
    assert df.shape[0] > 0, "Processed dataset is empty"

    # Check basic content structure
    reviews = df.iloc[:, 0]
    labels = df.iloc[:, 1]
    assert all(isinstance(r, str) and r.strip() for r in reviews), "Some reviews are invalid strings"
    assert labels.isin([0, 1]).all(), "Labels must be binary (0 or 1)"
