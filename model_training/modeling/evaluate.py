from pathlib import Path
import json

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from model_training.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from loguru import logger

import typer

app = typer.Typer()


@app.command()
def main(
    test_data_path: Path = INTERIM_DATA_DIR / 'data_interim.csv',
    vectorizer_path: Path = MODELS_DIR / 'bow_sentiment_model.pkl',
    model_path: Path = MODELS_DIR / 'sentiment_model.pkl',
    output_path: Path = Path("output/metrics.json")
):
    """
    Evaluates the trained model on the test set and writes metrics to output/metrics.json.
    """
    logger.info("Evaluating trained sentiment model...")

    # Load preprocessed review data (test set) and labels
    X = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv").values
    y_true = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv", header=None).values.ravel()

    # Load vectorizer and model
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    # Save metrics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.success(f"Evaluation complete. Metrics saved to {output_path}")


if __name__ == "__main__":
    app()
