"""
Monitoring tests for production-readiness and long-term model reliability.

Covers:
- Input data invariants: schema, nulls, and value constraints
- Model staleness: ensures the deployed model is recent (e.g., retrained within 90 days)
- Numerical stability: minor input changes (e.g., punctuation) should not significantly shift predictions
- Latency regression: ensures inference time remains within acceptable limits

These tests help detect silent failures or performance degradation in real-world serving environments.
"""

import os
from pathlib import Path
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_training.config import MODELS_DIR

# Paths (reuse from conftest)
VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"


# Data invariants hold for inputs


@pytest.mark.monitoring
def test_data_invariants_hold(df):
    """
    Ensure historic data still meets invariants: no nulls, column types, value ranges.
    """
    # Columns present
    assert set(df.columns) == {"Review", "Liked"}, "Historic data columns changed."

    # No nulls
    assert df['Review'].notnull().all(), "Null 'Review' found in historic data"
    assert df['Liked'].notnull().all(), "Null 'Liked' found in historic data"

    # Liked still {0,1}
    assert df['Liked'].isin([0, 1]).all(), "Historic 'Liked' has values outside {0,1}"


# Models are not too stale


@pytest.mark.monitoring
def test_model_staleness():
    """
    Ensure model timestamp is within acceptable window (e.g., < 90 days).
    """
    mtime = os.path.getmtime(MODEL_PATH)
    age_days = (time.time() - mtime) / (24 * 3600)
    assert age_days < 90, f"Model is {age_days: .1f} days old, may need retraining."


# Models are numerically stable


@pytest.mark.monitoring
def test_model_prediction_consistency_under_perturbation(trained_model):
    """
    Perturb an input very slightly (e.g., add/remove punctuation) and ensure prediction changes by < ±0.02.
    """
    base_text = "The food was okay."
    variant = "The food was okay"
    df1 = pd.DataFrame({'Review': [base_text]})
    df2 = pd.DataFrame({'Review': [variant]})
    p1 = trained_model.predict(df1)
    p2 = trained_model.predict(df2)
    assert abs(p1 - p2) < 0.02, f"Numerical instability: {p1: .3f} vs {p2: .3f}"


# Compute performance has not regressed


@pytest.mark.monitoring
def test_inference_latency_not_regressed(trained_model):
    """
    Record inference time; assert it stays under a threshold (e.g., 1 second).
    """
    sample = pd.DataFrame({'Review': ["Test"] * 100})
    start = time.time()
    _ = trained_model.predict(sample)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Monitoring: inference latency regressed to {elapsed: .3f}s"
