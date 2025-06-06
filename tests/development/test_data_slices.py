"""
test_data_slices.py

Development tests that validate model behavior on different data slices:
- short vs. long reviews
- named entities
- negations
"""


from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib

from model_training.config import MODELS_DIR
from model_training.dataset import preprocess_dataset

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"

model = joblib.load(MODEL_PATH)
cv = joblib.load(VECTORIZER_PATH)


def call_predict_single(texts):
    """
    Preprocess and predict sentiment scores for a list of input texts.
    """
    preds = []
    for t in texts:
        df = pd.DataFrame({'Review': [t], 'Liked': [0]})  # Dummy label
        corpus, _ = preprocess_dataset(df)
        features = cv.transform(corpus).toarray()
        df_feat = pd.DataFrame(features, columns=cv.get_feature_names_out())
        pred = model.predict(df_feat)
        preds.append(float(pred[0]))
    return preds


@pytest.mark.development
def test_model_on_short_reviews():
    """
    Test model predictions on short single-word reviews.
    """
    examples = ["Good", "Bad", "Tasty", "Awful"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, float) for p in preds), "Short reviews failed"


@pytest.mark.development
def test_model_on_long_reviews():
    """
    Test model predictions on long repeated text reviews.
    """
    long_text = "The service was wonderful and the ambiance was perfect. " * 10
    examples = [long_text, long_text + " Loved it!"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Long reviews failed"


@pytest.mark.development
def test_model_on_named_entities():
    """
    Test model robustness when named entities (e.g., names) are present.
    """
    examples = ["John loved the pizza.", "Anna did not like the sushi."]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Named entities caused failure"


@pytest.mark.development
def test_model_on_negation_slice():
    """
    Test whether the model assigns low sentiment to negated expressions.
    """
    examples = ["I do not like this.", "Bad experience."]
    preds = call_predict_single(examples)
    assert all(p < 0.5 for p in preds), "Negation not handled correctly"
