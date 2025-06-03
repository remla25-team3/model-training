import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_training.dataset import preprocess_dataset
from model_training.config import MODELS_DIR
import joblib

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"

model = joblib.load(MODEL_PATH)
cv = joblib.load(VECTORIZER_PATH)

def call_predict_single(texts):
    preds = []
    for t in texts:
        df = pd.DataFrame({'Review': [t]})
        processed = preprocess_dataset(df)
        features = cv.transform(processed).toarray()
        df_feat = pd.DataFrame(features, columns=cv.get_feature_names_out())
        pred = model.predict(df_feat)
        preds.append(float(pred[0]))
    return preds


@pytest.mark.development
def test_model_on_short_reviews():
    examples = ["Good", "Bad", "Tasty", "Awful"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, float) for p in preds), "Short reviews failed"


@pytest.mark.development
def test_model_on_long_reviews():
    long_text = "The service was wonderful and the ambiance was perfect. " * 10
    examples = [long_text, long_text + " Loved it!"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Long reviews failed"


@pytest.mark.development
def test_model_on_named_entities():
    examples = ["John loved the pizza.", "Anna didn't like the sushi."]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Named entities caused failure"


@pytest.mark.development
def test_model_on_negation_slice():
    examples = ["I do not like this.", "I never enjoyed the experience."]
    preds = call_predict_single(examples)
    assert all(p < 0.5 for p in preds), "Negation not handled correctly"
