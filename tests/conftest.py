import joblib
import pandas as pd
import pytest

from modeling.predict import SentimentPredictor


DATA_PATH = ""
SEED = 42


@pytest.fixture
def df():
    df = pd.read_csv(DATA_PATH)
    yield df


@pytest.fixture()
def trained_model():
    trained_model = SentimentPredictor(
        model_path="svc_sentiment_classifier", 
        features_path="csv_sentiment_model.pkl"
    )
    yield trained_model
