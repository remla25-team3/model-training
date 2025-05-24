import joblib
import pandas as pd
import pytest


DATA_PATH = ""
TRAINED_MODEL_PATH = ""
SEED = 42


@pytest.fixture
def df():
    df = pd.read_csv(DATA_PATH)
    yield df


@pytest.fixture()
def trained_model():
    trained_model = joblib.load(TRAINED_MODEL_PATH)
    yield trained_model
