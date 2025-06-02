import re
import sys
import pytest
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_training.dataset import download_dataset, preprocess_dataset

TMP_DIR = Path("tests/tmp")
RAW_PATH = TMP_DIR / "raw_data.tsv"
PREPROCESSED_PATH = TMP_DIR / "preprocessed.csv"


@pytest.fixture(scope="module")
def download_and_preprocess():
    """
    Downloads raw dataset and preprocesses it using official code.
    Returns: (raw_df, preprocessed_df)
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    raw_df = download_dataset(RAW_PATH)
    assert not raw_df.empty, "Downloaded dataset is empty."

    # Preprocess
    corpus = preprocess_dataset(raw_df)
    pre_df = pd.DataFrame(corpus)
    pre_df.to_csv(PREPROCESSED_PATH, index=False, header=False)

    return raw_df, pre_df


# RAW DATASET TESTS
def test_columns_present(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    assert set(raw_df.columns) == {"Review", "Liked"}

def test_no_nulls_in_raw_data(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    assert raw_df['Review'].notnull().all()
    assert raw_df['Liked'].notnull().all()

def test_liked_is_binary(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    assert raw_df['Liked'].isin([0, 1]).all()

def test_review_is_string(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    assert raw_df['Review'].apply(lambda x: isinstance(x, str)).all()

def test_class_balance_not_extreme(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    counts = raw_df['Liked'].value_counts(normalize=True)
    assert counts.min() > 0.1, "Class imbalance too high"

#PREPROCESSED DATASET TESTS
def test_preprocessed_not_empty(download_and_preprocess):
    _, pre_df = download_and_preprocess
    assert not pre_df.empty

def test_preprocessed_no_empty_strings(download_and_preprocess):
    _, pre_df = download_and_preprocess
    assert pre_df[0].apply(lambda x: isinstance(x, str) and x.strip()).all()

def test_no_duplicate_preprocessed_reviews(download_and_preprocess):
    _, pre_df = download_and_preprocess
    dups = pre_df[0].duplicated()
    assert dups.sum() == 0, f"Found {dups.sum()} duplicate preprocessed reviews"

def test_preprocessed_text_clean(download_and_preprocess):
    _, pre_df = download_and_preprocess
    pattern = re.compile(r'^[a-z ]+$')
    assert pre_df[0].apply(lambda x: bool(pattern.fullmatch(x))).all()

def test_preprocessed_token_length(download_and_preprocess):
    _, pre_df = download_and_preprocess
    token_counts = pre_df[0].str.split().str.len()
    assert token_counts.mean() > 2
    assert token_counts.max() < 100
