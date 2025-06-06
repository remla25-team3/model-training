from pathlib import Path
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_training.dataset import download_dataset, preprocess_dataset
from model_training.features import main

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
    corpus, labels = preprocess_dataset(raw_df)
    pre_df = pd.DataFrame({'Review': corpus, 'Liked': labels})
    pre_df.to_csv(PREPROCESSED_PATH, index=False, header=False)

    return raw_df, pre_df


# RAW DATASET TESTS


# Data and features expectations (schema)


@pytest.mark.data
def test_raw_columns_match_schema(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    expected_columns = {"Review", "Liked"}
    assert set(raw_df.columns) == expected_columns, (
        f"Raw columns {set(raw_df.columns)} != expected {expected_columns}"
    )


@pytest.mark.data
def test_raw_dtypes_match_schema(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    # Expect 'Review' to be string, 'Liked' to be int or 0/1
    assert raw_df['Review'].dtype == object or pd.api.types.is_string_dtype(raw_df['Review']), \
        f"'Review' dtype {raw_df['Review'].dtype} is not object/string"
    assert raw_df['Liked'].dtype in [int, 'int64', 'int32'], (
        f"'Liked' dtype {raw_df['Liked'].dtype} is not integer type"
    )


@pytest.mark.data
def test_no_nulls_in_raw_data(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    assert raw_df['Review'].notnull().all(), "Null values found in 'Review' column"
    assert raw_df['Liked'].notnull().all(), "Null values found in 'Liked' column"


@pytest.mark.data
def test_class_balance_not_extreme(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    counts = raw_df['Liked'].value_counts(normalize=True)
    assert counts.min() > 0.1, "Class imbalance too high (one class <10%)"


@pytest.mark.data
def test_review_length_bounds(download_and_preprocess):
    raw_df, _ = download_and_preprocess
    lengths = raw_df['Review'].str.len()
    assert lengths.min() > 0, "Reviews should not be empty"
    assert lengths.max() <= 500, "Reviews should not exceed 500 characters"


# PREPROCESSED DATASET TESTS


@pytest.mark.data
def test_preprocessed_not_empty(download_and_preprocess):
    _, pre_df = download_and_preprocess
    assert not pre_df.empty, "Preprocessed DataFrame should not be empty"


@pytest.mark.data
def test_preprocessed_no_empty_strings(download_and_preprocess):
    _, pre_df = download_and_preprocess
    assert pre_df["Review"].apply(lambda x: isinstance(x, str) and x.strip()).all(), (
        "Some preprocessed rows are empty or not strings"
    )


@pytest.mark.data
def test_no_duplicate_preprocessed_reviews(download_and_preprocess):
    _, pre_df = download_and_preprocess
    dups = pre_df["Review"].duplicated()
    assert dups.sum() == 0, f"Found {dups.sum()} duplicate preprocessed reviews"


# Features adhere to meta‐level requirements (e.g., lowercased, no punctuation, etc.)


@pytest.mark.data
def test_preprocessed_text_clean(download_and_preprocess):
    _, pre_df = download_and_preprocess
    pattern = re.compile(r'^[a-z ]+$')
    invalid_rows = pre_df[~pre_df["Review"].apply(lambda x: bool(pattern.fullmatch(x)))]
    print("Non-matching rows:")
    print(invalid_rows.head(10))
    assert invalid_rows.empty, "Some preprocessed reviews contain invalid characters"


@pytest.mark.data
@pytest.mark.parametrize("invalid", [".", ",", "!", "\n", "\t"])
def test_no_invalid_placeholders_in_review(download_and_preprocess, invalid):
    _, pre_df = download_and_preprocess
    reviews = pre_df["Review"].dropna().astype(str)
    assert not reviews.str.fullmatch(invalid).any(), (
        f"'Review' column should not contain placeholder '{invalid}'"
    )


# Data pipeline has appropriate privacy controls
# (e.g., no email addresses or phone numbers in raw/preprocessed)


@pytest.mark.data
def test_no_email_or_phone_in_raw(download_and_preprocess):
    _, pre_df = download_and_preprocess
    review_series = pre_df["Review"].dropna().astype(str)
    # Simple regex for emails and phone-like patterns
    email_pattern = r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'
    phone_pattern = r'\b(?:\+?\d[\d\-\s]{7,}\d)\b'
    assert not review_series.str.contains(email_pattern).any(), "Found email patterns in raw data"
    assert not review_series.str.contains(phone_pattern).any(), "Found phone‐number patterns in raw data"


# No feature’s cost is too much (covered more fully under infrastructure, but here check raw text length)


@pytest.mark.features
def test_preprocessed_token_length(download_and_preprocess):
    _, pre_df = download_and_preprocess
    token_counts = pre_df["Review"].str.split().str.len()
    assert token_counts.mean() > 2, "Average token count too low"
    assert token_counts.min() > 0, "Reviews should not be empty"
    assert token_counts.max() < 1000, "Reviews should not exceed 1000 characters"


# New features can be added quickly (if you add a new column, preprocess should ignore it)


@pytest.mark.extensible
def test_preprocess_handles_extra_columns(tmp_path):
    # Create a DataFrame with extra/unexpected column
    df_extra = pd.DataFrame({
        "Review": ["Hello world", "Test"],
        "Liked": [1, 0],
        "unrelated_column": ["foo", "bar"]
    })
    # Should still preprocess only 'Review' without error
    corpus, _ = preprocess_dataset(df_extra)
    assert isinstance(corpus, list) and all(isinstance(x, str) for x in corpus), \
        "Preprocess failed when DataFrame has extra columns"


# All features are beneficial (no zero-variance)


@pytest.mark.features
def test_preprocessed_has_variance(download_and_preprocess):
    _, pre_df = download_and_preprocess
    # Check that not all rows are identical
    assert pre_df["Review"].nunique() > 1, "Review column has zero variance"


@pytest.mark.data
def test_feature_extraction_pipeline():
    base_dir = Path("tests/data/tmp")
    input_path = base_dir / "data_interim.csv"
    bow_path = base_dir / "bow_sentiment_model_test.pkl"
    output_path = base_dir / "features_test.csv"

    # Ensure input exists
    base_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        pd.Series([
            "Amazing service and food!",
            "Worst experience ever.",
            "Average, nothing special."
        ]).to_csv(input_path, index=False, header=False)

    # Clean previous outputs
    for path in [bow_path, output_path]:
        if path.exists():
            path.unlink()

    # Run the feature extraction
    main(input_path=input_path, bow_path=bow_path, output_path=output_path)

    # Validate output CSV
    assert output_path.exists(), "Output feature CSV not created"
    df = pd.read_csv(output_path)
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    assert df.shape[1] > 0, "No features generated"


@pytest.mark.data
def test_preprocess_dataset_empty_input():
    empty_df = pd.DataFrame(columns=["Review", "Liked"])
    result = preprocess_dataset(empty_df)
    assert result == [], "Expected empty list for empty input"
