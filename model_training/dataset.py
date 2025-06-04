"""Dataset download and preprocessing pipeline for restaurant sentiment analysis."""

from pathlib import Path

from loguru import logger
import typer

import pandas as pd
#from lib_ml.preprocessing import preprocess











import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import List, Tuple

def _prepare_stopwords() -> set:
    """
    Prepare the stopwords for the preprocessing.
    """
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        return stop_words
    except Exception as e:
        print(f"Error preparing stopwords: {e}")
        return set()


def preprocess(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Process reviews from a DataFrame through text preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'Review' column
    
    Returns:
        List[str]: List of processed review texts
    
    Note:
        The function uses a default set of English stopwords (with "not" excluded) prepared by the `_prepare_stopwords` function.
    """
    # if 'Review' not in df.columns:
    #     raise ValueError("DataFrame must contain a 'Review' column")
        
    # stop_words = _prepare_stopwords()
    # corpus = []
    # ps = PorterStemmer()
    # seen = set()
    # pattern = re.compile(r'^[a-z ]+$') # allow only lowercase letters and spaces

    # # Ensure string type
    # df['Review'] = df['Review'].astype(str)

    # for review in df['Review']:
    #     # Remove non-alphabetic characters
    #     review = re.sub(r'[^a-zA-Z]', ' ', review)
    #     # Convert to lowercase
    #     review = review.lower()
    #     # Split into words
    #     words = review.split()
    #     # Remove stopwords and apply stemming
    #     processed_words = [ps.stem(word) for word in words if word not in stop_words]
    #     # Join back into a string
    #     processed_review = ' '.join(processed_words)

    #     # Deduplicate based on processed form
    #     if processed_review and pattern.fullmatch(processed_review) and processed_review not in seen:
    #         seen.add(processed_review)
    #         corpus.append(processed_review)
    
    # return corpus
    if 'Review' not in df.columns:
        raise ValueError("DataFrame must contain a 'Review' column")
    stop_words = _prepare_stopwords()
    ps = PorterStemmer()
    seen = set()
    pattern = re.compile(r'^[a-z ]+$')

    df = df.copy()
    df['Review'] = df['Review'].astype(str)

    corpus = []
    labels = []

    for _, row in df.iterrows():
        review = row['Review']
        label = row[1]  # assumes label is in second column

        review = re.sub(r'[^a-zA-Z]', ' ', review)
        review = review.lower()
        words = review.split()
        processed_words = [ps.stem(word) for word in words if word not in stop_words]
        processed_review = ' '.join(processed_words)

        if processed_review and pattern.fullmatch(processed_review) and processed_review not in seen:
            seen.add(processed_review)
            corpus.append(processed_review)
            labels.append(label)

    return corpus, labels













from model_training.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def download_dataset(save_dir: Path) -> pd.DataFrame:
    """
    Downloads the RestaurantReview dataset and stores it.
    :param save_dir: Directory to save the dataset in
    :return: Object of the dataset converted from CSV
    """
    dataset = pd.read_csv(
        'https://raw.githubusercontent.com/proksch/restaurant-sentiment/refs/heads/main'
        '/a1_RestaurantReviews_HistoricDump.tsv',
        sep='\t'
    )

    # Store the dataset
    dataset.to_csv(save_dir, sep='\t', index=False)

    return dataset


def preprocess_dataset(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Preprocesses the given dataset (csv) using `lib-ml`.
    :param dataset: RestaurantReview dataset as tsv
    :return: Corpus as list of processed review strings
    """
    if len(dataset) == 0 or dataset.shape[0] == 0:
        return []

    corpus, y = preprocess(dataset)

    # for i in range(0, dataset.shape[0]):
    #     review = preprocess(dataset['Review'][i])  # Apply lib-ml
    #     corpus.append(review)

    return corpus, y


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / 'a1_RestaurantReviews_HistoricDump.tsv',
    output_path: Path = INTERIM_DATA_DIR / 'data_interim.csv',
):
    """
    Downloads the RestaurantReviews dataset and applies preprocessing.
    """
    logger.info("Downloading and processing dataset...")

    # Download and store dataset
    dataset = download_dataset(input_path)

    # Apply preprocessing and store preprocessed data
    corpus, labels = preprocess_dataset(dataset)
    df = pd.DataFrame({'Review': corpus, 'Label': labels})
    df.to_csv(output_path, index=False, header=False)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
