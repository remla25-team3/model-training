"""Dataset download and preprocessing pipeline for restaurant sentiment analysis."""

from pathlib import Path

from lib_ml.preprocessing import preprocess
from loguru import logger
import pandas as pd
import typer

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
