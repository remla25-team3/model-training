from pathlib import Path

from loguru import logger
import typer

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib

from model_training.config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


def train_model(features_path: Path, dataset_path: Path, model_path: Path):
    """
    Trains the sentiment analysis model based on the Restaurant Sentiment
    Analysis project (https://github.com/proksch/restaurant-sentiment).

    The resulting model is stored in model_training/data/sentiment_model.pkl.
    """
    features = pd.read_csv(features_path).to_numpy()
    dataset = pd.read_csv(dataset_path, header=None)

    y = dataset.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, model_path)


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / 'features.csv',
    dataset_path: Path = INTERIM_DATA_DIR / 'data_interim.csv',
    model_path: Path = MODELS_DIR / 'sentiment_model.pkl',
):
    """
    Trains the sentiment model and stores the model in `models`.
    """
    logger.info("Training sentiment model...")

    train_model(features_path, dataset_path, model_path)

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
