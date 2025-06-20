"""Feature extraction from text corpus"""

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import typer

from model_training.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / 'data_interim.csv',
    bow_path: Path = MODELS_DIR / 'bow_sentiment_model.pkl',
    output_path: Path = PROCESSED_DATA_DIR / 'features.csv',
):
    """
    Transforms the interim dataset into features, and stores the resulting
    file in `data/processed`.
    """
    logger.info("Generating features from dataset...")

    corpus = pd.read_csv(input_path, header=None)[0].dropna()

    cv = CountVectorizer(max_features=1420)
    x = cv.fit_transform(corpus).toarray()

    # Export BoW dictionary to later use in prediction
    try:
        joblib.dump(cv, bow_path)
    except OSError as exc:
        logger.error(f"Error storing sentiment model: {exc}")

    # Export features
    df = pd.DataFrame(x, columns=cv.get_feature_names_out())
    df.to_csv(output_path, index=False)

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
