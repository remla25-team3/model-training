from pathlib import Path

from loguru import logger
import typer

import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

from model_training.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR

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

    corpus = pd.read_csv(input_path, header=None)[0]

    cv = CountVectorizer(max_features = 1420)
    x = cv.fit_transform(corpus).toarray()

    # Export BoW dictionary to later use in prediction
    try:
        with open(bow_path, 'wb') as bow_f:
            pickle.dump(cv, bow_f)
    except OSError or Exception:
        logger.error("Error storing sentiment model.")

    # Export features
    df = pd.DataFrame(x, columns=cv.get_feature_names_out())
    df.to_csv(output_path, index=False)

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
