"""Train model for sentiment analysis on restaurant reviews."""

from pathlib import Path

from config import MODELS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR
import joblib
from loguru import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import typer

app = typer.Typer()


def train_model(features_path: Path, dataset_path: Path, model_path: Path,
                X_test_path: Path, y_test_path: Path) -> float:
    """
    Trains the sentiment analysis model based on the Restaurant Sentiment
    Analysis project (https://github.com/proksch/restaurant-sentiment).

    The resulting model is stored in model_training/data/sentiment_model.pkl.
    """
    df = pd.read_csv(dataset_path)
    corpus = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    cv = CountVectorizer(max_features=1420)
    X_array = cv.fit_transform(corpus).toarray()
    feature_names = cv.get_feature_names_out()
    X_df = pd.DataFrame(X_array, columns=feature_names)

    joblib.dump(cv, features_path)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_test).to_csv(X_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False, header=False)

    classifier = SVC(probability=True)
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, model_path)

    y_pred = classifier.predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    acc = accuracy_score(y_test, y_pred)
    return acc


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / 'features.csv',
    dataset_path: Path = INTERIM_DATA_DIR / 'data_interim.csv',
    model_path: Path = MODELS_DIR / 'sentiment_model.pkl',
    X_test: Path = PROCESSED_DATA_DIR / "X_test.csv",
    y_test: Path = PROCESSED_DATA_DIR / "y_test.csv",
):
    """
    Trains the sentiment model and stores the model in `models`.
    """
    logger.info("Training sentiment model...")

    accuracy = train_model(features_path, dataset_path, model_path, X_test, y_test)

    logger.success(f"Modeling training complete. Accuracy: {accuracy: .4f}")


if __name__ == "__main__":
    app()
