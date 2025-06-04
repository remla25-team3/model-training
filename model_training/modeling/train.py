import joblib
import pandas as pd
import pickle
import typer

from config import MODELS_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from lib_ml.preprocessing import preprocess
from loguru import logger
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

app = typer.Typer()

def train_model(features_path: Path, dataset_path: Path, model_path: Path):
    """
    Trains the sentiment analysis model based on the Restaurant Sentiment
    Analysis project (https://github.com/proksch/restaurant-sentiment).

    The resulting model is stored in model_training/data/sentiment_model.pkl.
    """
    df = pd.read_csv(dataset_path, delimiter='\t', quoting=3)
    corpus = preprocess(df)
    
    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, -1].values[:len(X)]  #ensure X and y are same size
    pickle.dump(cv, open(features_path, "wb"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pd.DataFrame(X_test).to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False, header=False)

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
    dataset_path: Path = EXTERNAL_DATA_DIR / 'a1_RestaurantReviews_HistoricDump.tsv',
    model_path: Path = MODELS_DIR / 'sentiment_model.pkl',
):
    """
    Trains the sentiment model and stores the model in `models`.
    """
    logger.info("Training sentiment model...")

    accuracy = train_model(features_path, dataset_path, model_path)

    logger.success(f"Modeling training complete. Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    app()
