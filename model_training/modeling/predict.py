import joblib
import pandas as pd
from model_training.config import MODELS_DIR
from lib_ml.preprocessing import preprocess_text

class SentimentPredictor:

    def __init__(self):

        self.model = joblib.load(MODELS_DIR / "sentiment_model.pkl")
        self.vectorizer = joblib.load(MODELS_DIR / "bow_sentiment_model.pkl")

    #simple function to predict sentiment of a given text
    def predict(self, text: str) -> str:

        cleaned = preprocess_text(text)
        features = self.vectorizer.transform([cleaned]).toarray()
        prediction = self.model.predict(features)[0]
        return "Positive" if prediction == 1 else "Negative"

if __name__ == "__main__":

    predictor = SentimentPredictor()
    #simple test case
    sample = "The food was awful and I hated the experience."

    print(f"Review: {sample}")
    print(f"Predicted Sentiment: {predictor.predict(sample)}")