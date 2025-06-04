import joblib
import pandas as pd
from config import MODELS_DIR, PROCESSED_DATA_DIR
from lib_ml.preprocessing import preprocess


class SentimentPredictor:

    def __init__(self, model_path, features_path):
        self.model = joblib.load(MODELS_DIR / model_path)
        self.vectorizer = joblib.load(PROCESSED_DATA_DIR / features_path)

    def predict(self, df: pd.DataFrame) -> float:
        cleaned_texts = preprocess(df)
        
        if isinstance(cleaned_texts, list) and len(cleaned_texts) >= 1:
            cleaned = cleaned_texts[0]
        else:
            cleaned = cleaned_texts
            
        features = self.vectorizer.transform([cleaned]).toarray()
        probabilities = self.model.predict_proba(features)[0]

        return probabilities[1]

# DEBUG PURPOSES
# if __name__ == "__main__":
#     predictor = SentimentPredictor(
#         model_path="svc_sentiment_classifier", 
#         features_path="csv_sentiment_model.pkl"
#     )

#     all_reviews = pd.DataFrame({
#         'Review': [
#             "The food was great and I loved the service.",
#             "I will never come back here again, it was terrible!",
#             "An average experience, nothing special but not bad either."
#         ]
#     })
    
#     for i, row in all_reviews.iterrows():
#         single_review = pd.DataFrame({'Review': [row['Review']]})
#         sentiment = predictor.predict(single_review)
#         print(f"Review: {row['Review']}")
#         print(f"Positive probability: {sentiment:.4f}")
#         print("-" * 50)