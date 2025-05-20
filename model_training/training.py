import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
import joblib

def train_model():
    """
    Trains the sentiment analysis model based on the Restaurant Sentiment
    Analysis project (https://github.com/proksch/restaurant-sentiment).

    The resulting model is stored in model_training/data/sentiment_model.pkl.
    """

    # When downloading tokenizers from NLTK, an error SSL certificate error occurs.
    # To prevent it, we disable SSL checks (solution by user vkosuri found at:
    # https://github.com/gunthercox/ChatterBot/issues/930#issuecomment-322111087).
    # Better would be to pre-download the tokenizers so that we do not rely on downloads
    # at runtime.
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    from lib_ml.preprocessing import preprocess_text

    dataset = pd.read_csv('model_training/data/a1_RestaurantReviews_HistoricDump.tsv', delimiter ='\t', quoting = 3)

    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = preprocess_text(review)  # Apply lib-ml
        corpus.append(review)

    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    bow_path = 'model_training/data/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, 'model_training/data/c2_Classifier_Sentiment_Model')
    joblib.dump(classifier, 'model_training/data/sentiment_model.pkl')

if __name__ == "__main__":
    train_model()
