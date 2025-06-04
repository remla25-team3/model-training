import numpy as np
import pandas as pd
import pytest

from model_training.modeling.train import train_model
from model_training.dataset import preprocess_dataset


@pytest.mark.development
def test_nondeterminism_robustness(trained_model):
    """
    Test that the model is robust to nondeterminism (e.g., random seed).
    Run the same input multiple times and ensure output stays consistent.
    """
    input_texts = [
        "I'm neutral about it.",  # Neutral
        "I hate it!",            # Strong negative
        "I love it!",            # Strong positive
        "This is a very long text that should still produce consistent results. " * 10,  # Long text
        "Product #123 is great!",  # Text with numbers and symbols
        "The food was awful and I loved the experience.",  # Mixed sentiment
    ]

    for input_text in input_texts:
        results = []
        for _ in range(10):
            input_df = pd.DataFrame({'Review': [input_text]})
            result = trained_model.predict(input_df)
            results.append(result)
        
        assert all(r == pytest.approx(results[0], abs=0.01) for r in results), \
            f"Prediction inconsistent for input: {input_text}"
        assert all(0 <= r <= 1 for r in results), \
            f"Predictions out of [0,1] for input: {input_text}"
        std_dev = np.std(results)
        assert std_dev < 0.001, \
            f"Standard deviation of predictions ({std_dev}) is too high for input: {input_text}"

    with pytest.raises(Exception):
        trained_model.predict(None)
    with pytest.raises(Exception):
        trained_model.predict(pd.DataFrame())
        
    empty_df = pd.DataFrame({'Review': ['']})
    empty_result = trained_model.predict(empty_df)

    assert 0 <= empty_result <= 1, "Empty string prediction should still give result between 0 and 1"


@pytest.mark.development
def test_sentiment_common_words(trained_model):
    """
    Test if the model understands the most common words that carry positive and negative sentiment.
    This test should use simple sentences with clear positive or negative sentiment words.
    """
    positive_examples = [
        "This is fantastic!",
        "I love this product.",
        "Great service and amazing quality.",
        "Amazing experience overall."
    ]
    
    negative_examples = [
        "This is terrible.",
        "Did not like it.",
        "Poor service and awful quality.",
        "Disappointing experience overall."
    ]
    
    for text in positive_examples:
        input_df = pd.DataFrame({'Review': [text]})
        sentiment = trained_model.predict(input_df)
        assert sentiment > 0.5, f"Positive text '{text}' should have probability > 0.5, got {sentiment}"
    
    for text in negative_examples:
        input_df = pd.DataFrame({'Review': [text]})
        sentiment = trained_model.predict(input_df)
        assert sentiment < 0.5, f"Negative text '{text}' should have probability < 0.5, got {sentiment}"


@pytest.mark.development
def test_negation_sentiment_effect(trained_model):
    """
    Test the ability to correctly detect how negation in sentences affects sentiment.
    """
    negation_pairs = [
        ("I like this restaurant.", "I do not like this restaurant."),
        ("This is a good product.", "This is not a good product."),
        ("The service was good.", "The service was not good."),
        ("I liked the meal.", "I did not like the meal.")
    ]
    
    for positive, negative in negation_pairs:
        pos_df = pd.DataFrame({'Review': [positive]})
        neg_df = pd.DataFrame({'Review': [negative]})
        
        pos_sentiment = trained_model.predict(pos_df)
        neg_sentiment = trained_model.predict(neg_df)
        
        assert pos_sentiment > neg_sentiment, \
            f"Positive '{positive}' ({pos_sentiment}) should have higher probability than " \
            f"negative '{negative}' ({neg_sentiment})"
        
        # For stronger assertion, positive should be > 0.5 and negative < 0.5
        assert pos_sentiment > 0.5, f"Positive '{positive}' should have probability > 0.5"
        assert neg_sentiment < 0.5, f"Negative '{negative}' should have probability < 0.5"


@pytest.mark.development
def test_robustness_to_typos(trained_model):
    """
    Test if the model is not thrown off by simple typos like swapping two characters.
    This is a metamorphic test comparing sentiment of correctly spelled text with text containing typos.
    """
    typo_pairs = [
        ("This is a great product.", "This is a gerat product."),
        ("I love this movie.", "I lvoe this moive."),
        ("The food was delicious.", "The fooood was delicios."),
        ("Excellent service at this restaurant.", "Excelent servcie at this restuarant."),
        ("bad service.", "baad service"),
        ("I would recommend this place.", "I wuold recomend this pllace.")
    ]
    
    for correct, with_typo in typo_pairs:
        correct_df = pd.DataFrame({'Review': [correct]})
        typo_df = pd.DataFrame({'Review': [with_typo]})
        
        correct_sentiment = trained_model.predict(correct_df)
        typo_sentiment = trained_model.predict(typo_df)
        
        # The classification (positive/negative) should remain consistent
        correct_positive = correct_sentiment > 0.5
        typo_positive = typo_sentiment > 0.5
        assert correct_positive == typo_positive, \
            f"Typos changed sentiment classification.\n" \
            f"Correct: '{correct}' -> {correct_sentiment} ({'positive' if correct_positive else 'negative'})\n" \
            f"With typo: '{with_typo}' -> {typo_sentiment} ({'positive' if typo_positive else 'negative'})"


@pytest.mark.development
def test_irrelevance_ignoring_urls(trained_model):
    """
    Test if the model correctly ignores irrelevant parts of sentences, such as URLs.
    The presence of a URL should not affect the core understanding or sentiment.
    This is a metamorphic test comparing sentiment with and without URLs.
    """
    url_pairs = [
        # Positive examples
        ("This is amazing!", "This is amazing! Check it out: https://example.com/awesome"),
        ("Great experience today.", "Great experience today. See details at http://mysite.org/review"),
        ("I love this restaurant.", "I love this restaurant. Visit their website www.restaurant.com"),
        
        # Negative examples
        ("I had a terrible experience.", "I had a terrible experience. More info at example.net"),
        ("The food was bad.", "The food was bad. Photo here: https://food.pics/terrible-dish"),
        ("Would not recommend this place.", "Would not recommend this place. Read more: http://reviews.com/bad")
    ]
    
    for text_no_url, text_with_url in url_pairs:
        df_no_url = pd.DataFrame({'Review': [text_no_url]})
        df_with_url = pd.DataFrame({'Review': [text_with_url]})
        
        sentiment_no_url = trained_model.predict(df_no_url)
        sentiment_with_url = trained_model.predict(df_with_url)
        
        # Classification should remain the same
        classification_no_url = sentiment_no_url > 0.5  # True if positive
        classification_with_url = sentiment_with_url > 0.5
        assert classification_no_url == classification_with_url, \
            f"URLs changed sentiment classification.\n" \
            f"Without URL: '{text_no_url}' -> {sentiment_no_url} ({'positive' if classification_no_url else 'negative'})\n" \
            f"With URL: '{text_with_url}' -> {sentiment_with_url} ({'positive' if classification_with_url else 'negative'})"

@pytest.mark.development
def test_named_entities_sentiment_invariance(trained_model):
    """
    Test if the model correctly identifies and handles named entities, and
    if switching two locations or people does not affect the sentiment of the overall sentence.
    This is a metamorphic test comparing sentiment when named entities are swapped.
    """
    entity_pairs = [
        # Location pairs
        ("I had a great time in New York.", "I had a great time in San Francisco."),
        ("The service at McDonald's was terrible.", "The service at Burger King was terrible."),
        ("We enjoyed our stay at Hilton Hotel.", "We enjoyed our stay at Marriott Hotel."),
        
        # Person name pairs
        ("John provided excellent customer service.", "Mary provided excellent customer service."),
        ("I was disappointed by Robert's attitude.", "I was disappointed by Sarah's attitude."),
        ("Alex helped us find the perfect solution.", "Emma helped us find the perfect solution.")
    ]
    
    for text1, text2 in entity_pairs:
        df1 = pd.DataFrame({'Review': [text1]})
        df2 = pd.DataFrame({'Review': [text2]})
        
        sentiment1 = trained_model.predict(df1)
        sentiment2 = trained_model.predict(df2)
        
        # The sentiment should be approximately the same when only named entities change
        assert sentiment2 == pytest.approx(sentiment1, abs=0.15), \
            f"Named entities should not significantly affect sentiment scores.\n" \
            f"Text 1: '{text1}' -> {sentiment1}\n" \
            f"Text 2: '{text2}' -> {sentiment2}"
        
        # Classification should remain the same
        classification1 = sentiment1 > 0.5
        classification2 = sentiment2 > 0.5
        assert classification1 == classification2, \
            f"Named entity change affected sentiment classification.\n" \
            f"Text 1: '{text1}' -> {sentiment1} ({'positive' if classification1 else 'negative'})\n" \
            f"Text 2: '{text2}' -> {sentiment2} ({'positive' if classification2 else 'negative'})"


# CONSIDERATIONS OF INCLUSION / FAIRNESS

@pytest.mark.development
def test_fairness_gendered_language(trained_model):
    """
    Test if the model does not associate sentiment just based on gendered parts of sentences.
    This is a fairness test to ensure the model isn't biased based on gender references.
    """
    gender_pairs = [
        # Positive sentiment pairs
        ("He is a brilliant engineer.", "She is a brilliant engineer."),
        ("The man gave excellent customer service.", "The woman gave excellent customer service."),
        ("My father cooked a delicious meal.", "My mother cooked a delicious meal."),
        
        # Negative sentiment pairs
        ("He was very rude to the staff.", "She was very rude to the staff."),
        ("The male customer complained about everything.", "The female customer complained about everything."),
        ("My brother damaged the rental car.", "My sister damaged the rental car.")
    ]
    
    for male_text, female_text in gender_pairs:
        male_df = pd.DataFrame({'Review': [male_text]})
        female_df = pd.DataFrame({'Review': [female_text]})
        
        male_sentiment = trained_model.predict(male_df)
        female_sentiment = trained_model.predict(female_df)
        
        # The sentiment should be approximately the same regardless of gender reference
        assert female_sentiment == pytest.approx(male_sentiment, abs=0.1), \
            f"Gender references should not affect sentiment scores.\n" \
            f"Male: '{male_text}' -> {male_sentiment}\n" \
            f"Female: '{female_text}' -> {female_sentiment}"
        
        # Classification should remain the same
        male_positive = male_sentiment > 0.5
        female_positive = female_sentiment > 0.5
        assert male_positive == female_positive, \
            f"Gender reference changed sentiment classification.\n" \
            f"Male: '{male_text}' -> {male_sentiment} ({'positive' if male_positive else 'negative'})\n" \
            f"Female: '{female_text}' -> {female_sentiment} ({'positive' if female_positive else 'negative'})"


# MODEL QUALITY ON IMPORTANT SLICES (some covered in test_data_slices.py)
# (No additional code here, since slices are in test_data_slices.py)
