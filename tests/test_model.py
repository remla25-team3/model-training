import pytest
import numpy as np
import pandas as pd


def test_nondeterminism_robustness(trained_model):
    """
    Test that the model is robust to nondeterminism (e.g., random seed).
    This test should involve running the same input through the model multiple times
    and ensuring the output remains consistent or within an acceptable range.
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
            f"Sentiment prediction was not consistent across runs for input: {input_text}"
        assert all(0 <= r <= 1 for r in results), \
            f"Predictions should be between 0 and 1 for input: {input_text}"
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


def test_sentiment_common_words(trained_model):
    """
    Test if the model understands the most common words that carry positive and negative sentiment.
    This test should use simple sentences with clear positive or negative sentiment words.
    """
    positive_examples = [
        "This is fantastic!",
        "I love this product.",
        "Great service and amazing quality.",
        "Excellent experience overall."
    ]
    
    negative_examples = [
        "This is terrible.",
        "I hate this product.",
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


def test_negation_sentiment_effect(trained_model):
    """
    Test the ability to correctly detect how negation in sentences affects sentiment.
    """
    negation_pairs = [
        ("I like this restaurant.", "I don't like this restaurant."),
        ("This is a good product.", "This is not a good product."),
        ("The service was excellent.", "The service was not excellent."),
        ("I enjoyed the meal.", "I did not enjoy the meal.")
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


def test_robustness_to_typos(trained_model):
    """
    Test if the model is not thrown off by simple typos like swapping two characters.
    This is a metamorphic test comparing sentiment of correctly spelled text with text containing typos.
    """
    typo_pairs = [
        ("This is a great product.", "This is a gerat product."),
        ("I love this movie.", "I lvoe this moive."),
        ("The food was delicious.", "The food was delicios."),
        ("Excellent service at this restaurant.", "Excelent servcie at this restuarant."),
        ("Absolutely fantastic experience.", "Absolutley fantastc experience."),
        ("I would recommend this place.", "I wuold recomend this place.")
    ]
    
    for correct, with_typo in typo_pairs:
        correct_df = pd.DataFrame({'Review': [correct]})
        typo_df = pd.DataFrame({'Review': [with_typo]})
        
        correct_sentiment = trained_model.predict(correct_df)
        typo_sentiment = trained_model.predict(typo_df)
        
        # Test 1: The sentiment scores should be approximately the same despite typos
        assert typo_sentiment == pytest.approx(correct_sentiment, abs=0.2), \
            f"Model should be robust to typos.\n" \
            f"Correct: '{correct}' -> {correct_sentiment}\n" \
            f"With typo: '{with_typo}' -> {typo_sentiment}"
        
        # Test 2: The classification (positive/negative) should remain consistent
        correct_positive = correct_sentiment > 0.5
        typo_positive = typo_sentiment > 0.5
        assert correct_positive == typo_positive, \
            f"Typos changed sentiment classification.\n" \
            f"Correct: '{correct}' -> {correct_sentiment} ({'positive' if correct_positive else 'negative'})\n" \
            f"With typo: '{with_typo}' -> {typo_sentiment} ({'positive' if typo_positive else 'negative'})"


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
        ("The food was awful.", "The food was awful. I posted a photo: https://food.pics/terrible-dish"),
        ("Would not recommend this place.", "Would not recommend this place. Read more: http://reviews.com/bad")
    ]
    
    for text_no_url, text_with_url in url_pairs:
        df_no_url = pd.DataFrame({'Review': [text_no_url]})
        df_with_url = pd.DataFrame({'Review': [text_with_url]})
        
        sentiment_no_url = trained_model.predict(df_no_url)
        sentiment_with_url = trained_model.predict(df_with_url)
        
        # The sentiment should be approximately the same with or without URL
        assert sentiment_with_url == pytest.approx(sentiment_no_url, abs=0.15), \
            f"URLs should not significantly affect sentiment scores.\n" \
            f"Without URL: '{text_no_url}' -> {sentiment_no_url}\n" \
            f"With URL: '{text_with_url}' -> {sentiment_with_url}"
        
        # Classification should remain the same
        classification_no_url = sentiment_no_url > 0.5  # True if positive
        classification_with_url = sentiment_with_url > 0.5
        assert classification_no_url == classification_with_url, \
            f"URLs changed sentiment classification.\n" \
            f"Without URL: '{text_no_url}' -> {sentiment_no_url} ({'positive' if classification_no_url else 'negative'})\n" \
            f"With URL: '{text_with_url}' -> {sentiment_with_url} ({'positive' if classification_with_url else 'negative'})"


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


def test_ordering_of_events_sentiment(trained_model):
    """
    Test if the model understands the order of events and its relevance on the sentiment of the sentence.
    It should detect statements about past and current opinions and report the current as the sentiment.
    """
    ordering_examples = [
        # Past negative, current positive (should be positive)
        (
            "I used to hate this airline, but now I like it.", 
            "I like this airline now, even though I used to hate it."
        ),
        (
            "The service was terrible before, but it has improved dramatically.", 
            "It has improved dramatically, though the service was terrible before."
        ),
        
        # Past positive, current negative (should be negative)
        (
            "I loved this product when I bought it, but now it doesn't work at all.", 
            "Now this product doesn't work at all, even though I loved it when I bought it."
        ),
        (
            "The restaurant was excellent last year, but the quality has declined significantly.", 
            "The quality has declined significantly, even though the restaurant was excellent last year."
        )
    ]
    
    for version1, version2 in ordering_examples:
        df1 = pd.DataFrame({'Review': [version1]})
        df2 = pd.DataFrame({'Review': [version2]})
        
        sentiment1 = trained_model.predict(df1)
        sentiment2 = trained_model.predict(df2)
        
        # The sentiment should be approximately the same regardless of clause ordering
        assert sentiment2 == pytest.approx(sentiment1, abs=0.15), \
            f"Different clause ordering should maintain similar sentiment.\n" \
            f"Version 1: '{version1}' -> {sentiment1}\n" \
            f"Version 2: '{version2}' -> {sentiment2}"
        
        # For the "past negative, current positive" examples (first two pairs)
        if "used to hate" in version1 or "was terrible before" in version1:
            assert sentiment1 > 0.5, f"Should be positive sentiment: '{version1}' -> {sentiment1}"
            assert sentiment2 > 0.5, f"Should be positive sentiment: '{version2}' -> {sentiment2}"
        # For the "past positive, current negative" examples (last two pairs)
        else:
            assert sentiment1 < 0.5, f"Should be negative sentiment: '{version1}' -> {sentiment1}"
            assert sentiment2 < 0.5, f"Should be negative sentiment: '{version2}' -> {sentiment2}"
