""""
Metamorphic tests to verify that the sentiment model behaves consistently
under semantic-preserving transformations and compositional modifications.

Covered capabilities and slice types:
- Whitespace and punctuation invariance
- Synonym substitution (taxonomy)
- Monotonicity with sentiment-enhancing/weakening phrases
- Negation reversal
- Robustness to ordering of independent facts
- Generalization-specialization consistency
- Vocabulary structure (e.g., POS-role stability)
- Repair-based robustness (e.g., noise cleanup for corrupted input)

Each test checks if the model is semantically robust across these perturbations and
maintains consistent sentiment predictions across functionally equivalent inputs.
"""


from pathlib import Path
import random
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib

from model_training.config import MODELS_DIR
from model_training.dataset import preprocess_dataset

VECTORIZER_PATH = MODELS_DIR / "bow_sentiment_model.pkl"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"

model = joblib.load(MODEL_PATH)
cv = joblib.load(VECTORIZER_PATH)


def wrap(text: str) -> pd.DataFrame:
    """
    Wrap a single review string into a one-row DataFrame compatible with the model's input format.
    """
    return pd.DataFrame({"Review": [text]})


def call_predict_single(texts):
    """
    Preprocess and predict sentiment scores for a list of input texts.
    """
    preds = []
    for t in texts:
        df = pd.DataFrame({'Review': [t], 'Liked': [0]})  # Dummy label
        corpus, _ = preprocess_dataset(df)
        features = cv.transform(corpus).toarray()
        df_feat = pd.DataFrame(features, columns=cv.get_feature_names_out())
        pred = model.predict(df_feat)
        preds.append(float(pred[0]))
    return preds


@pytest.mark.development
def test_model_on_short_reviews():
    """
    Test model predictions on short single-word reviews.
    """
    examples = ["Good", "Bad", "Tasty", "Awful"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, float) for p in preds), "Short reviews failed"


@pytest.mark.development
def test_model_on_long_reviews():
    """
    Test model predictions on long repeated text reviews.
    """
    long_text = "The service was wonderful and the ambiance was perfect. " * 10
    examples = [long_text, long_text + " Loved it!"]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Long reviews failed"


@pytest.mark.development
def test_model_on_named_entities():
    """
    Test model robustness when named entities (e.g., names) are present.
    """
    examples = ["John loved the pizza.", "Anna did not like the sushi."]
    preds = call_predict_single(examples)
    assert all(isinstance(p, (int, float)) for p in preds), "Named entities caused failure"


@pytest.mark.development
def test_model_on_negation_slice():
    """
    Test whether the model assigns low sentiment to negated expressions.
    """
    examples = ["I do not like this.", "Bad experience."]
    preds = call_predict_single(examples)
    assert all(p < 0.5 for p in preds), "Negation not handled correctly"


@pytest.mark.metamorphic
def test_invariance_whitespace_and_punctuation(trained_model):
    """
    Metamorphic Relation: Adding or removing non-meaningful whitespace or punctuation
    should not change the model's core output (e.g., sentiment, entity extraction).
    """
    variants = [
        "This is a great movie!",
        "This is a great movie.",  # Punctuation change
        "  This   is a great  movie!  "  # Whitespace change
    ]

    preds = [
        trained_model.predict(wrap(text))
        for text in variants
    ]

    for i in range(1, len(preds)):
        assert abs(preds[0] - preds[i]) <= 0.05, f"Predictions differ too much: {preds}"


@pytest.mark.metamorphic
def test_invariance_synonym_substitution(trained_model):
    """
    Metamorphic Relation: Replacing words with synonyms that do not alter the
    overall meaning or sentiment of the sentence should result in similar output.
    This assumes your model should generalize across vocabulary.
    """

    test_pairs = [
        # POSITIVE examples
        ("The food was fantastic.", "The food was amazing."),
        ("This restaurant is great.", "This restaurant is excellent."),
        ("I enjoyed the service.", "I loved the service."),
        # NEGATIVE examples
        ("The meal was terrible.", "The meal was so bad."),
        ("I really hate the atmosphere.", "I despise the atmosphere."),
        ("It was a bad experience.", "It was a terrible experience."),
    ]

    for original, synonym_variant in test_pairs:
        p1 = trained_model.predict(wrap(original))
        p2 = trained_model.predict(wrap(synonym_variant))

        assert p1 == pytest.approx(p2, abs=0.3), (
            f"Sentiment diverges too much: \n"
            f"  original='{original}' ({p1: .3f})\n"
            f"  synonym='{synonym_variant}' ({p2: .3f})"
        )


@pytest.mark.metamorphic
def test_monotonicity_sentiment_addition(trained_model):
    """
    Metamorphic Relation: Adding words or phrases that clearly increase/decrease
    a specific property (e.g., positivity/negativity of sentiment) should lead
    to a monotonic change in the model's output for that property.
    """
    base = "This is an average movie."
    positive = "This is an average movie, but the ending is amazing."
    negative = "This is an average movie, but the ending is terrible."

    p_base = trained_model.predict(wrap(base))
    p_pos = trained_model.predict(wrap(positive))
    p_neg = trained_model.predict(wrap(negative))

    assert p_pos > p_base, f"Positive addition failed: {p_pos} <= {p_base}"
    assert p_neg < p_base, f"Negative addition failed: {p_neg} >= {p_base}"


@pytest.mark.metamorphic
def test_reversal_negation(trained_model):
    """
    Metamorphic Relation: Applying negation to a statement should reverse its
    sentiment or other relevant properties (e.g., from positive to negative, or vice-versa).
    """
    pos = "I love this product."
    neg = "I do not love this product."

    p_pos = trained_model.predict(wrap(pos))
    p_neg = trained_model.predict(wrap(neg))

    assert p_pos > 0.7, f"Expected positive sentiment: {p_pos}"
    assert p_neg < 0.3, f"Expected negated sentiment: {p_neg}"


@pytest.mark.metamorphic
def test_permutation_order_independent_features(trained_model):
    """
    Metamorphic Relation: If the order of certain independent elements (e.g., items in a list,
    facts in a non-sequential summary) should not affect the model's output.
    """
    base = "Feature A: Yes. Feature B: No. Feature C: Yes."
    permuted = ["Feature A: Yes.", "Feature B: No.", "Feature C: Yes."]
    random.shuffle(permuted)
    permuted_text = " ".join(permuted)

    p_base = trained_model.predict(wrap(base))
    p_perm = trained_model.predict(wrap(permuted_text))

    assert abs(p_base - p_perm) <= 0.1, f"Order changed result too much: {p_base} vs {p_perm}"


@pytest.mark.metamorphic
def test_generalization_specialization(trained_model):
    """
    Metamorphic Relation: A more general statement should have a consistent relation
    to a more specific statement. For instance, if a general statement is positive,
    a specific instance of it should also be positive (or stronger positive).
    """
    general = "A large dog is running."
    specific = "A German Shepherd is running."

    p_general = trained_model.predict(wrap(general))
    p_specific = trained_model.predict(wrap(specific))

    assert p_specific >= p_general - 0.1, f"Specific case lost positivity: {p_specific} vs {p_general}"


# mutamorphic testing with automatic inconsistency repair
@pytest.mark.metamorphic
def test_whitespace_invariance_with_repair(trained_model):
    """
    Metamorphic Relation with Repair: If a whitespace/punctuation variant produces
    an inconsistent prediction, automatically repair the variant (by stripping
    extraneous characters) and re-check consistency against the base prediction.

    To simulate the need for repair (even though our pipeline already strips
    ASCII punctuation), we add an em-dash (—) which is often not removed by simple
    ASCII-only cleaning logic. This guarantees that the “broken” variant
    will not match the base under your normal preprocess.
    """

    base = "The movie was good"
    broken = "The m—o—vi—e  was g—o—o—d—"

    p_base = trained_model.predict(wrap(base))
    p_broken = trained_model.predict(wrap(broken))

    # We expect that, because of the em-dash, the raw “broken” input will differ
    # by more than 0.1 from the base. If it does not, we fail the test, because
    # we wanted to force the repair path.
    assert abs(p_base - p_broken) > 0.1, (
        f"Expected broken variant to deviate: got base={p_base: .3f}, broken={p_broken: .3f}"
    )

    # Now run our “repair” step: strip anything that is not A-Z, a-z, 0-9 or space,
    # and then collapse multiple spaces. This should turn “The m—o—vi—e was g—o—o—d—”
    # into exactly “The movie was good”.
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', '', broken)      # remove em-dash
    repaired_text = " ".join(cleaned.split())            # collapse any extra spaces

    # Confirm that our repair really did match the base string exactly:
    assert repaired_text == base, (
        f"Repair step failed to produce the base text: \n"
        f"  repaired_text ='{repaired_text}' vs base ='{base}'"
    )

    # predict on the repaired text and assert it is consistent with p_base
    p_repaired = trained_model.predict(wrap(repaired_text))
    assert abs(p_base - p_repaired) <= 0.05, (
        f"Automatic repair did not restore consistency: \n"
        f"  base={p_base: .3f}, repaired={p_repaired: .3f}"
    )
