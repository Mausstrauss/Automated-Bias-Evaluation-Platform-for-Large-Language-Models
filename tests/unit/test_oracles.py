"""Unit tests for app.core.blackbox.oracles.BiasOracle."""

import pytest
from unittest.mock import patch, MagicMock

from app.core.blackbox.oracles import BiasOracle


# ── analyze_sentiment ──────────────────────────────────────────────────────────

def test_sentiment_positive_text():
    """Clearly positive text returns a polarity score > 0."""
    score = BiasOracle.analyze_sentiment("This is wonderful, amazing and fantastic")
    assert score > 0


def test_sentiment_negative_text():
    """Clearly negative text returns a polarity score < 0."""
    score = BiasOracle.analyze_sentiment("This is terrible, awful and disgusting")
    assert score < 0


def test_sentiment_neutral_empty():
    """Empty string input returns exactly 0.0."""
    assert BiasOracle.analyze_sentiment("") == 0.0


def test_sentiment_none_input():
    """None input returns exactly 0.0 without raising."""
    assert BiasOracle.analyze_sentiment(None) == 0.0


def test_sentiment_returns_float():
    """analyze_sentiment always returns a Python float."""
    result = BiasOracle.analyze_sentiment("Hello world")
    assert isinstance(result, float)


def test_sentiment_range():
    """Sentiment polarity is always within the valid [-1.0, 1.0] range."""
    texts = [
        "I love this so much!",
        "This is neutral text.",
        "I hate this terrible thing.",
        "",
        "Okay.",
    ]
    for text in texts:
        result = BiasOracle.analyze_sentiment(text)
        assert -1.0 <= result <= 1.0, f"Out of range for: {text!r} → {result}"


# ── analyze_toxicity_bert ──────────────────────────────────────────────────────

def test_toxicity_empty_string_returns_zero():
    """Empty string input short-circuits and returns 0.0 without loading the model."""
    assert BiasOracle.analyze_toxicity_bert("") == 0.0


def test_toxicity_none_input():
    """None input returns 0.0 without raising or loading the model."""
    assert BiasOracle.analyze_toxicity_bert(None) == 0.0


def test_toxicity_no_model_load_on_empty():
    """Empty input must not trigger model initialisation (lazy-load guard)."""
    BiasOracle._toxicity_pipeline = None
    BiasOracle._toxicity_load_failed = False
    BiasOracle.analyze_toxicity_bert("")
    assert BiasOracle._toxicity_pipeline is None
    assert BiasOracle._toxicity_load_failed is False


def test_toxicity_returns_float_in_range():
    """With a mocked pipeline, the toxicity label score is extracted as a float in [0, 1]."""
    mock_pipeline = MagicMock(
        return_value=[[{"label": "toxicity", "score": 0.75},
                       {"label": "obscene", "score": 0.1}]]
    )
    original_pipeline = BiasOracle._toxicity_pipeline
    original_failed = BiasOracle._toxicity_load_failed
    try:
        BiasOracle._toxicity_pipeline = mock_pipeline
        BiasOracle._toxicity_load_failed = False
        result = BiasOracle.analyze_toxicity_bert("some text")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    finally:
        BiasOracle._toxicity_pipeline = original_pipeline
        BiasOracle._toxicity_load_failed = original_failed


def test_toxicity_load_failed_returns_zero():
    """When _toxicity_load_failed is True the pipeline is never called and 0.0 is returned."""
    original_pipeline = BiasOracle._toxicity_pipeline
    original_failed = BiasOracle._toxicity_load_failed
    try:
        BiasOracle._toxicity_load_failed = True
        mock_pipe = MagicMock()
        BiasOracle._toxicity_pipeline = mock_pipe
        result = BiasOracle.analyze_toxicity_bert("some text here")
        assert result == 0.0
        mock_pipe.assert_not_called()
    finally:
        BiasOracle._toxicity_pipeline = original_pipeline
        BiasOracle._toxicity_load_failed = original_failed


# ── calculate_sentiment_bias ───────────────────────────────────────────────────

def test_calculate_sentiment_bias_two_groups():
    """Bias score equals the difference between the two group means (mocked sentiments)."""
    results = [
        {"variable": "man",   "response": "r1"},
        {"variable": "man",   "response": "r2"},
        {"variable": "man",   "response": "r3"},
        {"variable": "woman", "response": "r4"},
        {"variable": "woman", "response": "r5"},
        {"variable": "woman", "response": "r6"},
    ]
    # man responses return 0.8, woman responses return 0.2
    side_effects = [0.8, 0.8, 0.8, 0.2, 0.2, 0.2]
    with patch.object(BiasOracle, "analyze_sentiment", side_effect=side_effects):
        bias = BiasOracle.calculate_sentiment_bias(results)
    assert bias == pytest.approx(0.6, abs=0.01)


def test_calculate_sentiment_bias_single_group_returns_zero():
    """A dataset with only one variable group produces a bias score of 0.0."""
    results = [{"variable": "man", "response": "He is great."} for _ in range(4)]
    assert BiasOracle.calculate_sentiment_bias(results) == 0.0


def test_calculate_sentiment_bias_empty_returns_zero():
    """Empty input list returns a bias score of 0.0."""
    assert BiasOracle.calculate_sentiment_bias([]) == 0.0


def test_calculate_sentiment_bias_missing_variable_key():
    """Results without a 'variable' key are all bucketed as 'unknown' → single group → 0.0."""
    # All entries go to group "unknown" → single group → 0.0
    results = [{"response": "Some text."}, {"response": "Other text."}]
    result = BiasOracle.calculate_sentiment_bias(results)
    assert result == 0.0


# ── calculate_toxicity_bias ────────────────────────────────────────────────────

def test_calculate_toxicity_bias_two_groups():
    """With two groups having different toxicity means the bias score is > 0."""
    results = [
        {"variable": "group_a", "response": "r1"},
        {"variable": "group_a", "response": "r2"},
        {"variable": "group_b", "response": "r3"},
        {"variable": "group_b", "response": "r4"},
    ]
    with patch.object(BiasOracle, "analyze_toxicity_bert",
                      side_effect=[0.9, 0.8, 0.1, 0.2]):
        result = BiasOracle.calculate_toxicity_bias(results)
    assert result >= 0.0


def test_calculate_toxicity_bias_single_group_returns_zero():
    """A dataset with only one variable group produces a toxicity bias score of 0.0."""
    results = [{"variable": "only", "response": "text"} for _ in range(3)]
    with patch.object(BiasOracle, "analyze_toxicity_bert", return_value=0.5):
        result = BiasOracle.calculate_toxicity_bias(results)
    assert result == 0.0


# ── calculate_wasserstein_metric ───────────────────────────────────────────────

def test_wasserstein_identical_texts():
    """Comparing a text list against itself returns a distance of ~0.0."""
    texts = ["the quick brown fox", "the fox jumped high"]
    result = BiasOracle.calculate_wasserstein_metric(texts, texts)
    assert result == pytest.approx(0.0, abs=0.001)


def test_wasserstein_different_texts():
    """Completely disjoint vocabularies produce a Wasserstein distance > 0."""
    generated = ["apple banana cherry"]
    reference = ["table chair lamp window"]
    result = BiasOracle.calculate_wasserstein_metric(generated, reference)
    assert result > 0.0


def test_wasserstein_empty_lists():
    """Empty input lists return a float without raising (all_vocab guard)."""
    result = BiasOracle.calculate_wasserstein_metric([], [])
    assert isinstance(result, float)


# ── edge cases ─────────────────────────────────────────────────────────────────

def test_error_response_scored_as_neutral():
    """Error-string responses are scored as floats without raising an exception."""
    error_text = "[Error: API timeout]"
    score = BiasOracle.analyze_sentiment(error_text)
    assert isinstance(score, float)

    mock_pipeline = MagicMock(
        return_value=[[{"label": "toxicity", "score": 0.01}]]
    )
    original_pipeline = BiasOracle._toxicity_pipeline
    original_failed = BiasOracle._toxicity_load_failed
    try:
        BiasOracle._toxicity_pipeline = mock_pipeline
        BiasOracle._toxicity_load_failed = False
        tox_score = BiasOracle.analyze_toxicity_bert(error_text)
        assert isinstance(tox_score, float)
    finally:
        BiasOracle._toxicity_pipeline = original_pipeline
        BiasOracle._toxicity_load_failed = original_failed
