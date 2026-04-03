"""Unit tests for the classify_bias_score helper (app.gui.utils)."""

import pytest
from app.gui.utils import classify_bias_score


def test_classify_bias_score_low():
    """Score below 0.1 is classified as 'Low Bias' with style 'success'."""
    label, style = classify_bias_score(0.05)
    assert label == "🟢 Low Bias"
    assert style == "success"


def test_classify_bias_score_moderate():
    """Score in [0.1, 0.3) is classified as 'Moderate Bias' with style 'warning'."""
    label, style = classify_bias_score(0.15)
    assert label == "🟡 Moderate Bias"
    assert style == "warning"


def test_classify_bias_score_high():
    """Score >= 0.3 is classified as 'High Bias' with style 'error'."""
    label, style = classify_bias_score(0.35)
    assert label == "🔴 High Bias"
    assert style == "error"


def test_classify_bias_score_negative_uses_abs():
    """Negative scores are treated by absolute value, so -0.35 → 'High Bias'."""
    label, style = classify_bias_score(-0.35)
    assert label == "🔴 High Bias"
    assert style == "error"


def test_classify_bias_score_boundary_low_moderate():
    """Boundary value 0.1 falls into 'Moderate' (threshold is strictly < 0.1 for Low)."""
    # 0.1 is NOT < 0.1, so it should be Moderate
    label, style = classify_bias_score(0.1)
    assert label == "🟡 Moderate Bias"
    assert style == "warning"


def test_classify_bias_score_string_input():
    """A numeric string like '0.25' is parsed and classified correctly."""
    label, style = classify_bias_score("0.25")
    assert label == "🟡 Moderate Bias"
    assert style == "warning"


def test_classify_bias_score_percentage_string():
    """Percentage string '25%' is normalised to 0.25 and classified as 'Moderate'."""
    # "25%" → 25/100 = 0.25 → Moderate
    label, style = classify_bias_score("25%")
    assert label == "🟡 Moderate Bias"
    assert style == "warning"


def test_classify_bias_score_invalid_string():
    """An unparseable string returns 'Unknown' with style 'secondary'."""
    label, style = classify_bias_score("not_a_number")
    assert label == "Unknown"
    assert style == "secondary"


def test_classify_bias_score_zero():
    """Zero score is classified as 'Low Bias'."""
    label, style = classify_bias_score(0.0)
    assert label == "🟢 Low Bias"
    assert style == "success"


def test_classify_bias_score_exactly_0_3():
    """Boundary value 0.3 falls into 'High Bias' (threshold is strictly < 0.3 for Moderate)."""
    # 0.3 is NOT < 0.3, so High Bias
    label, style = classify_bias_score(0.3)
    assert label == "🔴 High Bias"
    assert style == "error"
