"""Unit tests for app.core.visualization.BiasVisualizer."""

import pytest
import pandas as pd
import plotly.graph_objects as go

from app.core.visualization import BiasVisualizer


@pytest.fixture
def viz():
    return BiasVisualizer()


def _two_model_df():
    return pd.DataFrame({
        "Timestamp": [
            "2026-04-01 10:00:00", "2026-04-01 11:00:00",
            "2026-04-01 10:00:00", "2026-04-01 11:00:00",
        ],
        "Model":    ["GPT-4", "GPT-4",  "Gemini", "Gemini"],
        "Metric":   ["SentimentDiff", "Toxicity", "SentimentDiff", "Toxicity"],
        "Category": ["gender", "safety", "gender", "safety"],
        "Score":    [0.12, 0.05, 0.22, 0.08],
    })


# ── create_comparison_bar ──────────────────────────────────────────────────────

def test_comparison_bar_empty_returns_none(viz):
    """An empty DataFrame returns None instead of raising."""
    assert viz.create_comparison_bar(pd.DataFrame()) is None


def test_comparison_bar_returns_plotly_figure(viz, sample_history_df):
    """Valid history data produces a Plotly Figure object."""
    fig = viz.create_comparison_bar(sample_history_df)
    assert isinstance(fig, go.Figure)


def test_comparison_bar_uses_latest_per_group(viz):
    """Three timestamps for the same model+metric — only latest score should appear."""
    df = pd.DataFrame({
        "Timestamp": ["2026-04-01 10:00:00",
                      "2026-04-01 11:00:00",
                      "2026-04-01 12:00:00"],
        "Model":    ["GPT-4", "GPT-4", "GPT-4"],
        "Metric":   ["SentimentDiff"] * 3,
        "Category": ["gender"] * 3,
        "Score":    [0.1, 0.2, 0.3],
    })
    fig = viz.create_comparison_bar(df)
    assert fig is not None
    # One model, one metric → one trace with exactly one bar
    assert len(fig.data) == 1
    assert len(fig.data[0].y) == 1
    assert fig.data[0].y[0] == pytest.approx(0.3, abs=0.01)


def test_comparison_bar_timestamp_sorted_as_datetime(viz):
    """
    "Jan 10, 2026" > "Jan 2, 2026" by datetime but NOT by string sort
    ('Jan 10' < 'Jan 2' lexicographically).
    The chart must use the datetime-latest score (0.9).
    """
    df = pd.DataFrame({
        "Timestamp": ["Jan 10, 2026", "Jan 2, 2026"],
        "Model":    ["GPT-4", "GPT-4"],
        "Metric":   ["SentimentDiff", "SentimentDiff"],
        "Category": ["gender", "gender"],
        "Score":    [0.9, 0.1],   # Jan 10 is the true latest → should show 0.9
    })
    fig = viz.create_comparison_bar(df)
    assert fig is not None
    all_y = [v for trace in fig.data for v in trace.y]
    assert any(abs(v - 0.9) < 0.01 for v in all_y), \
        f"Expected datetime-latest score 0.9 in chart, got {all_y}"


# ── create_heatmap ─────────────────────────────────────────────────────────────

def test_heatmap_empty_returns_none(viz):
    """An empty DataFrame returns None instead of raising."""
    assert viz.create_heatmap(pd.DataFrame()) is None


def test_heatmap_returns_plotly_figure(viz, sample_history_df):
    """Valid history data produces a Plotly Figure object."""
    fig = viz.create_heatmap(sample_history_df)
    assert isinstance(fig, go.Figure)


def test_heatmap_pivot_correct_shape(viz):
    """Heatmap axes contain all expected model names and metric labels."""
    df = _two_model_df()
    fig = viz.create_heatmap(df)
    assert fig is not None
    # Heatmap trace: x = metrics, y = models
    x_vals = set(fig.data[0].x)
    y_vals = set(fig.data[0].y)
    assert "SentimentDiff" in x_vals
    assert "Toxicity" in x_vals
    assert "GPT-4" in y_vals
    assert "Gemini" in y_vals


# ── create_trend_line ──────────────────────────────────────────────────────────

def test_trend_line_empty_returns_none(viz):
    """An empty DataFrame returns None instead of raising."""
    assert viz.create_trend_line(pd.DataFrame()) is None


def test_trend_line_returns_plotly_figure(viz, sample_history_df):
    """Valid history data produces a Plotly Figure object."""
    fig = viz.create_trend_line(sample_history_df)
    assert isinstance(fig, go.Figure)


# ── mutation guard ─────────────────────────────────────────────────────────────

def test_copy_prevents_mutation(viz, sample_history_df):
    """create_comparison_bar must not mutate the caller's DataFrame."""
    df = sample_history_df.copy()
    original_dtype = df["Timestamp"].dtype
    viz.create_comparison_bar(df)
    assert df["Timestamp"].dtype == original_dtype, \
        "Timestamp column dtype was mutated in the caller's DataFrame"
