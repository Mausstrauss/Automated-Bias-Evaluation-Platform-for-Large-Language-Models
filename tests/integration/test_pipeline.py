"""Integration tests — full pipeline smoke tests without real LLM calls."""

import json
import pytest
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock, ANY

from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.blackbox.oracles import BiasOracle
from app.core.aggregation import BiasAggregator
from app.core.visualization import BiasVisualizer
import app.scheduler.scheduler as sched_module
from app.scheduler.scheduler import run_daily_audit


_REAL_CSV = "data/word_lists/prompts/langbite_templates.csv"


def _fake_batch_two_groups(prompts):
    """Return fake LLM responses that mirror the variable field of each prompt."""
    results = []
    for p in prompts:
        entry = dict(p)
        var = p.get("variable", "unknown")
        if var == "man":
            entry["response"] = "He is strong and confident."
        elif var == "woman":
            entry["response"] = "She is emotional and hesitant."
        else:
            entry["response"] = "Neutral text about this topic."
        results.append(entry)
    return results


# ── full pipeline ──────────────────────────────────────────────────────────────

def test_full_pipeline_gender_templates(tmp_path):
    """End-to-end smoke test: load templates → fake LLM → score → aggregate → visualise."""
    prompts = load_and_expand_templates(_REAL_CSV)
    assert len(prompts) >= 2

    fake_results = _fake_batch_two_groups(prompts)

    bias_score = BiasOracle.calculate_sentiment_bias(fake_results)
    assert isinstance(bias_score, float)

    agg = BiasAggregator(output_dir=str(tmp_path / "output"))
    agg.add_result("TestModel", "SentimentDiff", "gender", bias_score)
    agg.save_to_history()

    df = agg.get_history_df()
    assert df.shape == (1, 5)
    assert set(df.columns) == {"Timestamp", "Model", "Metric", "Category", "Score"}

    viz = BiasVisualizer()
    fig = viz.create_comparison_bar(df)
    assert isinstance(fig, go.Figure)


def test_full_pipeline_empty_responses_handled(tmp_path):
    """Simulated error responses do not raise; pipeline completes and produces float scores."""
    prompts = load_and_expand_templates(_REAL_CSV)
    error_results = [dict(p, response="[Error: timeout]") for p in prompts]

    try:
        bias = BiasOracle.calculate_sentiment_bias(error_results)
        assert isinstance(bias, float)
        agg = BiasAggregator(output_dir=str(tmp_path / "output"))
        agg.add_result("M", "SentimentDiff", "gender", bias)
        agg.save_to_history()
    except Exception as exc:
        pytest.fail(f"Pipeline raised an unexpected exception: {exc}")


def test_full_pipeline_single_group_produces_zero_bias(tmp_path):
    """When all prompts share the same variable group, both sentiment and toxicity bias equal 0."""
    fake_results = [
        {"variable": "unknown", "response": "Some neutral text."},
        {"variable": "unknown", "response": "Another piece of text."},
        {"variable": "unknown", "response": "More content here."},
    ]
    sent_bias = BiasOracle.calculate_sentiment_bias(fake_results)
    tox_bias = BiasOracle.calculate_toxicity_bias(fake_results)
    assert sent_bias == 0.0
    assert tox_bias == 0.0


def test_aggregator_visualizer_integration(tmp_path):
    """Multiple runs accumulate in history; all three visualiser methods return Figure objects."""
    agg = BiasAggregator(output_dir=str(tmp_path / "output"))

    rows = [
        ("GPT-4",  "SentimentDiff", "gender", 0.12),
        ("GPT-4",  "SentimentDiff", "gender", 0.14),
        ("GPT-4",  "Toxicity",      "safety", 0.05),
        ("Gemini", "SentimentDiff", "gender", 0.22),
        ("Gemini", "Toxicity",      "safety", 0.08),
        ("Gemini", "Toxicity",      "safety", 0.10),
    ]
    for model, metric, cat, score in rows:
        agg.add_result(model, metric, cat, score)

    agg.save_to_history()
    df = agg.get_history_df()
    assert len(df) == 6

    viz = BiasVisualizer()
    assert isinstance(viz.create_comparison_bar(df), go.Figure)
    assert isinstance(viz.create_heatmap(df),         go.Figure)
    assert isinstance(viz.create_trend_line(df),       go.Figure)


# ── scheduler reads config ─────────────────────────────────────────────────────

def test_scheduler_config_read_by_audit(tmp_path, monkeypatch):
    """run_daily_audit reads scheduler_config.json and uses the configured provider."""
    config = {
        "active": True,
        "models": ["Google Gemini Pro"],
        "metrics": ["Sentiment Analysis"],
    }
    config_file = tmp_path / "scheduler_config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(sched_module, "CONFIG_PATH", str(config_file))

    # Mock LLMGenerator
    mock_llm_cls = MagicMock()
    mock_instance = MagicMock()
    mock_llm_cls.return_value = mock_instance
    mock_instance.generate_batch.return_value = [
        {"variable": "man",   "response": "He is great."},
        {"variable": "woman", "response": "She is capable."},
    ]
    monkeypatch.setattr(sched_module, "LLMGenerator", mock_llm_cls)

    # Mock BiasOracle to avoid real scoring
    monkeypatch.setattr(sched_module.BiasOracle, "calculate_sentiment_bias",
                        staticmethod(lambda r: 0.1))

    # Mock aggregator and file I/O
    mock_agg_cls = MagicMock()
    monkeypatch.setattr(sched_module, "BiasAggregator", mock_agg_cls)
    monkeypatch.setattr(sched_module.os, "makedirs", lambda *a, **kw: None)

    with patch("pandas.DataFrame.to_csv"):
        run_daily_audit()

    # LLMGenerator must have been called with the Gemini provider (from config)
    assert mock_llm_cls.called, "LLMGenerator was never instantiated"
    call_kwargs = mock_llm_cls.call_args
    assert call_kwargs.kwargs.get("provider") == "Google-Gemini" or \
           call_kwargs.args[0] == "Google-Gemini", \
        f"Expected provider='Google-Gemini', got: {call_kwargs}"
