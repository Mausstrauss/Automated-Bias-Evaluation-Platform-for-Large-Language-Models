"""Unit tests for app.scheduler.scheduler."""

import json
import pytest
from unittest.mock import MagicMock, patch

import app.scheduler.scheduler as sched_module
from app.scheduler.scheduler import (
    load_config,
    run_daily_audit,
    MODEL_MAPPING,
)


# ── load_config ────────────────────────────────────────────────────────────────

def test_load_config_returns_defaults_when_no_file(monkeypatch):
    """load_config returns a valid default dict when the config file does not exist."""
    monkeypatch.setattr(sched_module, "CONFIG_PATH", "/nonexistent/scheduler_config.json")
    result = load_config()
    assert "active" in result
    assert "models" in result
    assert "metrics" in result


def test_load_config_reads_json_file(tmp_path, monkeypatch):
    """load_config parses a valid JSON file and returns its contents."""
    config_data = {
        "active": False,
        "models": ["Google Gemini Pro"],
        "metrics": ["Toxicity Check"],
    }
    config_file = tmp_path / "scheduler_config.json"
    config_file.write_text(json.dumps(config_data))
    monkeypatch.setattr(sched_module, "CONFIG_PATH", str(config_file))
    result = load_config()
    assert result["active"] is False
    assert result["models"] == ["Google Gemini Pro"]


def test_load_config_corrupt_json_returns_defaults(tmp_path, monkeypatch):
    """Malformed JSON does not raise; load_config falls back to the default dict."""
    config_file = tmp_path / "bad.json"
    config_file.write_text("{not valid json!!!")
    monkeypatch.setattr(sched_module, "CONFIG_PATH", str(config_file))
    result = load_config()
    # Should not crash and return default-shaped dict
    assert isinstance(result, dict)
    assert "active" in result


def test_load_config_missing_active_defaults_true(tmp_path, monkeypatch):
    """A config file without the 'active' key is treated as active=True by the caller."""
    config_data = {"models": ["OpenAI GPT 3.5"], "metrics": ["Sentiment Analysis"]}
    config_file = tmp_path / "partial.json"
    config_file.write_text(json.dumps(config_data))
    monkeypatch.setattr(sched_module, "CONFIG_PATH", str(config_file))
    result = load_config()
    # Key is absent from file — caller should be able to default to True
    assert result.get("active", True) is True


# ── run_daily_audit ────────────────────────────────────────────────────────────

def test_run_daily_audit_skips_when_inactive(monkeypatch):
    """run_daily_audit exits early without touching LLMGenerator when active=False."""
    monkeypatch.setattr(sched_module, "load_config",
                        lambda: {"active": False, "models": [], "metrics": []})
    mock_llm_cls = MagicMock()
    monkeypatch.setattr(sched_module, "LLMGenerator", mock_llm_cls)
    run_daily_audit()
    mock_llm_cls.assert_not_called()


def test_run_daily_audit_skips_when_no_template_file(monkeypatch):
    """run_daily_audit exits early without touching LLMGenerator when the template CSV is missing."""
    monkeypatch.setattr(sched_module, "load_config", lambda: {
        "active": True,
        "models": ["OpenAI GPT 3.5"],
        "metrics": ["Sentiment Analysis"],
    })
    import os as _os
    real_exists = _os.path.exists

    def mock_exists(path):
        if "langbite" in path:
            return False
        return real_exists(path)

    monkeypatch.setattr(sched_module.os.path, "exists", mock_exists)
    mock_llm_cls = MagicMock()
    monkeypatch.setattr(sched_module, "LLMGenerator", mock_llm_cls)
    run_daily_audit()
    mock_llm_cls.assert_not_called()


# ── MODEL_MAPPING ──────────────────────────────────────────────────────────────

def test_model_mapping_contains_expected_providers():
    """MODEL_MAPPING contains both expected display-name-to-provider-ID entries."""
    assert "OpenAI GPT 3.5" in MODEL_MAPPING
    assert "Google Gemini Pro" in MODEL_MAPPING
    assert MODEL_MAPPING["OpenAI GPT 3.5"] == "OpenAI-GPT3.5"
