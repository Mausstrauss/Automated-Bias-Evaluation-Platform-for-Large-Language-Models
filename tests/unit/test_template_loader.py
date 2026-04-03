"""Unit tests for app.core.blackbox.template_loader."""

import pytest
from unittest.mock import patch
import pandas as pd

from app.core.blackbox.template_loader import (
    load_and_expand_templates,
    load_real_toxicity_prompts,
    load_bold_prompts,
)

REQUIRED_KEYS = {"prompt", "variable", "group", "source"}


# ── load_and_expand_templates ──────────────────────────────────────────────────

def test_load_valid_csv(temp_csv_path):
    """Valid CSV with 2 templates expands to prompts with all required keys."""
    prompts = load_and_expand_templates(temp_csv_path)
    assert len(prompts) > 0
    for entry in prompts:
        assert REQUIRED_KEYS.issubset(entry.keys()), f"Missing keys in {entry}"


def test_expand_produces_multiple_groups(temp_csv_path):
    """Expanding templates with pipe-separated values produces ≥ 2 distinct variable groups."""
    prompts = load_and_expand_templates(temp_csv_path)
    distinct = {p["variable"] for p in prompts}
    assert len(distinct) >= 2


def test_missing_file_returns_empty_list():
    """A non-existent CSV path returns an empty list without raising."""
    result = load_and_expand_templates("/nonexistent/path.csv")
    assert result == []


def test_nan_values_row_skipped(tmp_path):
    """Rows with NaN/empty values field are skipped; valid rows still expand."""
    csv_file = tmp_path / "nan_test.csv"
    csv_file.write_text(
        'template,variable_type,values\n'
        '"The <gender> is...",gender,"man|woman"\n'
        '"Another <x> template",x,\n'  # empty values → NaN
    )
    prompts = load_and_expand_templates(str(csv_file))
    # Only the first row with valid values should be expanded
    assert len(prompts) == 2
    for p in prompts:
        assert p["group"] == "gender"


def test_placeholder_fully_substituted(temp_csv_path):
    """No expanded prompt should contain an unresolved <placeholder> token."""
    prompts = load_and_expand_templates(temp_csv_path)
    for p in prompts:
        assert "<" not in p["prompt"], f"Unreplaced placeholder found: {p['prompt']}"


def test_multiple_values_expand_correctly(tmp_path):
    """A template with 3 pipe-separated values expands to exactly 3 prompts."""
    csv_file = tmp_path / "multi.csv"
    csv_file.write_text(
        'template,variable_type,values\n'
        '"Hello <name>.",name,"a|b|c"\n'
    )
    prompts = load_and_expand_templates(str(csv_file))
    assert len(prompts) == 3
    variables = {p["variable"] for p in prompts}
    assert variables == {"a", "b", "c"}


def test_source_field_is_csv_template(temp_csv_path):
    """Every expanded prompt from a CSV template has source='csv_template'."""
    prompts = load_and_expand_templates(temp_csv_path)
    for p in prompts:
        assert p["source"] == "csv_template"


# ── load_real_toxicity_prompts fallback ────────────────────────────────────────

def test_load_real_toxicity_fallback():
    """When HuggingFace is unavailable, the fallback CSV is used and all keys are present."""
    with patch("app.core.blackbox.template_loader.load_dataset",
               side_effect=Exception("no network")):
        result = load_real_toxicity_prompts(limit=10)
    assert len(result) > 0
    for item in result:
        assert "prompt" in item
        assert "variable" in item
        assert "group" in item
        assert "source" in item


# ── load_bold_prompts fallback ─────────────────────────────────────────────────

def test_load_bold_fallback():
    """When HuggingFace is unavailable, the BOLD fallback CSV returns the requested limit."""
    with patch("app.core.blackbox.template_loader.load_dataset",
               side_effect=Exception("no network")):
        result = load_bold_prompts(limit=10)
    assert len(result) == 10
    for item in result:
        assert "variable" in item
        assert "group" in item


def test_bold_fallback_csv_used_when_hf_fails():
    """Fallback-sourced BOLD prompts are tagged with source='BOLD-Fallback'."""
    with patch("app.core.blackbox.template_loader.load_dataset",
               side_effect=Exception("no network")):
        result = load_bold_prompts(limit=5)
    assert len(result) > 0
    for item in result:
        assert item.get("source") == "BOLD-Fallback"
