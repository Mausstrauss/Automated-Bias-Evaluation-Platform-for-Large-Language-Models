"""Unit tests for app.core.aggregation.BiasAggregator."""

import pytest
import pandas as pd

from app.core.aggregation import BiasAggregator

EXPECTED_COLS = {"Timestamp", "Model", "Metric", "Category", "Score"}


def _make_agg(tmp_path, subdir="output"):
    return BiasAggregator(output_dir=str(tmp_path / subdir))


# ── initialisation ─────────────────────────────────────────────────────────────

def test_init_creates_output_dir(tmp_path):
    """BiasAggregator creates the output directory on first initialisation."""
    out = tmp_path / "output"
    BiasAggregator(output_dir=str(out))
    assert out.exists()


def test_init_creates_history_csv(tmp_path):
    """A header-only history CSV with the correct schema is created on init."""
    agg = _make_agg(tmp_path)
    assert (tmp_path / "output" / "evaluation_history.csv").exists()
    df = pd.read_csv(str(tmp_path / "output" / "evaluation_history.csv"))
    assert EXPECTED_COLS.issubset(set(df.columns))


# ── add_result / buffer ────────────────────────────────────────────────────────

def test_add_result_goes_to_buffer_not_disk(tmp_path):
    """add_result accumulates entries in the in-memory buffer without writing to CSV."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "S", "cat", 0.1)
    agg.add_result("M", "S", "cat", 0.2)
    agg.add_result("M", "S", "cat", 0.3)
    assert len(agg.results_buffer) == 3
    # CSV should still have header only (0 data rows)
    df = pd.read_csv(agg.history_file)
    assert len(df) == 0


# ── save_to_history ────────────────────────────────────────────────────────────

def test_save_to_history_writes_to_disk(tmp_path):
    """save_to_history flushes the buffer to CSV; resulting file has (n_results, 5) shape."""
    agg = _make_agg(tmp_path)
    agg.add_result("A", "SentimentDiff", "gender", 0.1)
    agg.add_result("A", "Toxicity",      "safety", 0.2)
    agg.add_result("B", "SentimentDiff", "gender", 0.3)
    agg.save_to_history()
    df = pd.read_csv(agg.history_file)
    assert df.shape == (3, 5)


def test_save_to_history_correct_columns(tmp_path):
    """The persisted CSV contains exactly the five expected column names."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "S", "cat", 0.5)
    agg.save_to_history()
    df = pd.read_csv(agg.history_file)
    assert EXPECTED_COLS.issubset(set(df.columns))


def test_save_to_history_appends_not_overwrites(tmp_path):
    """Calling save_to_history twice accumulates all rows rather than overwriting."""
    agg = _make_agg(tmp_path)
    agg.add_result("A", "S", "c", 0.1)
    agg.add_result("A", "S", "c", 0.2)
    agg.save_to_history()

    agg.results_buffer.clear()
    agg.add_result("B", "S", "c", 0.3)
    agg.add_result("B", "S", "c", 0.4)
    agg.add_result("B", "S", "c", 0.5)
    agg.save_to_history()

    df = pd.read_csv(agg.history_file)
    assert len(df) == 5


def test_save_to_history_empty_buffer_noop(tmp_path):
    """save_to_history with an empty buffer does not touch the CSV file."""
    agg = _make_agg(tmp_path)
    mtime_before = (tmp_path / "output" / "evaluation_history.csv").stat().st_mtime
    agg.save_to_history()  # buffer is empty
    mtime_after = (tmp_path / "output" / "evaluation_history.csv").stat().st_mtime
    assert mtime_before == mtime_after


# ── get_history_df ─────────────────────────────────────────────────────────────

def test_get_history_df_returns_correct_shape(tmp_path):
    """get_history_df returns a DataFrame whose row count matches the number of saved results."""
    agg = _make_agg(tmp_path)
    for i in range(4):
        agg.add_result(f"M{i}", "S", "cat", float(i) * 0.1)
    agg.save_to_history()
    df = agg.get_history_df()
    assert df.shape == (4, 5)


def test_get_history_df_no_file_returns_empty(tmp_path):
    """get_history_df returns an empty DataFrame when the history CSV does not exist."""
    agg = _make_agg(tmp_path)
    # Delete the newly created CSV to simulate missing file
    import os
    os.remove(agg.history_file)
    result = agg.get_history_df()
    assert result.empty


def test_get_history_df_wrong_columns_returns_empty_with_schema(tmp_path):
    """A CSV with unexpected columns returns an empty DataFrame preserving the expected schema."""
    agg = _make_agg(tmp_path)
    # Overwrite with wrong column names
    pd.DataFrame({"A": [1], "B": [2]}).to_csv(agg.history_file, index=False)
    result = agg.get_history_df()
    assert result.empty
    assert EXPECTED_COLS.issubset(set(result.columns))


def test_get_history_df_corrupt_file_returns_empty(tmp_path):
    """A corrupt/binary CSV does not raise; an empty-ish DataFrame is returned instead."""
    agg = _make_agg(tmp_path)
    with open(agg.history_file, "w") as f:
        f.write("not,valid\x00csv\ncontent\x01here\n")
    result = agg.get_history_df()
    # Should not raise; returns empty-ish DataFrame
    assert isinstance(result, pd.DataFrame)


# ── get_results_df ─────────────────────────────────────────────────────────────

def test_get_results_df_returns_buffer_as_df(tmp_path):
    """get_results_df wraps the in-memory buffer as a DataFrame with the correct row count."""
    agg = _make_agg(tmp_path)
    agg.add_result("A", "S", "cat", 0.1)
    agg.add_result("B", "T", "cat", 0.2)
    df = agg.get_results_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


# ── score rounding ─────────────────────────────────────────────────────────────

def test_score_rounded_to_4_decimal_places(tmp_path):
    """Scores are stored rounded to 4 decimal places regardless of input precision."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "S", "cat", 0.123456789)
    assert agg.results_buffer[0]["Score"] == 0.1235


# ── calculate_use_case_score ───────────────────────────────────────────────────

def test_calculate_use_case_score_general(tmp_path):
    """General use-case weights both metrics equally; score equals the mean of abs values."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "SentimentDiff", "gender", 0.4)
    agg.add_result("M", "Wasserstein",   "vocab",  0.2)
    # general weights: both 1.0 → mean of abs values = (0.4 + 0.2) / 2 = 0.3
    score = agg.calculate_use_case_score("general")
    assert score == pytest.approx(0.3, abs=0.001)


def test_calculate_use_case_score_unknown_usecase_defaults_to_general(tmp_path):
    """An unrecognised use-case key falls back to the 'general' weight profile."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "SentimentDiff", "gender", 0.5)
    assert agg.calculate_use_case_score("nonexistent") == pytest.approx(
        agg.calculate_use_case_score("general"), abs=0.001
    )


# ── get_profile helpers ────────────────────────────────────────────────────────

def test_get_profile_by_metric_correct_keys(tmp_path):
    """get_profile_by_metric returns a dict keyed by metric name."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "SentimentDiff", "gender", 0.3)
    agg.add_result("M", "Toxicity",      "safety", 0.1)
    profile = agg.get_profile_by_metric()
    assert "SentimentDiff" in profile
    assert "Toxicity" in profile


def test_get_profile_by_bias_type_uses_abs_values(tmp_path):
    """get_profile_by_bias_type computes mean absolute score, so negative inputs are normalised."""
    agg = _make_agg(tmp_path)
    agg.add_result("M", "S", "gender", -0.4)
    profile = agg.get_profile_by_bias_type()
    assert profile["gender"] == pytest.approx(0.4, abs=0.001)
