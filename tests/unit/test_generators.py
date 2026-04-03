"""Unit tests for app.core.blackbox.generators."""

import os
import pytest
from unittest.mock import patch, MagicMock, call

from app.core.blackbox.generators import PROVIDER_CONFIG, LLMGenerator


# ── PROVIDER_CONFIG ────────────────────────────────────────────────────────────

def test_provider_config_google_gemini_model():
    """Google-Gemini is configured to use the gemini-2.0-flash model."""
    assert PROVIDER_CONFIG["Google-Gemini"]["model"] == "gemini-2.0-flash"


def test_provider_config_google_gemini_provider():
    """Google-Gemini uses the 'google_genai' LangChain provider string."""
    assert PROVIDER_CONFIG["Google-Gemini"]["provider"] == "google_genai"


def test_provider_config_openai_gpt35_model():
    """OpenAI-GPT3.5 maps to the gpt-3.5-turbo model identifier."""
    assert PROVIDER_CONFIG["OpenAI-GPT3.5"]["model"] == "gpt-3.5-turbo"


def test_provider_config_openai_gpt4_model():
    """OpenAI-GPT4 maps to the gpt-4o model identifier."""
    assert PROVIDER_CONFIG["OpenAI-GPT4"]["model"] == "gpt-4o"


def test_no_simulated_model_in_config():
    """There is no 'Simulated-Model' key — all providers require real API credentials."""
    assert "Simulated-Model" not in PROVIDER_CONFIG


# ── LLMGenerator.__init__ ──────────────────────────────────────────────────────

def test_unknown_provider_raises_value_error():
    """Passing an unrecognised provider key raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMGenerator(provider="Unknown-XYZ")


def test_api_key_set_in_environment_openai(monkeypatch):
    """Providing an api_key for an OpenAI provider injects it into OPENAI_API_KEY."""
    with patch("app.core.blackbox.generators.init_chat_model"):
        LLMGenerator(provider="OpenAI-GPT3.5", api_key="test-key-123")
    assert os.environ.get("OPENAI_API_KEY") == "test-key-123"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_api_key_set_in_environment_google(monkeypatch):
    """Providing an api_key for the Google-Gemini provider injects it into GOOGLE_API_KEY."""
    with patch("app.core.blackbox.generators.init_chat_model"):
        LLMGenerator(provider="Google-Gemini", api_key="test-key-456")
    assert os.environ.get("GOOGLE_API_KEY") == "test-key-456"
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


# ── LLMGenerator.generate_batch ───────────────────────────────────────────────

def _make_generator(provider="OpenAI-GPT3.5"):
    """Return a LLMGenerator with init_chat_model mocked out."""
    with patch("app.core.blackbox.generators.init_chat_model") as mock_init:
        mock_llm = MagicMock()
        mock_init.return_value = mock_llm
        gen = LLMGenerator(provider=provider, api_key="fake")
    return gen


def _fake_response(text="Generated text."):
    r = MagicMock()
    r.content = text
    return r


def test_generate_batch_adds_response_key():
    """Each result dict from generate_batch contains a 'response' key."""
    gen = _make_generator()
    gen.llm.invoke.return_value = _fake_response()
    with patch("app.core.blackbox.generators.time.sleep"):
        results = gen.generate_batch([{"prompt": "Say something.", "variable": "man"}])
    assert "response" in results[0]


def test_generate_batch_preserves_metadata():
    """All input metadata keys are copied into the result dicts alongside 'response'."""
    gen = _make_generator()
    gen.llm.invoke.return_value = _fake_response("Hello")
    prompts = [
        {"prompt": "P1", "variable": "man",   "group": "gender", "source": "csv"},
        {"prompt": "P2", "variable": "woman", "group": "gender", "source": "csv"},
    ]
    with patch("app.core.blackbox.generators.time.sleep"):
        results = gen.generate_batch(prompts)
    for orig, res in zip(prompts, results):
        for key in orig:
            assert key in res, f"Metadata key '{key}' missing from result"


def test_generate_batch_empty_prompt_skipped():
    """An empty-string prompt is skipped without calling the LLM; a sentinel string is stored."""
    gen = _make_generator()
    with patch("app.core.blackbox.generators.time.sleep"):
        results = gen.generate_batch([{"prompt": ""}])
    assert results[0]["response"] == "[Skipped: Empty Prompt]"
    gen.llm.invoke.assert_not_called()


def test_generate_batch_api_error_stored_not_raised():
    """API exceptions are caught and stored as '[Error: ...]' strings rather than propagating."""
    gen = _make_generator()
    gen.llm.invoke.side_effect = Exception("timeout")
    with patch("app.core.blackbox.generators.time.sleep"):
        results = gen.generate_batch([{"prompt": "Test prompt."}])
    assert "timeout" in results[0]["response"]
    assert "[Error:" in results[0]["response"]


def test_generate_batch_returns_same_length_as_input():
    """Output list length always matches input list length regardless of errors."""
    gen = _make_generator()
    gen.llm.invoke.return_value = _fake_response("ok")
    prompts = [{"prompt": f"Prompt {i}"} for i in range(5)]
    with patch("app.core.blackbox.generators.time.sleep"):
        results = gen.generate_batch(prompts)
    assert len(results) == 5
