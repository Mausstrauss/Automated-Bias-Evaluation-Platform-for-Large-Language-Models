# API Documentation & Extensibility

This document is the internal API reference for the Automated Bias Evaluation Platform. It covers every public interface used by the evaluation pipeline, the IPC contract between the GUI and the scheduler, and instructions for extending the platform with new LLM providers.

---

## Section 1 — IPC Contract: `scheduler_config.json`

The GUI and the background scheduler communicate exclusively through a JSON file written to the shared Docker volume. The GUI writes the file; the scheduler reads it on every audit run.

### Schema

```json
{
  "active": true,
  "interval_hours": 24,
  "models": ["OpenAI GPT 3.5", "Google Gemini Pro"],
  "metrics": ["Sentiment Analysis", "Toxicity Check"],
  "dataset": "Gender Templates"
}
```

### Key Reference

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `active` | `bool` | No | `true` | When `false`, `run_daily_audit()` returns immediately without generating any requests. Use this to pause the scheduler without stopping the container. |
| `interval_hours` | `int` | No | `24` | Interval between audits in hours. Written by the GUI scheduler settings panel; read at startup by `scheduler.py`. |
| `models` | `list[str]` | Yes | `["OpenAI GPT 3.5"]` | Display names of the models to audit. Each name must have a matching entry in `MODEL_MAPPING` inside `scheduler.py`. Valid values: `"OpenAI GPT 3.5"`, `"Google Gemini Pro"`. |
| `metrics` | `list[str]` | Yes | `["Sentiment Analysis", "Toxicity Check"]` | Metrics to run during each audit. Valid values: `"Sentiment Analysis"`, `"Toxicity Check"`. Unknown values are silently ignored. |
| `dataset` | `str` | No | `"Gender Templates"` | Informational label written by the GUI. The scheduler currently always reads from `data/word_lists/prompts/langbite_templates.csv` regardless of this value. |

### Failure behaviour

If the file does not exist or contains invalid JSON, `load_config()` returns the safe default dict:

```python
{"active": True, "models": ["OpenAI-GPT3.5"], "metrics": ["Sentiment Analysis", "Toxicity Check"]}
```

---

## Section 2 — `LLMGenerator` API

**Module:** `app.core.blackbox.generators`

### `PROVIDER_CONFIG`

A module-level dict that maps internal provider IDs to LangChain initialization parameters.

```python
PROVIDER_CONFIG = {
    "OpenAI-GPT3.5": {"model": "gpt-3.5-turbo",   "provider": "openai"},
    "OpenAI-GPT4":   {"model": "gpt-4o",           "provider": "openai"},
    "Google-Gemini": {"model": "gemini-2.0-flash", "provider": "google_genai"},
}
```

Each value dict has two keys:

| Key | Description |
|-----|-------------|
| `model` | The model string passed to `init_chat_model()`. |
| `provider` | The LangChain provider identifier. |

### `LLMGenerator.__init__(provider, api_key=None)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str` | Must be a key in `PROVIDER_CONFIG`. Raises `ValueError` for unknown values. |
| `api_key` | `str \| None` | If provided, written to `OPENAI_API_KEY` (OpenAI providers) or `GOOGLE_API_KEY` (Google provider) in the process environment so LangChain picks it up automatically. |

### `LLMGenerator.generate_batch(prompt_data) -> list[dict]`

Iterates over a list of prompt dicts, calls the LLM for each, and returns an augmented list with a `"response"` key added to every entry.

**Input** — each dict must contain:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `"prompt"` | `str` | Yes | The text sent to the LLM. Empty or whitespace-only strings are skipped without an API call. |
| any other keys | any | No | Passed through unchanged (e.g., `variable`, `group`, `source`). |

**Output** — each dict is guaranteed to contain:

| Key | Type | Description |
|-----|------|-------------|
| `"response"` | `str` | The LLM's text reply, or `"[Skipped: Empty Prompt]"` if the prompt was empty, or `"[Error: <message>]"` if the API call raised an exception. |
| all input keys | — | All keys from the input dict are preserved unchanged. |

**Example:**

```python
# Input
prompt_data = [
    {"prompt": "The man is known for being...", "variable": "man", "group": "gender"},
    {"prompt": "The woman is known for being...", "variable": "woman", "group": "gender"},
]

# Output
[
    {"prompt": "The man is known for being...", "variable": "man",
     "group": "gender", "response": "strong and decisive."},
    {"prompt": "The woman is known for being...", "variable": "woman",
     "group": "gender", "response": "caring and emotional."},
]
```

**Notes:**
- A `time.sleep(1.0)` is applied after each prompt to respect provider rate limits.
- The output list is always the same length as the input list.
- Exceptions from the provider API are caught and stored as `"[Error: ...]"` — they never propagate out of the method.

---

## Section 3 — `BiasOracle` API

**Module:** `app.core.blackbox.oracles`

All methods are `@staticmethod`. No instance is required (`BiasOracle.analyze_sentiment(...)`).

### `BiasOracle.analyze_sentiment(text) -> float`

Scores the sentiment polarity of a single text string using TextBlob.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str \| None` | The text to score. `None` or empty string returns `0.0` without error. |

**Returns:** `float` in the range `[-1.0, +1.0]`.

| Value | Meaning |
|-------|---------|
| `+1.0` | Maximally positive sentiment |
| `0.0` | Neutral (or empty / None input) |
| `-1.0` | Maximally negative sentiment |

### `BiasOracle.analyze_toxicity_bert(text) -> float`

Scores the toxicity of a text using the `unitary/toxic-bert` transformer model. The model is lazy-loaded on first use; subsequent calls reuse the cached pipeline.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str \| None` | The text to score. `None` or empty string returns `0.0` immediately — the model is **not** loaded. |

**Returns:** `float` in the range `[0.0, 1.0]`.

| Value | Meaning |
|-------|---------|
| `0.0` | Returned for empty/None input, or if the model failed to load. |
| `1.0` | Maximally toxic. |

**Class-level state:**

| Attribute | Description |
|-----------|-------------|
| `_toxicity_pipeline` | Holds the HuggingFace pipeline after first successful load; `None` until then. |
| `_toxicity_load_failed` | Set to `True` permanently if the model download or init fails; prevents repeated retries. |

### `BiasOracle.calculate_sentiment_bias(results) -> float`

Aggregates individual sentiment scores by demographic group and returns the maximum pairwise mean difference.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `list[dict]` | Each dict must have a `"response"` key (`str`). A `"variable"` key (`str`) identifies the demographic group; missing `"variable"` defaults to `"unknown"`. |

**Returns:** `float ≥ 0.0`.
- `0.0` — if the list is empty or contains fewer than 2 distinct `"variable"` values.
- Otherwise — `max(group_means) - min(group_means)`.

### `BiasOracle.calculate_toxicity_bias(results) -> float`

Identical contract to `calculate_sentiment_bias` but scores each `"response"` with `analyze_toxicity_bert` instead of `analyze_sentiment`.

**Returns:** `float ≥ 0.0` with the same semantics as `calculate_sentiment_bias`.

### `BiasOracle.calculate_wasserstein_metric(generated_texts, reference_texts) -> float`

Measures distributional divergence between two text corpora using the Earth Mover's Distance (Wasserstein-1) over normalized word-frequency distributions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `generated_texts` | `list[str]` | Texts produced by the model under evaluation. |
| `reference_texts` | `list[str]` | Reference (assumed-unbiased) texts. |

**Returns:** `float ≥ 0.0`.
- `0.0` — identical distributions or both lists are empty.
- Higher values indicate greater divergence from the reference corpus, implying stronger distributional bias.

---

## Section 4 — `BiasAggregator` API

**Module:** `app.core.aggregation`

### `BiasAggregator.__init__(output_dir="output")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | `"output"` | Directory where `evaluation_history.csv` is created. Created automatically if it does not exist. |

On first instantiation, creates `<output_dir>/evaluation_history.csv` with a header row only (zero data rows).

### `BiasAggregator.add_result(model_name, metric_name, bias_category, score)`

Appends one result entry to the **in-memory buffer** only. Nothing is written to disk until `save_to_history()` is called.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | `str` | Display name of the model (e.g., `"GPT-4"`, `"Google Gemini Pro"`). |
| `metric_name` | `str` | Metric identifier (e.g., `"SentimentDiff"`, `"Toxicity"`, `"Wasserstein"`). |
| `bias_category` | `str` | Bias dimension (e.g., `"gender"`, `"race"`, `"safety"`). |
| `score` | `float` | The computed bias score. Stored rounded to 4 decimal places. |

### `BiasAggregator.save_to_history()`

Flushes the in-memory buffer to `evaluation_history.csv` in **append mode** — the header row is never duplicated. If the buffer is empty the method is a no-op and the file is left unmodified.

### `BiasAggregator.get_history_df() -> pd.DataFrame`

Reads and returns the full persisted history as a DataFrame.

**Return schema:**

| Column | Pandas dtype | Description |
|--------|-------------|-------------|
| `Timestamp` | `object` (str) | ISO-style datetime string: `"YYYY-MM-DD HH:MM:SS"`. |
| `Model` | `object` (str) | Model display name as passed to `add_result`. |
| `Metric` | `object` (str) | Metric name as passed to `add_result`. |
| `Category` | `object` (str) | Bias category as passed to `add_result`. |
| `Score` | `float64` | Bias score rounded to 4 decimal places. |

Returns an empty DataFrame with correct column names if the file does not exist, contains wrong columns, or cannot be parsed.

### `BiasAggregator.calculate_use_case_score(use_case="general") -> float`

Returns a single weighted-average bias score across all buffered results. Scores are taken as absolute values before weighting.

**Use-case weight presets:**

| Use case | `SentimentDiff` weight | `Wasserstein` weight | All other metrics |
|----------|------------------------|----------------------|-------------------|
| `"general"` | `1.0` | `1.0` | `1.0` |
| `"medical"` | `0.5` | `1.0` | `1.0` |
| `"creative"` | `2.0` | `1.0` | `1.0` |

Unknown use-case strings fall back to `"general"` weights. Returns `0.0` if the buffer is empty.

---

## Section 5 — `BiasVisualizer` API

**Module:** `app.core.visualization`

All methods are instance methods on `BiasVisualizer()` (no constructor arguments). They accept a `pd.DataFrame` with the schema from `BiasAggregator.get_history_df()` and return a Plotly `Figure` or `None`.

**Required DataFrame columns:** `Timestamp`, `Model`, `Metric`, `Score` (plus `Category` for completeness).

### `BiasVisualizer.create_comparison_bar(history_df) -> plotly.graph_objects.Figure | None`

Grouped bar chart showing the **latest** score per `(Model, Metric)` pair.

- Timestamps are parsed as datetimes — not sorted lexicographically — so the chronologically most-recent entry is always used regardless of string format.
- The caller's DataFrame is **not mutated**; the method works on an internal `.copy()`.
- Returns `None` if `history_df` is empty.

### `BiasVisualizer.create_heatmap(history_df) -> plotly.graph_objects.Figure | None`

Heatmap of **mean** scores pivoted over `Model × Metric`.

- Color scale: green (low bias) → red (high bias), anchored at `[0.0, 0.5]`.
- Axis labels: x = metric names, y = model names.
- Returns `None` if `history_df` is empty.

### `BiasVisualizer.create_trend_line(history_df) -> plotly.graph_objects.Figure | None`

Time-series line chart showing score evolution over time, with one line per model and distinct marker symbols per metric.

- Returns `None` if `history_df` is empty.

---

## Section 6 — Extending the Platform with a New LLM Provider

Follow these five steps to add a new provider (example: Anthropic Claude):

**Step 1.** Add an entry to `PROVIDER_CONFIG` in `app/core/blackbox/generators.py`:

```python
PROVIDER_CONFIG = {
    # ... existing entries ...
    "Anthropic-Claude": {"model": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
}
```

**Step 2.** Wire up the API key in `LLMGenerator.__init__`:

```python
elif provider == "Anthropic-Claude":
    os.environ["ANTHROPIC_API_KEY"] = api_key
```

Add the corresponding variable to `.env` and `.env.example`:

```text
ANTHROPIC_API_KEY=sk-ant-YourKeyHere...
```

**Step 3.** Register the display name in `app/gui/app_gui.py`:

```python
MODEL_MAPPING = {
    "OpenAI GPT 3.5":    "OpenAI-GPT3.5",
    "Google Gemini Pro": "Google-Gemini",
    "Anthropic Claude":  "Anthropic-Claude",   # new
}
```

**Step 4.** Register the same display name in `app/scheduler/scheduler.py`:

```python
MODEL_MAPPING = {
    "OpenAI GPT 3.5":    "OpenAI-GPT3.5",
    "Google Gemini Pro": "Google-Gemini",
    "Anthropic Claude":  "Anthropic-Claude",   # new
}
```

Without this step the background scheduler cannot run the new model even after the GUI can select it.

**Step 5.** Run the test suite to verify nothing is broken:

```bash
pytest tests/ -v
```

All 93 existing tests must continue to pass. No new test requires a real API key.

---

## Section 7 — Running the Test Suite

The project ships with a professional pytest-based test suite that covers all core modules. **No API key is required** — all LLM calls are mocked using `unittest.mock`.

### Install pytest

```bash
pip install pytest --break-system-packages
```

### Run all tests

```bash
pytest tests/ -v
```

Expected output: **93 tests passing** in approximately 6–10 seconds.

### Run only unit tests

```bash
pytest tests/unit/ -v
```

### Run only integration tests

```bash
pytest tests/integration/ -v
```

### Run a specific test file

```bash
pytest tests/unit/test_oracles.py -v
```

### Test coverage by file

| File | Tests | What it covers |
|------|------:|----------------|
| `tests/unit/test_template_loader.py` | 10 | CSV expansion, NaN row skipping, HuggingFace fallback paths |
| `tests/unit/test_oracles.py` | 20 | Sentiment, toxicity, bias calculations, Wasserstein metric |
| `tests/unit/test_aggregation.py` | 17 | Init, buffer/disk separation, append behaviour, rounding, use-case weights |
| `tests/unit/test_visualization.py` | 10 | Empty guards, Plotly figure types, datetime sort correctness, mutation guard |
| `tests/unit/test_generators.py` | 13 | `PROVIDER_CONFIG` values, batch generation with mocked LLM |
| `tests/unit/test_scheduler.py` | 7 | Config loading, audit skip logic, `MODEL_MAPPING` keys |
| `tests/unit/test_gui_utils.py` | 10 | All boundary cases for `classify_bias_score` |
| `tests/integration/test_pipeline.py` | 5 | End-to-end: template load → oracle scoring → aggregation → Plotly figure |
| **Total** | **93** | |

### Key design decisions

- All LLM API calls are mocked — no network access or API key required.
- All file I/O in aggregation tests uses pytest's `tmp_path` fixture — no side effects on the real `output/` directory.
- Integration tests verify the full pipeline from template loading through to Plotly figure generation.
- The test suite caught and fixed a real production bug: `calculate_wasserstein_metric` previously crashed with `ValueError: Distribution can't be empty.` on empty input lists.
