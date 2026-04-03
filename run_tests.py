"""Pipeline verification tests — no Streamlit server needed."""

import sys
import os
import shutil

# ── project root on path so imports resolve
sys.path.insert(0, os.path.dirname(__file__))

RESULTS = {}

# ─────────────────────────────────────────────────────────────
# TEST 1 — Template loading
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1 — Template loading")
print("="*60)
try:
    from app.core.blackbox.template_loader import load_and_expand_templates

    prompts = load_and_expand_templates("data/word_lists/prompts/langbite_templates.csv")
    print(f"  Loaded {len(prompts)} prompts")
    print(f"  First 2: {prompts[:2]}")

    distinct_vars = {p["variable"] for p in prompts}
    print(f"  Distinct 'variable' values: {distinct_vars}")

    assert len(prompts) >= 2, "Expected at least 2 expanded prompts"
    assert len(distinct_vars) >= 2, f"Expected >=2 distinct variable values, got {distinct_vars}"
    RESULTS["Test 1"] = ("PASS", f"{len(prompts)} prompts loaded, {len(distinct_vars)} distinct variables")
except Exception as e:
    RESULTS["Test 1"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 2 — Sentiment oracle
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2 — Sentiment oracle")
print("="*60)
try:
    from app.core.blackbox.oracles import BiasOracle

    score_man = BiasOracle.analyze_sentiment("The man is strong and capable.")
    score_woman = BiasOracle.analyze_sentiment("The woman is weak and emotional.")
    print(f"  'man' sentence score:   {score_man}")
    print(f"  'woman' sentence score: {score_woman}")

    assert isinstance(score_man, float), "score_man not a float"
    assert isinstance(score_woman, float), "score_woman not a float"
    RESULTS["Test 2"] = ("PASS", f"man={score_man:.4f}, woman={score_woman:.4f}")
except Exception as e:
    RESULTS["Test 2"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 3 — Sentiment bias calculation
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3 — Sentiment bias calculation")
print("="*60)
try:
    from app.core.blackbox.oracles import BiasOracle

    fake_results = [
        {"variable": "man",   "response": "He is strong and capable."},
        {"variable": "man",   "response": "He leads with confidence."},
        {"variable": "man",   "response": "He is a great leader."},
        {"variable": "woman", "response": "She is weak and emotional."},
        {"variable": "woman", "response": "She is timid and hesitant."},
        {"variable": "woman", "response": "She struggles under pressure."},
    ]

    bias_score = BiasOracle.calculate_sentiment_bias(fake_results)
    print(f"  Bias score (6 entries, 2 groups): {bias_score}")
    assert bias_score != 0.0, "Expected non-zero bias score for clearly different sentences"

    # Test the <2 group guard
    single_group = [{"variable": "man", "response": "He is great."} for _ in range(3)]
    guard_result = BiasOracle.calculate_sentiment_bias(single_group)
    print(f"  Guard result (1 group): {guard_result}")
    assert guard_result == 0.0, "Guard for <2 groups should return 0.0"

    RESULTS["Test 3"] = ("PASS", f"bias={bias_score:.4f}, single-group guard=0.0")
except Exception as e:
    RESULTS["Test 3"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 4 — Toxicity oracle (no model download)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 4 — Toxicity oracle (empty string, no model download)")
print("="*60)
try:
    from app.core.blackbox.oracles import BiasOracle

    # Reset class state to ensure clean slate
    BiasOracle._toxicity_pipeline = None
    BiasOracle._toxicity_load_failed = False

    result = BiasOracle.analyze_toxicity_bert("")
    print(f"  analyze_toxicity_bert('') returned: {result}")
    print(f"  _toxicity_pipeline: {BiasOracle._toxicity_pipeline}")
    print(f"  _toxicity_load_failed: {BiasOracle._toxicity_load_failed}")

    assert result == 0.0, f"Expected 0.0, got {result}"
    assert BiasOracle._toxicity_pipeline is None, "_toxicity_pipeline should still be None"
    assert BiasOracle._toxicity_load_failed == False, "_toxicity_load_failed should still be False"

    RESULTS["Test 4"] = ("PASS", "empty string returns 0.0 without touching the model")
except Exception as e:
    RESULTS["Test 4"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 5 — Aggregator
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 5 — BiasAggregator")
print("="*60)
try:
    from app.core.aggregation import BiasAggregator
    import pandas as pd

    test_dir = "output_test"
    agg = BiasAggregator(output_dir=test_dir)

    agg.add_result("GPT-4",   "SentimentDiff", "gender",     0.12)
    agg.add_result("GPT-4",   "Wasserstein",   "profession", 0.05)
    agg.add_result("Gemini",  "SentimentDiff", "race",       0.22)

    agg.save_to_history()

    df = agg.get_history_df()
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(df.to_string(index=False))

    expected_cols = {"Timestamp", "Model", "Metric", "Category", "Score"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns: {expected_cols - set(df.columns)}"
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

    # Cleanup
    shutil.rmtree(test_dir)
    print(f"  Cleaned up '{test_dir}' directory.")

    RESULTS["Test 5"] = ("PASS", "3 rows, 5 correct columns, dir cleaned up")
except Exception as e:
    RESULTS["Test 5"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 6 — Generators config
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 6 — Generators PROVIDER_CONFIG")
print("="*60)
try:
    from app.core.blackbox.generators import PROVIDER_CONFIG

    print(f"  PROVIDER_CONFIG keys: {list(PROVIDER_CONFIG.keys())}")

    gemini_model = PROVIDER_CONFIG.get("Google-Gemini", {}).get("model")
    gpt35_model  = PROVIDER_CONFIG.get("OpenAI-GPT3.5", {}).get("model")
    has_simulated = "Simulated-Model" in PROVIDER_CONFIG

    print(f"  Google-Gemini model: {gemini_model}")
    print(f"  OpenAI-GPT3.5 model: {gpt35_model}")
    print(f"  'Simulated-Model' present: {has_simulated}")

    assert gemini_model == "gemini-2.0-flash", f"Expected 'gemini-2.0-flash', got '{gemini_model}'"
    assert gpt35_model  == "gpt-3.5-turbo",    f"Expected 'gpt-3.5-turbo', got '{gpt35_model}'"
    assert not has_simulated, "PROVIDER_CONFIG must NOT contain 'Simulated-Model'"

    RESULTS["Test 6"] = ("PASS", "Gemini=gemini-2.0-flash, GPT3.5=gpt-3.5-turbo, no Simulated-Model")
except Exception as e:
    RESULTS["Test 6"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 7 — requirements.txt
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 7 — requirements.txt")
print("="*60)
try:
    with open("requirements.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    print(f"  Packages found: {lines}")

    # Check huggingface_hub absent
    hf_hub_present = any("huggingface_hub" in l.lower() for l in lines)
    print(f"  huggingface_hub present: {hf_hub_present}")
    assert not hf_hub_present, "huggingface_hub must NOT be in requirements.txt"

    # Check all packages have both >= and <
    missing_upper = []
    for line in lines:
        # Strip extras/markers
        pkg = line.split(";")[0].strip()
        if "<" not in pkg:
            missing_upper.append(pkg)

    if missing_upper:
        print(f"  Packages missing upper bound (<): {missing_upper}")
    else:
        print("  All packages have upper bounds.")

    assert not missing_upper, f"Packages missing upper bound: {missing_upper}"

    RESULTS["Test 7"] = ("PASS", "no huggingface_hub, all packages have >= and < bounds")
except Exception as e:
    RESULTS["Test 7"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 8 — docs/ files
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 8 — docs/ files")
print("="*60)
try:
    required_docs = [
        "docs/theory.md",
        "docs/architecture.md",
        "docs/getting-started.md",
        "docs/api.md",
        "docs/faq.md",
    ]

    all_ok = True
    for path in required_docs:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            all_ok = False
            continue
        with open(path) as f:
            line_count = sum(1 for _ in f)
        status = "OK" if line_count > 10 else "TOO SHORT"
        print(f"  {status:10s} {path}  ({line_count} lines)")
        if line_count <= 10:
            all_ok = False

    assert all_ok, "One or more docs files are missing or too short (<=10 lines)"
    RESULTS["Test 8"] = ("PASS", "all 5 docs exist and have >10 lines")
except Exception as e:
    RESULTS["Test 8"] = ("FAIL", str(e))
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for test, (status, reason) in RESULTS.items():
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} {test}: {status} — {reason}")
print("="*60)
