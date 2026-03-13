"""Headless scheduler service that runs recurring bias audits."""

import schedule
import time
import os
import sys
import pandas as pd
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle
from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.aggregation import BiasAggregator

TARGET_MODEL = os.getenv("TARGET_MODEL", "OpenAI-GPT3.5")
API_KEY = os.getenv("LLM_API_KEY", None)
RUN_TIME = os.getenv("SCHEDULE_TIME", "03:00")

CONFIG_PATH = "scheduler_config.json"

MODEL_MAPPING = {
    "OpenAI GPT 3.5": "OpenAI-GPT3.5",
    "Google Gemini Pro": "Google-Gemini",
}


def load_config() -> dict:
    """Load scheduler configuration from JSON, with safe defaults on failure."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "active": True,
        "models": [os.getenv("TARGET_MODEL", "OpenAI-GPT3.5")],
        "metrics": ["Sentiment Analysis", "Toxicity Check"],
    }


def discover_models() -> list:
    """Optionally auto-discover available models from provider APIs."""
    models = []

    auto_flag = os.getenv("AUTO_DISCOVER_MODELS", "").lower()
    if auto_flag not in ("1", "true", "yes", "on"):
        return models

    # OpenAI discovery: check for presence of GPT-3.5 / GPT-4class models
    try:
        import openai

        openai.api_key = os.getenv("OPENAI_API_KEY") or API_KEY
        if openai.api_key:
            resp = openai.Model.list()
            ids = [m["id"] for m in resp["data"]]
            if any("gpt-4" in mid for mid in ids):
                models.append("OpenAI GPT 4")
            if any("gpt-3.5" in mid for mid in ids):
                models.append("OpenAI GPT 3.5")
    except Exception as e:
        print(f"[WARN] Auto-discovery for OpenAI failed: {e}")

    # Fallback: if nothing discovered, return empty and let config take over
    return models


def run_daily_audit():
    """Run one scheduled audit for the configured models."""

    config = load_config()
    if not config.get("active", True):
        print("[INFO] Scheduler disabled in configuration; skipping run.")
        return

    # Use auto-discovered models if enabled, otherwise configured ones
    auto_models = discover_models()
    if auto_models:
        target_models = auto_models
        print(f"[INFO] Auto-discovered models for audit: {target_models}")
    else:
        target_models = config.get("models", [TARGET_MODEL])

    metrics = config.get("metrics", ["Sentiment Analysis", "Toxicity Check"])

    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] Starting Scheduled Audit for {target_models}..."
    )

    # 1. SETUP
    aggregator = BiasAggregator()
    all_results = []

    csv_path = "data/word_lists/prompts/langbite_templates.csv"
    if not os.path.exists(csv_path):
        print(f" [WARNING] Template file not found at {csv_path}. Skipping run.")
        return

    prompts = load_and_expand_templates(csv_path)
    if not prompts:
        print(" [WARNING] No prompts loaded. Skipping run.")
        return

    print(f" [INFO] Loaded {len(prompts)} prompts for evaluation.")

    try:
        for display_name in target_models:
            provider_id = MODEL_MAPPING.get(display_name, display_name)

            # 3. GENERATION
            print(f" [INFO] Generating responses for {display_name} ({provider_id})...")
            generator = LLMGenerator(provider=provider_id, api_key=API_KEY)
            results = generator.generate_batch(prompts)
            all_results.extend(results)

            # 4. EVALUATION (ORACLES)
            if "Toxicity Check" in metrics:
                print(" [INFO] Running Toxicity Check...")
                tox_score = BiasOracle.calculate_toxicity_bias(results)
                aggregator.add_result(display_name, "Toxicity", "safety", tox_score)

            if "Sentiment Analysis" in metrics:
                print(" [INFO] Running Sentiment Analysis...")
                sent_score = BiasOracle.calculate_sentiment_bias(results)
                aggregator.add_result(display_name, "SentimentDiff", "gender", sent_score)

        # 5. RAW DATA LOGGING (Traceability)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "output/logs"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = f"{log_dir}/audit_{timestamp_str}.csv"
        pd.DataFrame(all_results).to_csv(log_filename, index=False)
        print(f" [INFO] Raw audit logs saved to {log_filename}")

        # 6. SAVE TO HISTORY (Aggregation)
        aggregator.save_to_history()
        print(f" [INFO] Audit finished successfully. Results saved to history.")

    except Exception as e:
        print(f" [ERROR] Error during scheduled audit: {e}")


def main():
    # --- SCHEDULING SETUP ---
    print(f" [INFO] Bias Monitor Scheduler Initialized")
    print(f" Target Model (env default): {TARGET_MODEL}")
    print(f" Schedule Time: Every day at {RUN_TIME}")

    # Register scheduled job
    schedule.every().day.at(RUN_TIME).do(run_daily_audit)

    # Main scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()

