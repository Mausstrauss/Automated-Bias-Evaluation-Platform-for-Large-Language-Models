"""Headless scheduler service that runs recurring bias audits."""

import schedule  # FIX: Existing dependency for simple cron-like scheduling.
import time
import os
import sys
import pandas as pd  
from datetime import datetime
import json  # FIX: Needed to read scheduler_config.json written by the GUI.


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle
from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.aggregation import BiasAggregator


TARGET_MODEL = os.getenv("TARGET_MODEL", "OpenAI-GPT3.5")  # FIX: Preserve default target for backward compatibility.
API_KEY = os.getenv("LLM_API_KEY", None)  # FIX: Keep environment-based API key fallback.
RUN_TIME = os.getenv("SCHEDULE_TIME", "03:00")  # FIX: Document default run time in English.

CONFIG_PATH = "scheduler_config.json"  # FIX: Shared location for GUI-saved scheduler configuration.

MODEL_MAPPING = {  # FIX: Map human-readable labels to provider IDs, matching the GUI.
    "OpenAI GPT 3.5": "OpenAI-GPT3.5",
    "Google Gemini Pro": "Google-Gemini",
}


def load_config() -> dict:  # FIX: Helper to read user configuration from shared volume.
    """Load scheduler configuration from JSON, with safe defaults on failure."""  # FIX: Short docstring describing fallback behavior.
    if os.path.exists(CONFIG_PATH):  # FIX: Only attempt file IO when config exists.
        try:
            with open(CONFIG_PATH) as f:  # FIX: Read JSON config produced by the GUI.
                return json.load(f)  # FIX: Return parsed configuration on success.
        except Exception:  # FIX: Ignore malformed config files and fall back to defaults.
            pass
    return {  # FIX: Provide reasonable defaults when no config is present or loading fails.
        "active": True,
        "models": [os.getenv("TARGET_MODEL", "OpenAI-GPT3.5")],
        "metrics": ["Sentiment Analysis", "Toxicity Check"],
    }


def run_daily_audit():
    """Run one scheduled audit for the configured `TARGET_MODEL`."""
   
    config = load_config()  # FIX: Read latest scheduler configuration, if any.
    if not config.get("active", True):  # FIX: Respect the GUI toggle that can disable the scheduler.
        print("[INFO] Scheduler disabled in configuration; skipping run.")  # FIX: Log skip reason for visibility.
        return  # FIX: Exit early when the scheduler is not active.

    target_models = config.get("models", [TARGET_MODEL])  # FIX: Allow multiple display labels to be configured.
    metrics = config.get("metrics", ["Sentiment Analysis", "Toxicity Check"])  # FIX: Use configured metrics when present.

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] Starting Scheduled Audit for {target_models}...")  # FIX: Log all models that will be audited.

    # 1. SETUP
    aggregator = BiasAggregator()  # FIX: New aggregator per run; history is persisted to CSV.
    all_results = []  # FIX: Collect raw results from all models so earlier models are not lost when writing the CSV log.

    csv_path = "data/word_lists/prompts/langbite_templates.csv"  # FIX: Use the same baseline templates as the GUI.
    if not os.path.exists(csv_path):  # FIX: Guard against missing template file.
        print(f" [WARNING] Template file not found at {csv_path}. Skipping run.")  # FIX: Explain why no audit is run.
        return  # FIX: Exit run when there is no data to evaluate.

    prompts = load_and_expand_templates(csv_path)  # FIX: Reuse existing loader to expand templates into prompts.
    if not prompts:  # FIX: Skip run when template expansion yields no prompts.
        print(" [WARNING] No prompts loaded. Skipping run.")  # FIX: Keep operator informed about empty datasets.
        return  # FIX: Avoid passing empty data through the pipeline.

    print(f" [INFO] Loaded {len(prompts)} prompts for evaluation.")  # FIX: Preserve informational logging.

    try:
        for display_name in target_models:  # FIX: Support auditing multiple models per scheduled run.
            provider_id = MODEL_MAPPING.get(display_name, display_name)  # FIX: Translate GUI label to provider ID where possible.

            # 3. GENERATION
            print(f" [INFO] Generating responses for {display_name} ({provider_id})...")  # FIX: Clarify which model is being evaluated.
            generator = LLMGenerator(provider=provider_id, api_key=API_KEY)  # FIX: Use resolved provider identifier with existing generator.
            results = generator.generate_batch(prompts)  # FIX: Reuse existing batch-generation logic.
            all_results.extend(results)  # FIX: Append this model's outputs to the shared list so all audited models appear in the raw log.

            # 4. EVALUATION (ORACLES)
            if "Toxicity Check" in metrics:  # FIX: Respect configuration for which metrics to compute.
                print(" [INFO] Running Toxicity Check...")  # FIX: Preserve step-level logging.
                tox_score = BiasOracle.calculate_toxicity_bias(results)  # FIX: Use shared oracle implementation.
                aggregator.add_result(display_name, "Toxicity", "safety", tox_score)  # FIX: Attribute results to the human-readable model name.

            if "Sentiment Analysis" in metrics:  # FIX: Only compute sentiment when requested.
                print(" [INFO] Running Sentiment Analysis...")  # FIX: Preserve log message.
                sent_score = BiasOracle.calculate_sentiment_bias(results)  # FIX: Reuse sentiment oracle.
                aggregator.add_result(display_name, "SentimentDiff", "gender", sent_score)  # FIX: Keep existing metric semantics.

        # Note: a semantic evaluator metric is not implemented in this prototype.

        # 5. RAW DATA LOGGING (Traceability)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # FIX: Timestamp for file naming remains unchanged.
        log_dir = "output/logs"  # FIX: Keep logs in existing directory for compatibility.
        os.makedirs(log_dir, exist_ok=True)  # FIX: Ensure log directory exists before writing.

        log_filename = f"{log_dir}/audit_{timestamp_str}.csv"  # FIX: Drop model from filename now that multiple models may be audited per run.
        pd.DataFrame(all_results).to_csv(log_filename, index=False)  # FIX: Persist combined raw results for all evaluated models instead of only the last one.
        print(f" [INFO] Raw audit logs saved to {log_filename}")  # FIX: Update log message to match new filename.

        # 6. SAVE TO HISTORY (Aggregation)
        aggregator.save_to_history()  # FIX: Append all model results collected in this run to the shared history CSV.
        print(f" [INFO] Audit finished successfully. Results saved to history.")  # FIX: Maintain summary log statement.

    except Exception as e:
        print(f" [ERROR] Error during scheduled audit: {e}")  # FIX: Keep broad error handling to avoid crashing the scheduler loop.

# --- SCHEDULING SETUP ---

print(f" [INFO] Bias Monitor Scheduler Initialized")  # FIX: Initialization banner for the scheduler service.
print(f" Target Model (env default): {TARGET_MODEL}")  # FIX: Clarify that this is the environment default value.
print(f" Schedule Time: Every day at {RUN_TIME}")  # FIX: Keep schedule log, phrased in English.

# Register scheduled job
schedule.every().day.at(RUN_TIME).do(run_daily_audit)  # FIX: Hook run_daily_audit into the daily schedule.



# Main scheduler loop
while True:  # FIX: Preserve simple infinite loop to keep scheduler alive.
    schedule.run_pending()  # FIX: Execute any jobs that are due.
    time.sleep(60)  # FIX: Poll once per minute to balance load and responsiveness.