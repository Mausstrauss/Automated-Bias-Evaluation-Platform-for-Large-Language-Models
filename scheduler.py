import schedule
import time
import os
import sys
import pandas as pd  
from datetime import datetime


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle, LLMAsJudgeOracle
from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.aggregation import BiasAggregator


TARGET_MODEL = os.getenv("TARGET_MODEL", "Simulated-Model") 
API_KEY = os.getenv("LLM_API_KEY", None)
RUN_TIME = os.getenv("SCHEDULE_TIME", "03:00") # Standard: 3 Uhr morgens

def run_daily_audit():
   
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] Starting Scheduled Audit for {TARGET_MODEL}...")

    # 1. SETUP
    aggregator = BiasAggregator()
    
   
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
        # 3. GENERATION
        # Initialisierung mit API Key aus Environment Variable
        generator = LLMGenerator(provider=TARGET_MODEL, api_key=API_KEY)
        results = generator.generate_batch(prompts)

        # 4. EVALUATION (ORACLES)
        
        # A) Toxicity (Lokal)
        print(" [INFO] Running Toxicity Check...")
        tox_score = BiasOracle.calculate_toxicity_bias(results)
        aggregator.add_result(TARGET_MODEL, "Toxicity", "safety", tox_score)

        # B) Sentiment (Lokal)
        print(" [INFO] Running Sentiment Analysis...")
        sent_score = BiasOracle.calculate_sentiment_bias(results)
        aggregator.add_result(TARGET_MODEL, "SentimentDiff", "gender", sent_score)

        # C) LLM Judge (Nur wenn wir im Simulations-Modus sind oder API Kosten egal sind)
        # Um Kosten zu sparen, könnte man den Judge im Scheduler deaktivieren oder limitieren
        if "Simulated" in TARGET_MODEL:
            print(" [INFO] Running Simulated Judge...")
            judge_engine = LLMGenerator(provider="Simulated-Model")
            judge = LLMAsJudgeOracle(judge_engine)
            
            biased_count = sum(1 for r in results if judge.evaluate_response(r["prompt"], r["response"])["is_biased"])
            judge_score = biased_count / len(results) if results else 0
            aggregator.add_result(TARGET_MODEL, "GPTBIAS", "semantic", judge_score)

        # 5. RAW DATA LOGGING (Nachvollziehbarkeit)
        # Speichert die genauen Antworten und Prompts für spätere Audits
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "output/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"{log_dir}/audit_{TARGET_MODEL}_{timestamp_str}.csv"
        pd.DataFrame(results).to_csv(log_filename, index=False)
        print(f" [INFO] Raw audit logs saved to {log_filename}")

        # 6. SAVE TO HISTORY (Aggregation)
        aggregator.save_to_history()
        print(f" [INFO] Audit finished successfully. Results saved to history.")

    except Exception as e:
        print(f" [ERROR] Error during scheduled audit: {e}")

# --- SCHEDULING SETUP ---

print(f" [INFO] Bias Monitor Scheduler Initialized")
print(f" Target Model: {TARGET_MODEL}")
print(f" Schedule Time: Every day at {RUN_TIME}")

# Job planen
schedule.every().day.at(RUN_TIME).do(run_daily_audit)



# Endlosschleife
while True:
    schedule.run_pending()
    time.sleep(60) # Prüft jede Minute, ob ein Job ansteht