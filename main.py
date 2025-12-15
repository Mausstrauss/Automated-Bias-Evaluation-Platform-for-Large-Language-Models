import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NUR Black-Box Imports
from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle
from app.core.blackbox.oracles import LLMAsJudgeOracle
from app.core.aggregation import BiasAggregator

def run_blackbox_pipeline():
    print("\n--- Starting Black-Box Evaluation Pipeline ---")
    
    # 1. SETUP
    aggregator = BiasAggregator()
    model_name = "Simulated-GPT"
    
    # 2. LOAD DATA
    # Wir nutzen LangBite Templates
    csv_path = "data/word_lists/prompts/langbite_templates.csv"
    if not os.path.exists(csv_path):
        print(f"Creating dummy data at {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write('template,variable_type,values\n"The <gender> is...",gender,"man|woman"')
            
    prompts = load_and_expand_templates(csv_path)
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    # 3. GENERATION 
    print(f"Generating responses for {len(prompts)} prompts...")
    generator = LLMGenerator(provider=model_name)
    results = generator.generate_batch(prompts)
    
    # 4. ORACLES 
    
    print("-> Calculating Toxicity Bias (loading model)...")
    tox_diff = BiasOracle.calculate_toxicity_bias(results)
    aggregator.add_result(model_name, "Toxicity", "safety", tox_diff)
    print(f"   Toxicity Difference: {tox_diff:.4f}")
    
    # Oracle A: Sentiment Analysis
    sent_diff = BiasOracle.calculate_sentiment_bias(results)
    aggregator.add_result(model_name, "SentimentDiff", "gender", sent_diff)
    print(f"-> Sentiment Difference Score: {sent_diff:.4f}")
     # Metric C: Wasserstein Distance (Distributional Bias)
    print("Calculating Wasserstein Distance...")

    # Filter results into two groups (man vs woman) based on the 'variable' field
    group_a_text = [r["response"] for r in results if r["variable"] == "man"]
    group_b_text = [r["response"] for r in results if r["variable"] == "woman"]

    if group_a_text and group_b_text:
        # Calculate distance
        w_score = BiasOracle.calculate_wasserstein_metric(group_a_text, group_b_text)
        aggregator.add_result(model_name, "Wasserstein", "gender", w_score)
        print(f"-> Wasserstein Distance: {w_score:.4f}")
    else:
        print("-> Skipped Wasserstein (insufficient data for comparison)")
    # Oracle B: LLM-as-a-Judge
    # In Produktion: Hier nimmst du GPT-4 als Judge
    judge_engine = LLMGenerator(provider="Simulated-Judge") 
    judge_oracle = LLMAsJudgeOracle(judge_engine)
    
    print("Running LLM-as-a-Judge on responses...")
    biased_count = 0
    for res in results:
        # Wir prüfen jede Antwort auf Bias
        verdict = judge_oracle.evaluate_response(res["prompt"], res["response"])
        if verdict["is_biased"]:
            biased_count += 1
            print(f"   [Flagged]: {res['response']} -> Reason: {verdict['reason']}")
            
    # Score berechnen (Ratio of biased responses)
    judge_score = biased_count / len(results)
    aggregator.add_result(model_name, "GPTBIAS", "gender", judge_score)
    print(f"-> Judge Bias Score: {judge_score:.4f}")

    # 5. AGGREGATION & REPORTING
    print("\n--- Final Report ---")
    profile = aggregator.get_profile_by_metric()
    print("Bias Profile:", profile)
    
    # Hier würde man jetzt das PDF oder HTML generieren
    # visualizer.plot...

if __name__ == "__main__":
    run_blackbox_pipeline()