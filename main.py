"""CLI entry point to run a single black-box audit without the GUI."""

import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle
from app.core.aggregation import BiasAggregator

def run_blackbox_pipeline():
    """Execute one end-to-end black-box evaluation run for a single model."""
    print("\n--- Starting Black-Box Evaluation Pipeline ---")
    
   
    aggregator = BiasAggregator()
    model_name = os.getenv("TARGET_MODEL", "OpenAI-GPT3.5")
    
 
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

   
    print(f"Generating responses for {len(prompts)} prompts...")
    generator = LLMGenerator(provider=model_name)
    results = generator.generate_batch(prompts)
    
   
    print("-> Calculating Toxicity Bias (loading model)...")
    tox_diff = BiasOracle.calculate_toxicity_bias(results)
    aggregator.add_result(model_name, "Toxicity", "safety", tox_diff)
    print(f"   Toxicity Difference: {tox_diff:.4f}")
    
   
    sent_diff = BiasOracle.calculate_sentiment_bias(results)
    aggregator.add_result(model_name, "SentimentDiff", "gender", sent_diff)
    print(f"-> Sentiment Difference Score: {sent_diff:.4f}")
     
    print("Calculating Wasserstein Distance...")  # FIX: Keep existing log message for continuity.

   
    from collections import Counter  # FIX: Import Counter locally to avoid changing module-level imports.
    var_counts = Counter(r.get("variable") for r in results)  # FIX: Count occurrences of each variable value in a data-driven way.
    top_two = [v for v, _ in var_counts.most_common(2) if v is not None]  # FIX: Select the two most common non-null variable values.
    if len(top_two) == 2:  # FIX: Only compute Wasserstein when at least two groups are present.
        group_a_text = [r["response"] for r in results if r.get("variable") == top_two[0]]  # FIX: Gather responses for the first dominant group.
        group_b_text = [r["response"] for r in results if r.get("variable") == top_two[1]]  # FIX: Gather responses for the second dominant group.
    else:
        group_a_text, group_b_text = [], []  # FIX: Fall back to empty groups if we cannot identify two distinct variables.

    if group_a_text and group_b_text:  # FIX: Preserve behavior: only compute metric when both groups have data.
        w_score = BiasOracle.calculate_wasserstein_metric(group_a_text, group_b_text)  # FIX: Reuse existing Wasserstein implementation.
        aggregator.add_result(model_name, "Wasserstein", "gender", w_score)  # FIX: Keep same metric and category labels.
        print(f"-> Wasserstein Distance: {w_score:.4f}")  # FIX: Preserve informational output.
    else:
        print("-> Skipped Wasserstein (insufficient data for comparison)")  # FIX: Preserve skip message when we cannot form two groups.
    
   
    print("\n--- Final Report ---")
    profile = aggregator.get_profile_by_metric()
    print("Bias Profile:", profile)
    

if __name__ == "__main__":
    run_blackbox_pipeline()