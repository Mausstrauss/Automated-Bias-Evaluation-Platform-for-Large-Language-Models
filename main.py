"""
CLI entry point for one-off headless bias audits without the GUI.

Usage:
    python main.py

Note: Requires a valid API key set in the environment (OPENAI_API_KEY or GOOGLE_API_KEY).
Update the model_name variable to match your available provider:
  "OpenAI-GPT3.5" — requires OPENAI_API_KEY
  "Google-Gemini" — requires GOOGLE_API_KEY
"""

from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle
from app.core.blackbox.template_loader import load_and_expand_templates
from app.core.aggregation import BiasAggregator


def run_blackbox_pipeline():
    csv_path = "data/word_lists/prompts/langbite_templates.csv"
    prompts = load_and_expand_templates(csv_path)

    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    model_name = "OpenAI-GPT3.5"
    generator = LLMGenerator(provider=model_name, api_key="")
    results = generator.generate_batch(prompts)

    enriched = []
    for res in results:
        text = res.get("response", "")
        res["sentiment_score"] = BiasOracle.analyze_sentiment(text)
        res["toxicity_score"] = BiasOracle.analyze_toxicity_bert(text)
        enriched.append(res)

    sent_diff = BiasOracle.calculate_sentiment_bias(enriched)
    tox_diff = BiasOracle.calculate_toxicity_bias(enriched)

    aggregator = BiasAggregator()
    aggregator.add_result(model_name, "SentimentDiff", "gender", sent_diff)
    aggregator.add_result(model_name, "Toxicity", "safety", tox_diff)
    aggregator.save_to_history()

    print(f"SentimentDiff: {sent_diff:.4f}")
    print(f"ToxicityDiff:  {tox_diff:.4f}")
    print("Results saved to output/evaluation_history.csv")


if __name__ == "__main__":
    run_blackbox_pipeline()
