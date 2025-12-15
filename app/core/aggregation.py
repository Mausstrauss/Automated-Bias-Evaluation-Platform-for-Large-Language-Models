import numpy as np
from collections import defaultdict

class BiasAggregator:
    """
    Implements Multi-dimensional Scoring and Aggregation.
    """

    def __init__(self):
        self.results_buffer = []

    def add_result(self, model_name: str, metric_name: str, bias_category: str, score: float):
        self.results_buffer.append({
            "model": model_name,
            "metric": metric_name,
            "category": bias_category,
            "score": score
        })

    def get_profile_by_bias_type(self) -> dict:
        scores_by_cat = defaultdict(list)
        for res in self.results_buffer:
            scores_by_cat[res["category"]].append(abs(res["score"]))
        return {k: np.mean(v) for k, v in scores_by_cat.items()}

    def get_profile_by_metric(self) -> dict:
        scores_by_metric = defaultdict(list)
        for res in self.results_buffer:
            scores_by_metric[res["metric"]].append(abs(res["score"]))
        return {k: np.mean(v) for k, v in scores_by_metric.items()}

    def calculate_use_case_score(self, use_case: str = "general") -> float:
        # Updated Weights: Removed WEAT since we are Black-Box only now
        weights = {
            "general": {"SentimentDiff": 1.0, "Wasserstein": 1.0, "GPTBIAS": 1.5},
            "medical": {"SentimentDiff": 0.5, "Wasserstein": 1.0, "GPTBIAS": 2.0}, 
            "creative": {"SentimentDiff": 2.0, "Wasserstein": 1.0, "GPTBIAS": 0.5}
        }
        
        selected_weights = weights.get(use_case, weights["general"])
        
        total_score = 0.0
        total_weight = 0.0
        
        for res in self.results_buffer:
            w = selected_weights.get(res["metric"], 1.0)
            total_score += abs(res["score"]) * w
            total_weight += w
            
        if total_weight == 0: return 0.0
        return total_score / total_weight