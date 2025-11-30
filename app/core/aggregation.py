# core/aggregation.py
import numpy as np
from collections import defaultdict

class BiasAggregator:
    """
    Implements Multi-dimensional Scoring and Aggregation.
    Avoids single-score reduction; generates Bias Profiles instead.
    """

    def __init__(self):
        self.results_buffer = []

    def add_result(self, model_name: str, metric_name: str, bias_category: str, score: float):
        """
        Ingests a single evaluation result.
        :param metric_name: e.g., 'WEAT', 'Wasserstein', 'GPTBIAS'
        :param bias_category: e.g., 'gender', 'race', 'religion'
        """
        self.results_buffer.append({
            "model": model_name,
            "metric": metric_name,
            "category": bias_category,
            "score": score
        })

    def get_profile_by_bias_type(self) -> dict:
        """
        Profile 1: Aggregation by Bias Type.
        Returns a vector of scores averaged across all metrics for each category.
        Example: {'gender': 0.15, 'race': 0.05}
        """
        scores_by_cat = defaultdict(list)
        for res in self.results_buffer:
            # We use absolute values because for WEAT, -0.5 is just as biased as +0.5
            scores_by_cat[res["category"]].append(abs(res["score"]))
            
        return {k: np.mean(v) for k, v in scores_by_cat.items()}

    def get_profile_by_metric(self) -> dict:
        """
        Profile 2: Aggregation by Metric Type.
        Useful for comparing White-box vs Black-box performance.
        """
        scores_by_metric = defaultdict(list)
        for res in self.results_buffer:
            scores_by_metric[res["metric"]].append(abs(res["score"]))
            
        return {k: np.mean(v) for k, v in scores_by_metric.items()}

    def calculate_use_case_score(self, use_case: str = "general") -> float:
        """
        Profile 3: Aggregation by Use Case (Weighted Scoring).
        Applies specific weights depending on the deployment scenario.
        """
        # Define Weight Templates [cite: 237]
        weights = {
            "general": {"WEAT": 1.0, "Wasserstein": 1.0, "GPTBIAS": 1.0},
            "medical": {"WEAT": 0.2, "Wasserstein": 0.5, "GPTBIAS": 2.0}, # EquityMedQA importance [cite: 238]
            "creative": {"WEAT": 0.5, "Wasserstein": 2.0, "GPTBIAS": 1.0}
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