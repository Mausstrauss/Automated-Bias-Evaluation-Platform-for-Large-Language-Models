"""CSV-backed aggregation layer for bias evaluation results."""

import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

class BiasAggregator:
    """Accumulate scores in-memory and flush them to a simple history CSV."""

    def __init__(self, output_dir="output"):
        self.results_buffer = []
        self.output_dir = output_dir
        self.history_file = os.path.join(output_dir, "evaluation_history.csv")
        
       
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(self.history_file):
            df = pd.DataFrame(columns=["Timestamp", "Model", "Metric", "Category", "Score"])
            df.to_csv(self.history_file, index=False)

    def add_result(self, model_name: str, metric_name: str, bias_category: str, score: float):
        
        entry = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_name,
            "Metric": metric_name,
            "Category": bias_category,
            "Score": round(score, 4)
        }
        self.results_buffer.append(entry)

    def save_to_history(self):
        if not self.results_buffer:
            return

        new_df = pd.DataFrame(self.results_buffer)
        new_df.to_csv(self.history_file, mode='a', header=False, index=False)
        print(f"Results saved to {self.history_file}")

    def get_results_df(self):
        return pd.DataFrame(self.results_buffer)

    def get_history_df(self):
        """Return the persisted history as a DataFrame, or an empty frame on error."""
        try:
            df = pd.read_csv(self.history_file)
            expected = {"Timestamp", "Model", "Metric", "Category", "Score"}
            if not expected.issubset(df.columns):
                return pd.DataFrame(columns=list(expected))
            return df
        except FileNotFoundError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame(columns=["Timestamp", "Model", "Metric", "Category", "Score"])

    def _profile_by_key(self, key: str) -> dict:
        if not self.results_buffer:
            return {}
        buckets: dict = defaultdict(list)
        for res in self.results_buffer:
            buckets[res[key]].append(abs(res["Score"]))
        return {k: np.mean(v) for k, v in buckets.items()}

    def get_profile_by_bias_type(self) -> dict:
        return self._profile_by_key("Category")

    def get_profile_by_metric(self) -> dict:
        """Aggregation by Metric (for Radar Chart)"""
        return self._profile_by_key("Metric")

    def calculate_use_case_score(self, use_case: str = "general") -> float:
        
        weights = {
            "general": {"SentimentDiff": 1.0, "Wasserstein": 1.0},
            "medical": {"SentimentDiff": 0.5, "Wasserstein": 1.0},
            "creative": {"SentimentDiff": 2.0, "Wasserstein": 1.0}
        }
        
        selected_weights = weights.get(use_case, weights["general"])
        
        total_score = 0.0
        total_weight = 0.0
        
        for res in self.results_buffer:
            w = selected_weights.get(res["Metric"], 1.0)
            total_score += abs(res["Score"]) * w
            total_weight += w
            
        if total_weight == 0: return 0.0
        return total_score / total_weight