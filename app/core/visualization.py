"""Plotly-based visualization helpers for bias evaluation results."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class BiasVisualizer:
    """Build Plotly figures from the aggregated history data."""

    def create_comparison_bar(self, history_df: pd.DataFrame):
        """Bar chart comparing the latest score per (Model, Metric)."""
        if history_df.empty: return None  # Preserve early exit when there is no history.
        history_df = history_df.copy()  # Work on a copy to avoid mutating caller's DataFrame.
        history_df["Timestamp"] = pd.to_datetime(history_df["Timestamp"], errors="coerce")  # Parse timestamps as datetimes for correct sorting.
        latest_df = history_df.sort_values("Timestamp").groupby(["Model", "Metric"]).tail(1)  # Ensure chronological sort uses true datetime semantics.
        
        fig = px.bar(
            latest_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Model Benchmarking: Direct Comparison",
            color_discrete_sequence=px.colors.qualitative.Bold, 
            text_auto='.2f'
        )
        
        fig.update_layout(yaxis_title="Bias Score (Lower is Better)")
        return fig

    def create_heatmap(self, history_df: pd.DataFrame):
        """Heatmap over (Model, Metric) using mean scores."""
        if history_df.empty: return None
        
        
        pivot_df = history_df.groupby(["Model", "Metric"])["Score"].mean().unstack()
        
        fig = px.imshow(
            pivot_df,
            labels=dict(x="Metric", y="Model", color="Bias Score"),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale="RdYlGn_r", 
            range_color=[0, 0.5], 
            text_auto='.2f',
            title="Bias Heatmap (Red = High Risk)"
        )
        return fig

    def create_trend_line(self, history_df: pd.DataFrame):
        """Time-series line plot of scores (kept for potential future use)."""
        if history_df.empty: return None
        
        fig = px.line(
            history_df, 
            x="Timestamp", 
            y="Score", 
            color="Model", 
            symbol="Metric",
            markers=True,
            title="Bias Evolution over Time"
        )
        return fig