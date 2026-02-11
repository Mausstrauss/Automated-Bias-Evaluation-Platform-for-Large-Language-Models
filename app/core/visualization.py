import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class BiasVisualizer:
    

    def create_comparison_bar(self, history_df: pd.DataFrame):
        
        if history_df.empty: return None
        
        
        latest_df = history_df.sort_values("Timestamp").groupby(["Model", "Metric"]).tail(1)
        
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