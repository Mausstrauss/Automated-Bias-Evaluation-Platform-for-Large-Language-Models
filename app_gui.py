import streamlit as st
import sys
import os
import pandas as pd

# Add your project to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.whitebox.model_wrapper import HFModelWrapper
from app.core.whitebox.weat import calculate_weat_effect_size
from app.core.aggregation import BiasAggregator
from app.core.visualization import BiasVisualizer

# --- TITLE ---
st.title("Bias Evaluation Platform ")
st.write("Automated continuous evaluation of LLM bias.")

# --- SIDEBAR (CONTROLS) ---
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select White-Box Model",
    ["microsoft/Phi-3-mini-4k-instruct", "bert-base-german-cased", "meta-llama/Meta-Llama-3-8B"]
)
provider = st.sidebar.selectbox("Select Black-Box Provider", ["Simulated-Model", "OpenAI", "Deepseek"])

if st.sidebar.button("RUN EVALUATION 🚀"):
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    aggregator = BiasAggregator()

    # --- PHASE 1 ---
    status_text.text(f"Phase 1: Loading {model_name}...")
    try:
        wrapper = HFModelWrapper(model_name)
        # (Simplified logic for demo)
        t1 = wrapper.get_embeddings_for_list(["Man", "Father"])
        t2 = wrapper.get_embeddings_for_list(["Woman", "Mother"])
        a1 = wrapper.get_embeddings_for_list(["Career", "Boss"])
        a2 = wrapper.get_embeddings_for_list(["Family", "Home"])
        score = calculate_weat_effect_size(t1, t2, a1, a2)
        aggregator.add_result(model_name, "WEAT", "gender", score)
        st.success(f"Phase 1 Complete: WEAT Score = {score:.4f}")
    except Exception as e:
        st.error(f"White-Box Failed: {e}")
    
    progress_bar.progress(50)

    # --- PHASE 2 ---
    status_text.text("Phase 2: Running Black-Box Oracles...")
    # (Simplified logic - assuming simulated)
    import random
    sent_score = random.uniform(0.1, 0.5) # Placeholder for real generator
    aggregator.add_result(provider, "Sentiment", "gender", sent_score)
    st.success(f"Phase 2 Complete: Sentiment Score = {sent_score:.4f}")
    
    progress_bar.progress(100)
    
    # --- VISUALIZATION ---
    st.header("Bias Profile Results")
    profile = aggregator.get_profile_by_bias_type()
    
    # Use the visualizer to create the plot
    visualizer = BiasVisualizer()
    visualizer.plot_radar_profile(profile, model_name)
    
    # Display the saved image in the browser
    filename = f"output/bias_radar_{model_name.replace('/', '_')}.png"
    if os.path.exists(filename):
        st.image(filename, caption="Multi-Axis Bias Profile")
    
    # Also show raw data as a table
    st.table(pd.DataFrame(profile.items(), columns=["Category", "Bias Score"]))