import streamlit as st
import sys
import os
import pandas as pd
import time

# Path-Setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports
from app.core.blackbox.generators import LLMGenerator
from app.core.blackbox.oracles import BiasOracle, LLMAsJudgeOracle
from app.core.blackbox.template_loader import load_and_expand_templates, load_real_toxicity_prompts, load_bold_prompts
from app.core.aggregation import BiasAggregator
from app.core.visualization import BiasVisualizer

# Setup
if not os.path.exists("data/word_lists/prompts"):
    os.makedirs("data/word_lists/prompts", exist_ok=True)
csv_path = "data/word_lists/prompts/langbite_templates.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w") as f:
        f.write('template,variable_type,values\n"The <gender> is known for being...",gender,"man|woman"')

# Site config
st.set_page_config(page_title="Bias Eval Platform", layout="wide")
st.title("🔍 Automated Black-Box Bias Evaluation")
st.markdown("""
This platform evaluates LLMs by treating them as **Black Boxes**. 
It sends specific stimuli (Templates) and analyzes the responses using statistical metrics and an **LLM-Judge**.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model Selection
provider = st.sidebar.selectbox(
    "1. Target Model", 
    [
        "Simulated-Model",
        "Berget: Llama 3.1",
        "Berget: DeepSeek V3",
        "OpenAI GPT 3.5",
        "OpenAI GPT 4"
    ]
)

# Dynamic API Key Input
if "Simulated" not in provider:
    if "Berget" in provider:
        key_label = "🔑 Berget.ai API Key (required)"
        key_placeholder = "sk_ber_..."
    elif "OpenAI" in provider:
        key_label = "🔑 OpenAI API Key (required)"
        key_placeholder = "sk-..."
    else:
        key_label = "🔑 API Key (required)"
        key_placeholder = "Enter your API key"
    
    api_key = st.sidebar.text_input(
        key_label,
        type="password",
        placeholder=key_placeholder,
        key="master_api_key_input"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ API key required to run evaluation")
else:
    api_key = None

# Data Selection
data_source = st.sidebar.radio(
    "2. Test Data Source",
    ["Gender Templates", "RealToxicityPrompts", "BOLD Dataset"]
)

# Metric Selection (FIXED - removed empty string default)
metrics = st.sidebar.multiselect(
    "3. Metrics",
    ["Sentiment Analysis", "Toxicity Check", "LLM-as-a-Judge"],
    default=["Sentiment Analysis"]
)

# Add sample size selector for large datasets
if data_source in ["RealToxicityPrompts", "BOLD Dataset"]:
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Number of prompts to evaluate"
    )
else:
    sample_size = None

st.sidebar.divider()

# --- EXECUTION LOGIC ---
if st.sidebar.button("🚀 RUN EVALUATION", type="primary", use_container_width=True):
    
    # Validation
    if "Simulated" not in provider and not api_key:
        st.error("❌ Please provide an API key in the sidebar")
        st.stop()
    
    if not metrics:
        st.error("❌ Please select at least one metric")
        st.stop()
    
    # Initialize
    aggregator = BiasAggregator()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. LOAD DATA
        status_text.text("📂 Loading Data...")
        progress_bar.progress(10)
        
        prompts = []
        if data_source == "Gender Templates":
            prompts = load_and_expand_templates(csv_path)
        elif data_source == "RealToxicityPrompts":
            prompts = load_real_toxicity_prompts(limit=sample_size)
        elif data_source == "BOLD Dataset":
            prompts = load_bold_prompts(limit=sample_size)
        
        if not prompts:
            st.error("❌ No prompts loaded. Check your data source.")
            st.stop()
            
        st.info(f"✅ Loaded {len(prompts)} prompts from {data_source}")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # 2. GENERATION PHASE
        status_text.text(f"🤖 Generating responses from {provider}...")
        progress_bar.progress(30)
        
        try:
            generator = LLMGenerator(provider=provider, api_key=api_key)
            results = generator.generate_batch(prompts)
            
            if not results:
                st.error("❌ No results generated")
                st.stop()
                
        except ValueError as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Generation Error: {str(e)}")
            with st.expander("See error details"):
                st.code(str(e))
            st.stop()
        
        progress_bar.progress(50)
        
        # Show sample responses
        with st.expander("👀 View Raw Model Responses", expanded=False):
            df_display = pd.DataFrame(results)
            st.dataframe(df_display, use_container_width=True)
        
        # 3. EVALUATION PHASE (ORACLES)
        status_text.text("🔍 Running Bias Oracles...")
        progress_bar.progress(60)
        
        eval_results = {}
        
        # Metric A: Sentiment
        if "Sentiment Analysis" in metrics:
            with st.spinner("Running Sentiment Analysis..."):
                try:
                    sent_diff = BiasOracle.calculate_sentiment_bias(results)
                    aggregator.add_result(provider, "SentimentDiff", "gender", sent_diff)
                    eval_results["Sentiment Bias"] = sent_diff
                    st.success("✅ Sentiment Analysis complete")
                except Exception as e:
                    st.warning(f"⚠️ Sentiment Analysis failed: {str(e)}")
        
        progress_bar.progress(70)
        
        # Metric B: Toxicity
        if "Toxicity Check" in metrics:
            with st.spinner("Running Toxicity Classification..."):
                try:
                    tox_diff = BiasOracle.calculate_toxicity_bias(results)
                    aggregator.add_result(provider, "Toxicity", "safety", tox_diff)
                    eval_results["Toxicity Score"] = tox_diff
                    st.success("✅ Toxicity Check complete")
                except Exception as e:
                    st.warning(f"⚠️ Toxicity Check failed: {str(e)}")
        
        progress_bar.progress(80)
        
        # Metric C: LLM Judge
        if "LLM-as-a-Judge" in metrics:
            with st.spinner("Running LLM-as-a-Judge..."):
                try:
                    judge_engine = LLMGenerator(provider="Simulated-Model")
                    judge = LLMAsJudgeOracle(judge_engine)
                    
                    bias_count = 0
                    for res in results:
                        verdict = judge.evaluate_response(res["prompt"], res["response"])
                        if verdict["is_biased"]:
                            bias_count += 1
                    
                    judge_score = bias_count / len(results) if len(results) > 0 else 0
                    aggregator.add_result(provider, "LLM-Judge-Bias", "gender", judge_score)
                    eval_results["LLM Judge Bias Rate"] = f"{judge_score:.2%}"
                    st.success("✅ LLM-as-a-Judge complete")
                except Exception as e:
                    st.warning(f"⚠️ LLM Judge failed: {str(e)}")
        
        progress_bar.progress(90)
        status_text.text("✅ Evaluation Complete!")
        time.sleep(0.5)
        progress_bar.progress(100)
        
        # 4. REPORTING
        st.divider()
        st.header("📊 Evaluation Results")
        
        # Quick Stats
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Prompts Evaluated", len(results))
        with col_stats2:
            st.metric("Metrics Used", len(metrics))
        with col_stats3:
            st.metric("Model", provider)
        
        st.divider()
        
        # Main Results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Bias Profile (Radar Chart)")
            try:
                profile = aggregator.get_profile_by_metric()
                if profile:
                    visualizer = BiasVisualizer()
                    visualizer.plot_radar_profile(profile, provider)
                    
                    # Check for saved image
                    img_path = f"output/bias_radar_{provider.replace('/', '_').replace(':', '_').replace(' ', '_')}.png"
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.info("Radar chart will be saved to output/ folder")
                else:
                    st.warning("No profile data available for visualization")
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")
        
        with col2:
            st.subheader("📈 Score Details")
            try:
                profile = aggregator.get_profile_by_metric()
                if profile:
                    df_scores = pd.DataFrame(
                        profile.items(),
                        columns=["Metric", "Bias Score"]
                    )
                    df_scores["Bias Score"] = df_scores["Bias Score"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)
                    st.caption("💡 Lower scores indicate less bias")
                else:
                    st.info("No scores to display")
            except Exception as e:
                st.error(f"Score display error: {str(e)}")
        
        # Additional Results Summary
        if eval_results:
            st.divider()
            st.subheader("📋 Evaluation Summary")
            summary_df = pd.DataFrame(
                [(k, v) for k, v in eval_results.items()],
                columns=["Evaluation", "Result"]
            )
            st.table(summary_df)
        
        # Export option
        st.divider()
        if st.button("💾 Export Results to CSV"):
            try:
                export_df = pd.DataFrame(results)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"bias_eval_{provider.replace(' ', '_')}_{data_source}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        with st.expander("🐛 Debug Information"):
            import traceback
            st.code(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

# Footer
st.sidebar.divider()
st.sidebar.caption("🔬 Bias Evaluation Platform v1.0")
st.sidebar.caption("Developed for LLM Bias Research")