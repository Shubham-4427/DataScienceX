import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import json
import requests
import warnings
warnings.filterwarnings('ignore')

from ml_engine import run_ml_pipeline
from explainer import get_llm_explanation, detect_problem_type

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataScienceX | ABB EngineeredX 2.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0d0d0d;
    color: #e8e8e8;
}

.main-header {
    background: linear-gradient(135deg, #ff6b00 0%, #ff9a00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.8rem;
    font-weight: 600;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.sub-header {
    color: #888;
    font-size: 0.95rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #ff6b00;
    border-radius: 4px;
    padding: 16px 20px;
    margin: 8px 0;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #ff6b00;
}

.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.insight-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 20px;
    margin: 12px 0;
    line-height: 1.7;
    color: #ccc;
}

.tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #ff6b00;
    color: #ff6b00;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 2px;
    margin: 2px;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #ff6b00;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

div[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #1e1e1e;
}

.stButton > button {
    background: #ff6b00 !important;
    color: #000 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    border-radius: 3px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #ff9a00 !important;
    transform: translateY(-1px);
}

.stTextArea textarea, .stSelectbox select {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.model-winner {
    background: linear-gradient(135deg, #1a0f00, #1e1200);
    border: 1px solid #ff6b00;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ DataScienceX")
    st.markdown("<p style='color:#888; font-size:0.8rem; font-family:IBM Plex Mono'>ABB EngineeredX 2.0</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("**How it works:**")
    st.markdown("""
    <div style='color:#aaa; font-size:0.85rem; line-height:1.8'>
    1️⃣ Upload your CSV dataset<br>
    2️⃣ Describe your goal<br>
    3️⃣ Select your target column<br>
    4️⃣ Let AI detect & run the best ML model<br>
    5️⃣ Get plain-English insights
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("**Supported Tasks:**")
    for task in ["📈 Regression", "🎯 Classification", "🔵 Clustering"]:
        st.markdown(f"<span class='tag'>{task}</span>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("<p style='color:#555; font-size:0.75rem'>Built for ABB EngineeredX 2.0<br>Problem Statement #8</p>", unsafe_allow_html=True)

# ─── Main Content ───────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>DataScienceX</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Adaptive ML Intelligence Platform</p>", unsafe_allow_html=True)
st.markdown("")

# ─── Upload Section ─────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>01 — Load Dataset</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your CSV file here",
    type=["csv"],
    help="Upload any structured dataset in CSV format"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df.shape[0]:,}</div>
            <div class='metric-label'>Rows</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{df.shape[1]}</div>
            <div class='metric-label'>Columns</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        missing = df.isnull().sum().sum()
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{missing}</div>
            <div class='metric-label'>Missing Values</div>
        </div>""", unsafe_allow_html=True)
    
    with st.expander("👁 Preview Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # ─── Goal Input ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>02 — Describe Your Goal</div>", unsafe_allow_html=True)
    
    user_goal = st.text_area(
        "What do you want to find out from this data?",
        placeholder="e.g. 'Predict house prices based on features' or 'Classify whether a customer will churn' or 'Group customers by behavior'",
        height=80
    )
    
    # ─── Column Selection ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>03 — Configure</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Target Column (what to predict)",
            options=["-- None (for clustering) --"] + list(df.columns),
        )
    with col2:
        problem_override = st.selectbox(
            "Problem Type (or let AI decide)",
            options=["🤖 Auto-detect", "📈 Regression", "🎯 Classification", "🔵 Clustering"]
        )
    
    target = None if target_col == "-- None (for clustering) --" else target_col
    
    # ─── Run Button ──────────────────────────────────────────────────────────
    st.markdown("")
    run_btn = st.button("⚡ Run Analysis", use_container_width=False)
    
    if run_btn:
        if not user_goal.strip():
            st.warning("Please describe your goal first.")
        else:
            # Detect problem type
            with st.spinner("🤖 Detecting problem type..."):
                if problem_override == "🤖 Auto-detect":
                    problem_type = detect_problem_type(user_goal, df, target)
                else:
                    mapping = {"📈 Regression": "regression", "🎯 Classification": "classification", "🔵 Clustering": "clustering"}
                    problem_type = mapping[problem_override]
            
            st.markdown(f"""
            <div style='margin:12px 0'>
                <span style='color:#888; font-size:0.85rem'>Detected Task → </span>
                <span class='tag'>{'📈 REGRESSION' if problem_type=='regression' else '🎯 CLASSIFICATION' if problem_type=='classification' else '🔵 CLUSTERING'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Run ML
            with st.spinner("⚙️ Training models and evaluating..."):
                results = run_ml_pipeline(df, target, problem_type)
            
            if "error" in results:
                st.error(f"❌ {results['error']}")
            else:
                # ─── Results ─────────────────────────────────────────────
                st.markdown("<div class='section-title'>04 — Results</div>", unsafe_allow_html=True)
                
                # Best model highlight
                best = results["best_model"]
                st.markdown(f"""
                <div class='model-winner'>
                    <div style='color:#ff6b00; font-family:IBM Plex Mono; font-size:0.75rem; letter-spacing:2px'>BEST MODEL</div>
                    <div style='font-size:1.5rem; font-weight:600; margin:4px 0'>{best['name']}</div>
                    <div style='color:#aaa; font-size:0.9rem'>{best['primary_metric_name']}: <span style='color:#ff6b00; font-family:IBM Plex Mono'>{best['primary_metric_value']:.4f}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Model comparison table
                if len(results["all_models"]) > 1:
                    with st.expander("📊 All Models Compared", expanded=True):
                        model_df = pd.DataFrame(results["all_models"])
                        st.dataframe(model_df, use_container_width=True, hide_index=True)
                
                # Charts
                st.markdown("<div class='section-title'>05 — Visualizations</div>", unsafe_allow_html=True)
                
                chart_cols = st.columns(2)
                
                # Feature importance / cluster plot
                if "feature_importance" in results:
                    with chart_cols[0]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_facecolor('#1a1a1a')
                        ax.set_facecolor('#1a1a1a')
                        fi = results["feature_importance"]
                        top_n = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                        names, vals = zip(*top_n)
                        colors = ['#ff6b00' if i == 0 else '#cc5500' if i < 3 else '#663300' for i in range(len(names))]
                        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
                        ax.set_title("Feature Importance", color='#e8e8e8', fontsize=11, pad=10)
                        ax.tick_params(colors='#888', labelsize=8)
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#2a2a2a')
                        ax.xaxis.label.set_color('#888')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
                # Correlation heatmap
                with chart_cols[1]:
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) >= 2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_facecolor('#1a1a1a')
                        ax.set_facecolor('#1a1a1a')
                        corr = numeric_df.corr()
                        mask = np.triu(np.ones_like(corr, dtype=bool))
                        cmap = sns.diverging_palette(20, 220, as_cmap=True)
                        sns.heatmap(corr, mask=mask, cmap=cmap, ax=ax, 
                                   annot=len(corr) <= 8, fmt='.2f',
                                   linewidths=0.5, linecolor='#2a2a2a',
                                   cbar_kws={'shrink': 0.8})
                        ax.set_title("Correlation Matrix", color='#e8e8e8', fontsize=11, pad=10)
                        ax.tick_params(colors='#888', labelsize=7)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                
                # Cluster scatter if clustering
                if "cluster_plot_data" in results:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fig.patch.set_facecolor('#1a1a1a')
                    ax.set_facecolor('#1a1a1a')
                    plot_data = results["cluster_plot_data"]
                    scatter = ax.scatter(plot_data["x"], plot_data["y"], 
                                        c=plot_data["labels"], cmap='plasma', 
                                        alpha=0.7, s=30)
                    ax.set_title(f"Cluster Visualization ({results['best_model']['name']})", 
                                color='#e8e8e8', fontsize=11, pad=10)
                    ax.tick_params(colors='#888')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#2a2a2a')
                    plt.colorbar(scatter, ax=ax).ax.tick_params(colors='#888')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # ─── AI Explanation ──────────────────────────────────────
                st.markdown("<div class='section-title'>06 — AI Insights</div>", unsafe_allow_html=True)
                
                with st.spinner("🧠 Generating insights..."):
                    explanation = get_llm_explanation(
                        user_goal=user_goal,
                        problem_type=problem_type,
                        results=results,
                        df_info={
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                            "missing": int(df.isnull().sum().sum())
                        }
                    )
                
                st.markdown(f"<div class='insight-box'>{explanation}</div>", unsafe_allow_html=True)
                
                # ─── Download ────────────────────────────────────────────
                st.markdown("<div class='section-title'>07 — Export</div>", unsafe_allow_html=True)
                
                report = {
                    "goal": user_goal,
                    "problem_type": problem_type,
                    "best_model": results["best_model"],
                    "all_models": results["all_models"],
                    "ai_insights": explanation
                }
                
                st.download_button(
                    "⬇ Download Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name="datasciencex_report.json",
                    mime="application/json"
                )

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#444'>
        <div style='font-size:4rem; margin-bottom:16px'>⚡</div>
        <div style='font-family: IBM Plex Mono; font-size:1rem; color:#555'>
            Upload a CSV dataset to begin
        </div>
        <div style='margin-top:32px; display:flex; justify-content:center; gap:16px; flex-wrap:wrap'>
            <span class='tag'>REGRESSION</span>
            <span class='tag'>CLASSIFICATION</span>  
            <span class='tag'>CLUSTERING</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
