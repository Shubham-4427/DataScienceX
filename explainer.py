import requests
import json
import pandas as pd
import numpy as np


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"


def detect_problem_type(user_goal: str, df: pd.DataFrame, target: str = None) -> str:
    """Use heuristics + LLM to detect problem type."""

    goal_lower = user_goal.lower()

    # Quick keyword heuristics first
    cluster_words = ["group", "cluster", "segment", "categorize", "pattern", "find groups", "unsupervised"]
    class_words = ["classify", "predict whether", "detect", "identify", "is it", "will it", "churn", "spam", "fraud", "binary", "category"]
    reg_words = ["predict", "forecast", "estimate", "how much", "how many", "price", "sales", "revenue", "value", "amount"]

    if any(w in goal_lower for w in cluster_words) or not target:
        return "clustering"

    if target:
        # Check target column cardinality
        unique_vals = df[target].nunique()
        if unique_vals <= 10:
            return "classification"
        elif df[target].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if unique_vals > 20:
                return "regression"

    if any(w in goal_lower for w in class_words):
        return "classification"
    if any(w in goal_lower for w in reg_words):
        return "regression"

    return "classification"  # safe default


def get_llm_explanation(user_goal: str, problem_type: str, results: dict, df_info: dict) -> str:
    """Call Claude API to generate plain-English insights."""

    best = results.get("best_model", {})
    all_models = results.get("all_models", [])
    fi = results.get("feature_importance", {})
    n_clusters = results.get("n_clusters", None)

    fi_text = ""
    if fi:
        top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        fi_text = "Top features by importance: " + ", ".join([f"{k} ({v:.3f})" for k, v in top_features])

    prompt = f"""You are a senior data scientist presenting results to a non-technical business stakeholder.

USER GOAL: {user_goal}

DATASET INFO:
- Rows: {df_info['shape'][0]}, Columns: {df_info['shape'][1]}
- Missing values: {df_info['missing']}
- Columns: {', '.join(df_info['columns'][:15])}

PROBLEM TYPE DETECTED: {problem_type}

MODEL RESULTS:
- Best Model: {best.get('name', 'N/A')}
- Best Score ({best.get('primary_metric_name', '')}): {best.get('primary_metric_value', 'N/A')}
- All Models Compared: {json.dumps(all_models, indent=2)}
{f'- {fi_text}' if fi_text else ''}
{f'- Number of clusters found: {n_clusters}' if n_clusters else ''}

Write a clear, insightful explanation (4-6 sentences) that:
1. Summarizes what the model found / achieved
2. Explains the performance in plain English (what does the score mean?)
3. Highlights which features matter most (if available)
4. Gives 1-2 actionable recommendations based on the results
5. Mentions any caveats or next steps

Use clear language, no jargon. Make it feel like a smart colleague explaining findings, not a robot. Format as flowing paragraphs — no bullet points."""

    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": ""  # handled by proxy
            },
            json={
                "model": MODEL,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"]
        else:
            return _fallback_explanation(user_goal, problem_type, results)

    except Exception as e:
        return _fallback_explanation(user_goal, problem_type, results)


def _fallback_explanation(user_goal: str, problem_type: str, results: dict) -> str:
    """Fallback if API call fails."""
    best = results.get("best_model", {})
    name = best.get("name", "the model")
    metric = best.get("primary_metric_name", "score")
    value = best.get("primary_metric_value", 0)

    if problem_type == "regression":
        quality = "excellent" if value > 0.85 else "good" if value > 0.65 else "moderate"
        return (
            f"The analysis ran {problem_type} to address your goal: '{user_goal}'. "
            f"{name} performed best with an R² of {value:.4f}, indicating {quality} predictive power — "
            f"meaning the model explains {value*100:.1f}% of the variance in your target variable. "
            f"Consider collecting more data or engineering additional features to improve performance further."
        )
    elif problem_type == "classification":
        quality = "strong" if value > 0.85 else "reasonable" if value > 0.65 else "moderate"
        return (
            f"The analysis ran classification to address your goal: '{user_goal}'. "
            f"{name} achieved an F1 score of {value:.4f}, which represents {quality} performance. "
            f"This means the model correctly identifies the target class with good precision and recall. "
            f"Review misclassified examples to identify patterns that could help improve the model further."
        )
    else:
        n = results.get("n_clusters", "several")
        return (
            f"The clustering analysis of your data found {n} natural groupings. "
            f"{name} achieved a silhouette score of {value:.4f} (closer to 1.0 is better), "
            f"indicating {'well-separated' if value > 0.5 else 'reasonably distinct'} clusters. "
            f"Explore the characteristics of each cluster to understand what makes them different."
        )
