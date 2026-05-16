# ⚡ DataScienceX — ABB EngineeredX 2.0 | Problem Statement #8

An **Adaptive ML Intelligence Platform** that lets anyone — technical or not — upload a dataset, describe their goal in plain English, and get automatic ML analysis with AI-generated insights.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 What It Does

| Feature | Description |
|---|---|
| **Auto Problem Detection** | Detects regression / classification / clustering from your goal text |
| **Multi-Model Training** | Trains 4 models per task, compares them, picks the best |
| **Feature Importance** | Shows which columns drive predictions most |
| **AI Insights** | Generates plain-English explanation of results using Claude |
| **Visualizations** | Correlation heatmap, feature importance chart, cluster plot |
| **Report Export** | Download a JSON report of all findings |

---

## 📊 Supported Tasks

### Regression
Predict a continuous numeric value (e.g. house price, sales, temperature)
- Models: Linear Regression, Ridge, Random Forest, Gradient Boosting
- Metric: R² Score, RMSE

### Classification  
Predict a category or class (e.g. spam/not spam, churn/no churn)
- Models: Logistic Regression, Random Forest, Gradient Boosting, KNN
- Metric: Accuracy, F1 Score

### Clustering
Discover hidden groups in data (no label needed)
- Models: KMeans, Agglomerative Clustering
- Metric: Silhouette Score

---

## 📁 Project Structure

```
ds_assistant/
├── app.py              # Streamlit UI
├── ml_engine.py        # ML pipeline (preprocessing + models)
├── explainer.py        # LLM integration + problem type detection
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 💡 Example Use Cases

- Upload `titanic.csv` → Goal: "Predict passenger survival" → Classification
- Upload `housing.csv` → Goal: "Estimate house prices" → Regression  
- Upload `customers.csv` → Goal: "Group customers by behavior" → Clustering

---

## 🏆 Built For

**ABB EngineeredX 2.0** — Problem Statement #8  
*Design and evaluate data science language models that can adapt to different data science models*

The core idea: an LLM-powered layer that understands a user's intent and routes to the right ML workflow automatically — making data science accessible to domain experts, not just data scientists.
