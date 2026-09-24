# 🎬 Netflix Customer Churn & Engagement Analytics Using AI

> **IBM SkillsBuild | AICTE AI Internship Programme — 2025**
> An end-to-end machine learning project to predict subscriber churn, quantify engagement, and surface actionable retention strategies for a Netflix-style streaming platform.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [ML Approach](#-ml-approach)
- [Key Outputs](#-key-outputs)
- [Results & Findings](#-results--findings)
- [Business Recommendations](#-business-recommendations)
- [Future Scope](#-future-scope)
- [License](#-license)

---

## 🔍 Project Overview

This project applies a complete, reproducible machine learning pipeline to predict customer churn on a 5,000-subscriber Netflix dataset. It covers every stage of the data science lifecycle — from raw data ingestion and preprocessing, through feature engineering, exploratory data analysis, and model training, to evaluation, interpretation, and business insight generation.

Six supervised classifiers are trained and benchmarked. The best-performing model — **Gradient Boosting** — achieves a **ROC-AUC of 0.998** and an **F1-score of 0.989**, demonstrating near-perfect discrimination between churned and retained customers. Feature importance analysis consistently identifies `avg_watch_time_per_day`, a composite `engagement_score`, `watch_hours`, and `last_login_days` as the primary churn drivers.

All analysis is delivered in a single, fully executable **Jupyter Notebook** (`Netflix_Churn_Analytics.ipynb`) accompanied by a comprehensive academic project report (`Netflix_Churn_Report.docx`).

---

## ❗ Problem Statement

Customer churn — the voluntary cancellation of a streaming subscription — directly erodes revenue and increases the cost of re-acquisition. Without proactive early-warning systems, platforms respond only after a customer has already decided to leave.

> **Core question:** Given demographic, behavioural, and transactional subscriber data, can a machine learning model reliably identify at-risk customers *before* they churn — and which factors most strongly drive that risk?

Three inter-related problems are addressed:

1. **Binary classification** of churn vs. retention from structured subscriber data.
2. **Feature importance ranking** to identify the strongest predictors of churn.
3. **Business recommendation synthesis** to translate analytical findings into prioritised retention actions.

---

## 🎯 Objectives

| # | Objective |
|---|-----------|
| O1 | Perform comprehensive EDA to uncover behavioural patterns distinguishing churned vs. retained customers |
| O2 | Engineer six derived features (engagement score, inactivity flag, age group, watch segment, etc.) to enhance model accuracy |
| O3 | Train and evaluate six supervised ML classifiers using stratified splitting and 5-fold cross-validation |
| O4 | Identify top churn predictors via Random Forest, Gradient Boosting, and Logistic Regression importance analyses |
| O5 | Derive ten prioritised business recommendations within an Impact vs. Effort priority matrix |
| O6 | Demonstrate proficiency in the full Python data science stack: pandas, NumPy, Matplotlib, seaborn, scikit-learn |
| O7 | Produce a reproducible Jupyter Notebook executable from start to finish with a single kernel restart |

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **File** | `netflix_customer_churn.csv` |
| **Records** | 5,000 subscribers |
| **Features** | 14 (13 input features + 1 binary target) |
| **Target** | `churned` — 0 = Retained, 1 = Churned |
| **Class balance** | ~50/50 (49.7% retained / 50.3% churned) |
| **Missing values** | None |
| **Source** | Synthetic dataset designed for educational churn analytics |

### Feature Summary

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Integer | Customer age (18–70) |
| `gender` | Categorical | Male / Female / Other |
| `subscription_type` | Categorical | Basic / Standard / Premium |
| `watch_hours` | Float | Total watch hours in observation window |
| `last_login_days` | Integer | Days since most recent login (0–90) |
| `region` | Categorical | Africa, Asia, Europe, North America, Oceania, South America |
| `device` | Categorical | TV / Mobile / Desktop / Laptop / Tablet |
| `monthly_fee` | Float | Subscription fee in USD (8.99 / 13.99 / 17.99) |
| `payment_method` | Categorical | Credit Card / Debit Card / PayPal / Gift Card / Crypto |
| `number_of_profiles` | Integer | Profiles on account (1–5) |
| `avg_watch_time_per_day` | Float | Average daily watch time in hours |
| `favorite_genre` | Categorical | Most-watched genre |
| `churned` | Binary | **Target** — 0: Retained, 1: Churned |

---

## 🛠 Technologies Used

| Library | Version | Role |
|---------|---------|------|
| Python | 3.13 | Core language |
| pandas | 3.0.6 | Data loading, manipulation, aggregation |
| NumPy | 2.4.2 | Numerical computation |
| Matplotlib | 3.11.2 | Static and composite visualisation |
| seaborn | 0.13.2 | Statistical visualisation (KDE, heatmaps, box plots) |
| scikit-learn | 1.9.1 | ML models, preprocessing, evaluation metrics |
| Jupyter Notebook | 7.6.3 | Interactive development environment |

---

## 📁 Project Structure

```
IBM Project/
│
├── netflix_customer_churn.csv        # Raw dataset (5,000 subscribers, 14 features)
├── Netflix_Churn_Analytics.ipynb     # Main Jupyter Notebook (end-to-end pipeline)
├── Netflix_Churn_Report.docx         # Academic project report (18 sections)
├── requirements.txt                  # Python dependency pinlist
└── README.md                         # This file
```

> **Output artifact:** Running the notebook also generates `netflix_churn_dashboard.png` — an executive summary dashboard exported as a high-resolution PNG.

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.9 or higher**
- `pip` package manager

### 1. Clone or download the repository

```bash
git clone https://github.com/<your-username>/netflix-churn-analytics.git
cd netflix-churn-analytics
```

Or simply place all project files in a single folder.

### 2. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs exactly:

```
numpy==2.4.2
pandas==3.0.6
matplotlib==3.11.2
seaborn==0.13.2
scikit-learn==1.9.1
notebook==7.6.3
```

---

## ▶️ How to Run

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open `Netflix_Churn_Analytics.ipynb` in the browser interface that appears.

### Execute the full pipeline

In the notebook menu:

```
Kernel → Restart Kernel and Run All Cells
```

The notebook runs **top to bottom without manual intervention**. All cells are self-contained and sequentially dependent.

### Expected runtime

| Phase | Approximate Time |
|-------|-----------------|
| Data loading & EDA | < 10 seconds |
| Feature engineering | < 5 seconds |
| Model training (all 6) | 30–90 seconds |
| Cross-validation | 20–60 seconds |
| All visualisations | < 15 seconds |
| **Total** | **~2–3 minutes** |

> ⚠️ SVM and Gradient Boosting (200 estimators) are the most compute-intensive steps. On older hardware, total runtime may extend to 5–7 minutes.

---

## 🤖 ML Approach

### Pipeline Architecture

```
Raw CSV
  │
  ├─ Data Cleaning (drop ID, strip whitespace)
  ├─ EDA (7 visualisation blocks)
  ├─ Feature Engineering (6 derived features)
  │     ├── engagement_score
  │     ├── high_watcher
  │     ├── inactive_30d
  │     ├── age_group
  │     ├── fee_per_profile
  │     └── watch_segment
  ├─ Label Encoding (8 categorical columns)
  ├─ Stratified Train/Test Split (80/20, seed=42)
  ├─ StandardScaler (for LR, SVM, KNN)
  │
  ├─ Model Training
  │     ├── Logistic Regression      (linear baseline)
  │     ├── Decision Tree            (max_depth=6)
  │     ├── Random Forest            (200 trees, max_depth=10)
  │     ├── Gradient Boosting        (200 estimators, lr=0.08)
  │     ├── SVM RBF                  (C=1.0, probability=True)
  │     └── K-Nearest Neighbours     (k=7)
  │
  ├─ Evaluation
  │     ├── Accuracy, Precision, Recall, F1, ROC-AUC
  │     ├── ROC curves (all 6 models overlaid)
  │     ├── Confusion matrices (top 2 models)
  │     └── 5-fold cross-validation
  │
  └─ Interpretation
        ├── Random Forest feature importance
        ├── Gradient Boosting feature importance
        └── Logistic Regression coefficient magnitudes
```

### Experimental Design

- **Split:** Stratified 80/20 — preserves 50.3% churn rate in both train and test sets
- **Scaling:** StandardScaler applied only to scale-sensitive algorithms (LR, SVM, KNN); tree models use raw unscaled features
- **Validation:** 5-fold stratified cross-validation on full dataset for generalisation check
- **Random seed:** `42` throughout for full reproducibility

---

## 📈 Key Outputs

The notebook generates the following outputs:

| Output | Description |
|--------|-------------|
| Churn distribution charts | Bar + pie chart of retained vs. churned customers |
| Histogram overlays | Numerical feature distributions split by churn status |
| Box plots | Median/IQR comparison of 6 numerical features vs. churn |
| Churn rate bar charts | Churn % for every categorical feature (6 charts) |
| Correlation heatmap | Pearson correlation matrix of numerical features + target |
| KDE plot | Days-since-login density by churn status |
| Engineered feature charts | Engagement score, churn by age group, churn by watch segment |
| Subscription × Device heatmap | Churn rate cross-tabulation |
| Region × Subscription heatmap | Regional churn patterns by tier |
| Genre dual-axis chart | Churn rate and average watch hours by favourite genre |
| Model comparison bar chart | All 5 metrics across all 6 models |
| ROC curve overlay | All 6 ROC curves on one plot with AUC labels |
| Confusion matrices | Top-2 models side by side |
| Feature importance charts | RF horizontal bar, GB top-12 bar, LR coefficient magnitudes |
| Priority matrix scatter | Recommendation Impact vs. Effort quadrant chart |
| Executive dashboard PNG | `netflix_churn_dashboard.png` — 6-panel summary figure |

---

## 📊 Results & Findings

### Model Performance (Test Set, n = 1,000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| **Gradient Boosting** ⭐ | **0.989** | **0.990** | **0.989** | **0.989** | **0.998** |
| Random Forest | 0.963 | 0.964 | 0.963 | 0.963 | 0.995 |
| SVM (RBF) | 0.906 | 0.907 | 0.906 | 0.906 | 0.972 |
| Decision Tree | 0.926 | 0.926 | 0.925 | 0.925 | 0.971 |
| Logistic Regression | 0.895 | 0.896 | 0.895 | 0.896 | 0.966 |
| K-Nearest Neighbours | 0.865 | 0.865 | 0.866 | 0.866 | 0.943 |
| *Random Baseline* | *0.500* | *0.500* | *0.500* | *0.500* | *0.500* |

### Top Predictive Features

```
1. avg_watch_time_per_day    ████████████████████  (strongest predictor)
2. engagement_score          ████████████████
3. watch_hours               ████████████
4. last_login_days           ██████████
5. watch_segment             ████████
```

### Key Findings

- 📺 **Engagement is everything** — customers watching < 0.35 hrs/day are disproportionately at churn risk
- ⏱️ **30-day inactivity is a reliable trigger** — inactive ≥30d customers churn at ~1.8× the rate of active users
- 🧮 **Feature engineering pays off** — the composite `engagement_score` ranks 2nd in importance across both RF and GB
- 💳 **Basic-tier subscribers churn most** — price-sensitive, low-investment customers are the highest-risk cohort
- 👨‍👩‍👧‍👦 **More profiles = more sticky** — multi-profile accounts show significantly lower churn rates
- 🌍 **Region and device matter** — geographic and device-type differentials exceed 5 pp in some segments

---

## 💡 Business Recommendations

| Priority | Recommendation | Impact | Effort |
|----------|---------------|--------|--------|
| 🔴 1 | Deploy real-time churn scoring API (Gradient Boosting) | High | Medium |
| 🔴 2 | Inactivity-triggered re-engagement at days 15/22/28 | High | Low |
| 🔴 3 | Engagement score as weekly KPI with alert thresholds | High | Low |
| 🔴 4 | Subscription tier upgrade incentives for Basic users | High | Medium |
| 🟡 5 | Regional content localisation investment | High | High |
| 🟡 6 | Device-specific UX audits and A/B testing | Medium | High |
| 🟡 7 | Genre-led content strategy for high-churn genres | Medium | High |
| 🟢 8 | Multi-profile household incentive programmes | Medium | Low |
| 🟢 9 | Payment friction reduction (expiry alerts, retry logic) | Medium | Low |
| 🟢 10 | Monthly model retraining pipeline with drift monitoring | High | Medium |

> 💰 **Estimated impact:** A 10% churn reduction on 2,514 churned customers at $13.66 avg fee → **~$41,200 additional annual revenue** per 5,000-subscriber segment.

---

## 🔭 Future Scope

- **SHAP Explainability** — per-subscriber, per-feature churn explanations using TreeExplainer
- **Time-Series Modelling** — LSTM / TCN on weekly engagement trends for temporal churn signals
- **XGBoost / LightGBM / CatBoost** benchmarking for further performance gains
- **Customer Lifetime Value (CLV) integration** — prioritise retention spend by expected revenue impact
- **Real-time Kafka pipeline** — sub-hourly churn score updates from event streams
- **A/B Testing Framework** — measure causal retention impact of each recommendation
- **Survival Analysis** — Cox Proportional Hazards / Kaplan-Meier for time-to-churn estimation
- **One-Hot Encoding** — replace LabelEncoder for nominal categoricals in linear/distance-based models

---

## 📄 Project Report

A full academic report (`Netflix_Churn_Report.docx`) accompanies this project and covers all 18 sections required for the IBM SkillsBuild | AICTE AI Internship submission:

> Abstract · Introduction · Problem Statement · Objectives · Dataset Description · Methodology · Preprocessing · EDA · Feature Engineering · ML Models · Evaluation Metrics · Results · Key Findings · Business Recommendations · Limitations · Future Scope · Conclusion · References (15 citations)

---

## 📦 Repository Contents

```
netflix_customer_churn.csv        ← Dataset
Netflix_Churn_Analytics.ipynb     ← Jupyter Notebook (main deliverable)
Netflix_Churn_Report.docx         ← Academic report (18 sections)
requirements.txt                  ← Pinned dependencies
README.md                         ← This file
```

---

## 🏷️ Tags

`machine-learning` `churn-prediction` `customer-analytics` `gradient-boosting` `random-forest` `pandas` `scikit-learn` `jupyter-notebook` `feature-engineering` `ibm-skillsbuild` `aicte-internship` `streaming-analytics` `python`

---

*Developed as part of the IBM SkillsBuild | AICTE AI Internship Programme — Cohort 2025.*
