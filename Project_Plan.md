# Machine Learning Analysis of Rugby Union Performance Data
## Dissertation Planning & Scoping Document

**Module:** COMP3931 / COMP3932 — Individual / Synoptic Project  
**University:** University of Leeds  
**Author:** Cian Woodsy · cian6woodsy@gmail.com  
**Date:** April 2026  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Summary](#2-dataset-summary)
3. [Proposed ML Project Ideas](#3-proposed-ml-project-ideas)
4. [Recommended Approach](#4-recommended-approach)
5. [Suggested Dissertation Structure](#5-suggested-dissertation-structure)
6. [Project Folder Structure](#6-project-folder-structure)
7. [Recommended Technology Stack](#7-recommended-technology-stack)
8. [Immediate Next Steps](#8-immediate-next-steps)
9. [Academic & Ethical Considerations](#9-academic--ethical-considerations)

---

## 1. Project Overview

This project applies machine learning techniques to Rugby Union match statistics collected across two seasons (2022/23 and 2023/24). The dataset comprises per-game player performance metrics for Backs and Forwards, structured as CSV files. The central aim is to design, implement, and evaluate ML algorithms that extract meaningful patterns from this data — patterns not readily discoverable through conventional statistical methods.

The project satisfies the COMP3931/3932 requirements by:
- Posing a clear **core Computer Science problem** (ML algorithm design and evaluation)
- Producing a **software deliverable** (a Python-based ML analysis pipeline)
- Conducting **rigorous background research** in sports analytics and ML
- Applying **objective, measurable evaluation criteria** throughout

---

## 2. Dataset Summary

| Attribute | Detail |
|---|---|
| Seasons covered | 2022/23 (unlabelled folder) and 2023/24 (`23_24` folder) |
| Position groups | Backs, Forwards |
| Games per season/group | ~19 games |
| Total CSV files | ~76 (19 games × 2 seasons × 2 position groups) |
| Data structure | Row 1 = player names; Rows 2–32 = numeric statistics (31 metrics) |
| Players per game file | ~8 (Backs), ~12 (Forwards) |
| File locations | `Dissertation/Misc/{season}/{position}/game{N}.csv` |

### Confirmed Row Mapping (verified against AiL1 V Trinity 7_10_2023)

All 30 data rows have been confirmed from the original spreadsheet. Three bold summary rows in the spreadsheet are **excluded** from the CSVs (they are re-derived in code):

| CSV Row | Statistic | Category |
|---|---|---|
| 1 | Gainline+ (2.Gainline▶Gainline+) | Attack – Ball Carrier |
| 2 | Gainline 0 (2.Gainline▶Gainline 0) | Attack – Ball Carrier |
| 3 | Unsuccessful Carry | Attack – Ball Carrier |
| 4 | **Total Metres Made** *(summary, kept)* | Attack – Ball Carrier |
| 5 | Defender Beaten | Attack – Ball Carrier |
| 6 | Linebreak Made | Attack – Ball Carrier |
| 7 | Linebreak Conceded | Attack – Ball Carrier |
| 8 | Try | Attack – Ball Carrier |
| 9 | Penalty Try | Attack – Ball Carrier |
| 10 | Successful Pass | Attack – Passing |
| 11 | Unsuccessful Pass | Attack – Passing |
| 12 | Successful Offload | Attack – Passing |
| 13 | Unsuccessful Offload | Attack – Passing |
| 14 | Support▶Pos Attack Ruck | Attack – Support |
| 15 | Support▶Neg Attack Ruck | Attack – Support |
| 16 | Support▶Neutral Attack Ruck | Attack – Support |
| 17 | In Possession▶Positive Support | Attack – Support |
| 18 | In Possession▶Ineffective Support | Attack – Support |
| 19 | Dominant Tackle | Defence |
| 20 | Effective Tackle | Defence |
| 21 | Tackle Assist | Defence |
| 22 | Missed Tackle | Defence |
| 23 | Unsuccessful Tackle | Defence |
| 24 | Positive Barge | Defence |
| 25 | Ineffective Barge | Defence |
| 26 | Turnover Won | Turnovers |
| 27 | Turnover Lost | Turnovers |
| 28 | Discipline – Pen For | Discipline |
| 29 | Discipline – Pen Against | Discipline |
| 30 | Discipline – Yellow Card | Discipline |

**Excluded summary rows** (re-derived in `data_loader.py`):
- *Total Positive Carries* = Gainline+ + Gainline 0
- *Total Positive Tackle Count* = Dominant + Effective Tackle
- *Total Ineffective Tackle Count* = Missed + Unsuccessful Tackle

**Note (Forwards only):** The original spreadsheet also includes set-piece team columns (Scrum, Lineout, Maul) which are **not** present in the individual player CSVs.

---

## 3. Proposed ML Project Ideas

Ten ideas follow, each grounded in the available data and the rigour expected at Level 3. Ideas marked ⭐ are primary recommendations.

---

### ⭐ Idea 1 — Player Position Classification from Performance Statistics

**Research question:** Given only a player's in-game statistics, can a machine learning classifier accurately predict whether they are a Back or Forward?

**Why it matters:** This tests whether the 31 performance metrics encode the positional distinctions that coaches and analysts intuit — making it both a rigorous ML task and a genuine rugby insight.

| Dimension | Detail |
|---|---|
| ML approach | Supervised classification: Logistic Regression, SVM, Random Forest, XGBoost/LightGBM |
| Features | The 31 per-game performance metrics |
| Target label | Back (0) or Forward (1) |
| Academic depth | Feature importance via SHAP values reveals *which* statistics most differentiate positions |
| Evaluation | Cross-validation accuracy, F1-score, ROC-AUC, confusion matrix; compare to naïve baseline |
| Extension | Fine-grained classification (e.g., prop vs. hooker vs. lock) if sub-position labels can be recovered |
| Suitability | ✅ Excellent — well-scoped, clear CS core, strong interpretability story |

---

### ⭐ Idea 2 — Player Performance Prediction Using Historical Game Data

**Research question:** Given a player's statistics across prior games, can a regression model accurately predict their performance in the next game?

**Why it matters:** Predictive modelling has direct practical utility for coaching decisions and injury load management, and provides a compelling ML evaluation challenge.

| Dimension | Detail |
|---|---|
| ML approach | Linear/Ridge/Lasso Regression, Random Forest Regressor, Gradient Boosting; LSTM if sequence modelling is justified |
| Features | Rolling-window aggregates of past N games; game index (temporal position in season) |
| Target | One or more performance metrics in game N+1 (e.g., metres gained, tackles made) |
| Academic depth | Temporal cross-validation (strict no-leakage splits), comparison of stationary vs. sequential models, season-boundary analysis |
| Evaluation | RMSE, MAE, R² per player and in aggregate; compare to naïve baselines (predict next = last, predict next = season mean) |
| Extension | Multi-output prediction of all 31 stats simultaneously (see Idea 8) |
| Suitability | ✅ Excellent — directly applicable, strong academic literature to cite, technically solid |

---

### ⭐ Idea 3 — Unsupervised Player Profiling and Archetype Discovery

**Research question:** Without using any position labels, do clustering algorithms discover meaningful groupings that correspond to rugby archetypes?

**Why it matters:** Unsupervised learning applied to sports data is a rich research area. Discovering that data-driven clusters align with (or meaningfully deviate from) known positions would be a genuinely novel contribution.

| Dimension | Detail |
|---|---|
| ML approach | K-Means, DBSCAN, Agglomerative Clustering; dimensionality reduction via PCA and t-SNE for visualisation |
| Features | Per-player season-aggregated statistics (mean, std, max across all games) |
| Academic depth | Cluster validity metrics (silhouette score, Davies-Bouldin index); qualitative interpretation of clusters; cross-season cluster stability |
| Evaluation | Do clusters align with Back/Forward split? Do sub-clusters correspond to recognisable rugby roles? |
| Extension | Track cluster membership changes season-over-season — a data-driven player development narrative |
| Suitability | ✅ Excellent — visually compelling, rich academic depth, novel rugby insight |

---

### Idea 4 — Key Performance Indicator (KPI) Discovery via Feature Selection

**Research question:** Which of the 31 metrics are the most predictive of overall player quality, and which are redundant?

| Dimension | Detail |
|---|---|
| ML approach | Recursive Feature Elimination, mutual information, LASSO-based selection, SHAP-based ranking |
| Academic depth | Comparison of filter vs. wrapper vs. embedded selection methods; stability analysis across cross-validation folds |
| Evaluation | Model accuracy vs. number of features retained; domain validation of selected KPIs |
| Suitability | ⚠️ Good as a chapter within Ideas 1–3, but limited scope as a standalone dissertation |

---

### Idea 5 — Season-Over-Season Player Development Modelling

**Research question:** Can the change in a player's statistics between the 2022/23 and 2023/24 seasons be modelled and used to characterise player development?

| Dimension | Detail |
|---|---|
| ML approach | Regression on delta-statistics; Gaussian Process Regression for uncertainty; principal component analysis of change vectors |
| Academic depth | Handling small sample sizes; confidence intervals on predictions; statistical significance testing of season differences |
| Evaluation | Leave-one-player-out cross-validation; correlation of predicted vs. actual improvement |
| Suitability | ⚠️ Moderate — excellent add-on chapter; challenging standalone due to sample size |

---

### Idea 6 — Anomaly Detection in Individual Game Performances

**Research question:** Can ML algorithms automatically flag statistically anomalous individual performances — both breakout games and unexpected underperformance?

| Dimension | Detail |
|---|---|
| ML approach | Isolation Forest, Local Outlier Factor, One-Class SVM, Autoencoder-based detection |
| Features | Per-game per-player statistics normalised against that player's historical baseline |
| Evaluation | Precision/recall if ground-truth outliers are known; qualitative review of flagged games |
| Suitability | ⚠️ Moderate — interesting but rigorous evaluation without ground truth labels is difficult |

---

### Idea 7 — Player Similarity Network and Graph-Based Analysis

**Research question:** Can a graph of player statistical similarities reveal community structure that mirrors rugby positional groupings?

| Dimension | Detail |
|---|---|
| ML approach | Cosine/Euclidean similarity matrices; graph construction; community detection (Louvain); optional: Node2Vec embeddings |
| Academic depth | Graph topology analysis; community interpretation vs. ground truth; cross-season graph evolution |
| Evaluation | Modularity of communities; comparison to position-based groupings; stability across similarity thresholds |
| Suitability | ⚠️ Moderate — novel and visually compelling but requires strong graph ML justification |

---

### Idea 8 — Multi-Output Regression for Comprehensive Performance Forecasting

**Research question:** Can a multi-output regression model simultaneously and accurately predict all performance metrics for a player's next game?

| Dimension | Detail |
|---|---|
| ML approach | Multi-output Random Forest, chained regressors, neural network with multi-head output |
| Academic depth | Output correlation exploitation; comparison of independent vs. jointly-trained models |
| Evaluation | Per-target and averaged RMSE; comparison to single-output baselines |
| Suitability | ⚠️ Moderate — technically interesting natural extension of Idea 2 |

---

### Idea 9 — Transfer Learning Across Position Groups

**Research question:** Can a model trained on Forward data generalise to Backs, and what does the domain gap reveal about positional similarity?

| Dimension | Detail |
|---|---|
| ML approach | Domain adaptation; fine-tuning; Maximum Mean Discrepancy (MMD); TrAdaBoost |
| Academic depth | Covariate shift analysis; statistical distribution tests; transfer learning evaluation framework |
| Evaluation | Source-only vs. adapted model accuracy on target domain |
| Suitability | ⚠️ Ambitious — high novelty, best suited to a strong ML student |

---

### Idea 10 — Ensemble Methods for Composite Player Performance Scoring

**Research question:** Can an ensemble of ML models produce a reliable, data-driven composite performance score that outperforms any single metric or model?

| Dimension | Detail |
|---|---|
| ML approach | Stacking ensemble (base learners + meta-learner); Bayesian model averaging; comparison to expert-defined scoring |
| Academic depth | Ensemble diversity metrics; ablation study; comparison to traditional rugby rating systems |
| Evaluation | Internal consistency; validation against season outcomes or expert rankings |
| Suitability | ⚠️ Good capstone idea that synthesises multiple ML approaches |

---

## 4. Recommended Approach

The recommended dissertation focus is a **unified study combining Ideas 1, 2, and 3** — moving from supervised to predictive to unsupervised learning across the same dataset. This provides:

- A clear narrative arc (classification → prediction → discovery)
- Three distinct ML techniques for comparison and contrast
- Sufficient depth for 30 pages without repetition
- A coherent contribution to sports analytics and ML

A secondary recommendation is to incorporate **Idea 4 (KPI discovery)** as part of the analysis in Chapter 4, using SHAP values from the classification model to inform feature selection across all subsequent chapters.

---

## 5. Suggested Dissertation Structure

| Chapter | Content | Primary Idea |
|---|---|---|
| Ch. 1 — Introduction | Motivation, aims, objectives, deliverables, report structure | — |
| Ch. 2 — Background Research | Rugby analytics literature; ML in sports; position classification; performance prediction; player clustering; critical review of prior work | — |
| Ch. 3 — Data & Preprocessing | Dataset description; row labelling; EDA; normalisation strategy; train/test split design; feature engineering | All |
| Ch. 4 — Position Classification | Model design, training, evaluation; SHAP feature importance; KPI identification | Idea 1 |
| Ch. 5 — Performance Prediction | Regression models; temporal validation; season boundary analysis; baseline comparison | Idea 2 |
| Ch. 6 — Player Clustering | Unsupervised archetype discovery; PCA/t-SNE; cross-season comparison | Idea 3 |
| Ch. 7 — Results & Discussion | Cross-chapter evaluation; comparison of approaches; rugby domain interpretation; limitations | All |
| Appendix A | Self-appraisal: reflection, lessons learned, ethical/legal/social/professional issues | — |
| Appendix B | External materials: datasets, libraries, third-party code | — |

> **Marking note:** Per the Final Report Guidance, the main body must not exceed **30 pages**. Allocate roughly 4–5 pages per chapter, with the Results chapter being the longest (6–7 pages). Figures and tables count towards the limit but are encouraged to replace prose wherever possible.

---

## 6. Project Folder Structure

```
Dissertation/
├── Misc/                        ← Original raw data (do not modify)
│   ├── Back/                    ← 2022/23 season, Backs
│   ├── Forward/                 ← 2022/23 season, Forwards
│   └── 23_24/
│       ├── Back/                ← 2023/24 season, Backs
│       └── Forward/             ← 2023/24 season, Forwards
│
├── Parameters/                  ← Dissertation guidance documents
│
└── Project/                     ← All your work lives here
    ├── data/
    │   ├── raw/                 ← Copies of CSVs from Misc/ (read-only)
    │   └── processed/           ← Cleaned, labelled, feature-engineered data
    ├── notebooks/               ← Jupyter notebooks for EDA & experiments
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_position_classification.ipynb
    │   ├── 03_performance_prediction.ipynb
    │   └── 04_player_clustering.ipynb
    ├── src/                     ← Reusable Python source code
    │   ├── data_loader.py       ← Load & parse all CSVs into a DataFrame
    │   ├── features.py          ← Feature engineering functions
    │   ├── models.py            ← Model training & evaluation wrappers
    │   └── visualisation.py     ← Plotting utilities
    ├── models/                  ← Saved trained models (.pkl / .joblib)
    ├── results/                 ← Evaluation outputs, metrics JSON, plots
    ├── reports/                 ← Planning docs, interim reports, dissertation drafts
    ├── requirements.txt         ← Python dependencies
    └── README.md                ← Project setup instructions
```

---

## 7. Recommended Technology Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| Language | Python 3.10+ | Primary implementation language |
| Data handling | pandas, numpy | Data loading, manipulation, feature engineering |
| Machine learning | scikit-learn | Classification, regression, clustering, pipeline management |
| Boosting | XGBoost / LightGBM | High-performance gradient boosting models |
| Deep learning (optional) | PyTorch | LSTM or autoencoder if sequence/anomaly modelling is pursued |
| Explainability | SHAP | Feature importance, model interpretation, KPI discovery |
| Dimensionality reduction | scikit-learn (PCA), umap-learn | Visualising high-dimensional player profiles |
| Visualisation | matplotlib, seaborn, plotly | EDA charts, cluster plots, evaluation figures |
| Notebooks | Jupyter Lab | Interactive exploration and reproducible experiments |
| Version control | Git + GitHub | Code versioning; provide link to supervisor and assessor |
| Environment | conda or venv + requirements.txt | Reproducible environment |
| Report | LaTeX (preferred) or Word template | Must use the university-provided template |

### Suggested `requirements.txt`
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
matplotlib>=3.7
seaborn>=0.13
plotly>=5.18
jupyterlab>=4.0
umap-learn>=0.5
joblib>=1.3
```

---

## 8. Immediate Next Steps

| Priority | Task | Notes |
|---|---|---|
| 🔴 Critical | **Label the 31 CSV row metrics** | Confirm each row index maps to a rugby statistic; validate against a known game |
| 🔴 Critical | **Copy raw data to `Project/data/raw/`** | Preserve the originals in `Misc/`; work only on copies |
| 🟠 High | **Set up Python environment** | Install requirements.txt; create `src/data_loader.py` |
| 🟠 High | **Conduct EDA** (`01_data_exploration.ipynb`) | Distribution plots, correlation heatmaps, missing value audit, season comparison |
| 🟠 High | **Agree dissertation focus with supervisor** | Present the combined Ideas 1–3 approach; confirm scope suits COMP3931 |
| 🟡 Medium | **Literature review** | Key topics: ML in rugby/sport, player classification, performance prediction, clustering in sport; target 15–20 papers |
| 🟡 Medium | **Define evaluation framework** | Specify metrics, validation strategy, and success criteria *before* training any models (marks awarded for this) |
| 🟡 Medium | **Set up Git repository** | Required by the report guidelines; share with supervisor and assessor |
| 🟢 Lower | **Draft Introduction chapter** | Write aims, objectives, and deliverables early; these guide all subsequent decisions |

---

## 9. Academic & Ethical Considerations

Per the **Final Report Guidance (COMP3931/3932)**, a discussion of legal, social, ethical, and professional issues is **mandatory in Appendix A**. Even if an issue does not apply, you must justify this with specific reference to your project.

Relevant considerations for this project include:

**Data Privacy:** The dataset contains named players with identifiable performance statistics. Even if this is amateur/semi-professional data, consider whether explicit consent was obtained for ML analysis and publication. BCS guidelines on data handling apply.

**Bias in ML Models:** Performance models trained on historical data may encode existing biases (e.g., certain positions being underrepresented, or selection bias in which games are included). Any predictive tool derived from this work should not be applied uncritically to real-world selection decisions.

**Professional Responsibility:** If results are shared with a club or coaching staff, the limitations and uncertainty of ML predictions must be communicated clearly. A model that misclassifies a player's position or mispredicts their performance could influence selection decisions with real consequences for individuals.

**No Sensitive Personal Data:** The dataset contains only sports statistics and not sensitive personal data (health records, financial data, etc.), which simplifies the ethical picture considerably — but this should still be explicitly stated in the appendix.

---

*Document version 1.0 — April 2026*  
*Generated by Claude (Cowork mode) based on analysis of the Dissertation/Misc dataset and Dissertation/Parameters guidance documents.*
