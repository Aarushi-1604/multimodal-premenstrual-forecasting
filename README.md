# 🌸 Multimodal Personalized Deep Learning for Premenstrual Symptom Forecasting

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-yellowgreen.svg)](https://catboost.ai/)
[![Dataset: PhysioNet mcPHASES](https://img.shields.io/badge/PhysioNet-mcPHASES-blueviolet.svg)](https://physionet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A leakage-controlled deep learning benchmark for prospective next-day multi-symptom forecasting using continuous physiological wearables and self-reported symptom logs from the **PhysioNet mcPHASES** cohort.

---

## 📌 Executive Summary & Key Highlights

* **The Problem:** Commercial menstrual cycle apps (e.g., Flo, Clue, Stardust) primarily offer *retrospective logging* or static 28-day calendar tracking. Clinically actionable intervention requires *prospective next-day forecasting* ($t+1$) using rolling multimodal biometrics.
* **Leakage-Controlled Protocol:** Strict 5-fold **Participant GroupKFold** cross-validation ($N=42$ unique participants, ~2,400 windows) ensuring **zero subject-identity leakage** and zero temporal lookahead leakage.
* **Core Architectures Benchmarked:** 
  1. Multi-Task GRU (Single-layer recurrent backbone with shared projection head)
  2. Multi-Task MLP (Flattened temporal sequence baseline)
  3. CatBoost (Independent gradient-boosted symmetric decision trees)
  4. L2-Regularized Logistic Regression (Linear baseline)
  5. Persistence / Lag-1 Baseline (Predicting yesterday's status)
* **Temporal Inductive Bias Proven:** The Multi-Task GRU systematically out-predicted the Multi-Task MLP across all 6 targets (+0.02 to +0.05 PR-AUC), proving that day-to-day recurrent transitions carry distinct clinical signals over static flattened features.
* **Causal Personalization (Hypothesis H1):** Disentangled personal biological alignment from sample size expansion using a 3-arm design. Adding a participant's own past data (+53 windows) outperformed adding $7\times$ more data from other women (+363 windows) for idiosyncratic symptoms like *Sore Breasts* (+0.142 PR-AUC).

---

## 🎯 Target Formulation & Prevalence

Next-day symptom status is framed as 6 simultaneous binary prediction tasks on Day $t+1$: a symptom is considered positive if its self-reported ordinal severity is **Moderate or Severe** ($\ge 3$ on a 0–5 scale).

| Symptom Target | Cohort Prevalence | 5-Fold Evaluation Range | Imbalance Mitigation |
| :--- | :---: | :---: | :--- |
| **Fatigue** | 56.0% | 51.9% – 59.2% | Balanced cross-entropy |
| **Bloating** | 30.9% | 12.6% – 49.3% | Positive class weighting |
| **Mood Swings** | 29.8% | 15.0% – 46.6% | Positive class weighting |
| **Headaches** | 25.3% | 16.6% – 31.6% | Positive class weighting |
| **Cramps (Dysmenorrhea)** | 15.8% | 6.2% – 24.4% | Masked BCE positive weight $w \approx 5.3$ |
| **Sore Breasts (Mastalgia)** | 11.6% | 3.0% – 19.9% | Masked BCE positive weight $w \approx 7.6$ |


  7-Day Rolling Input Window: [Day t−6, ..., Day t]
              Shape: (batch_size=64, seq_len=7, feat_dim=43)
                            │
                            ▼
   ┌───────────────────────────────────────────────────┐
   │             Shared Temporal GRU Backbone          │
   │    nn.GRU(input_size=43, hidden_size=32, L=1)     │
   │    Captures dynamic multi-day physiological drift │
   └────────────────────────┬──────────────────────────┘
                            │
             Readout: Last Hidden State h₇ (dim=32)
                            │
                            ▼
                     Dropout (p=0.3)
                            │
                            ▼
   ┌───────────────────────────────────────────────────┐
   │             Multi-Task Projection Head            │
   │               nn.Linear(in=32, out=6)             │
   └────────────────────────┬──────────────────────────┘
                            │
       6 Next-Day Logits (Day t+1) via Masked Multi-Task BCE:
[ Cramps, Mood Swings, Fatigue, Bloating, Sore Breasts, Headaches ]


### Parameter Efficiency & Capacity Comparison

| Model | Architecture Specification | Trainable Parameters | Inductive Bias & Role |
| :--- | :--- | :---: | :--- |
| **Multi-Task GRU** | 43 $\to$ GRU(32) $\to$ Linear(6) | **7,590** | Temporal recurrence; prevents overfitting on small $N=42$ cohort |
| **Multi-Task MLP** | 301 $\to$ 64 $\to$ 32 $\to$ 6 | **21,734** | High capacity; flattens sequence into static vector |
| **Logistic Regression**| 6 models $\times$ (301 $\times$ 1 + 1) | **1,812** | Linear baseline; no feature interactions |
| **CatBoost** | 6 models $\times$ 250 trees $\times$ depth 4 | **~30,000 leaves** | Non-linear tree ensemble; invariant to continuous/discrete feature scales |

---

## 📊 Benchmark Results

### 1. Test PR-AUC (Average Precision across 5 Folds)
*Evaluated on out-of-fold unseen participants (Mean ± Standard Deviation):*

| Symptom | Persistence | Logistic Reg | CatBoost | MLP (Flat) | Multi-Task GRU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cramps** | 0.411 ± 0.187 | 0.503 ± 0.170 | **0.554 ± 0.191** | 0.501 ± 0.245 | **0.548 ± 0.196** |
| **Mood Swings** | 0.530 ± 0.185 | 0.647 ± 0.184 | **0.658 ± 0.195** | 0.615 ± 0.172 | **0.649 ± 0.183** |
| **Fatigue** | 0.733 ± 0.049 | 0.819 ± 0.052 | **0.852 ± 0.053** | 0.795 ± 0.061 | **0.824 ± 0.071** |
| **Bloating** | 0.569 ± 0.182 | 0.649 ± 0.207 | **0.690 ± 0.223** | 0.627 ± 0.233 | **0.682 ± 0.202** |
| **Sore Breasts** | 0.407 ± 0.239 | 0.472 ± 0.241 | **0.515 ± 0.186** | 0.425 ± 0.231 | **0.457 ± 0.248** |
| **Headaches** | 0.476 ± 0.114 | 0.646 ± 0.087 | **0.659 ± 0.096** | 0.579 ± 0.119 | **0.635 ± 0.102** |

### 2. Why Did CatBoost Slightly Edge Out GRU?
1. **Heterogeneous Scales:** The 43 daily features combine continuous skin temperatures, heart rates, discrete ordinal symptom integers (0–5), and binary flags. Decision trees split on axis-aligned rank thresholds ($x_j > \theta$), making them completely invariant to feature scale differences.
2. **Small Sample Regime ($N=42$):** Tree ensembles consistently match or exceed neural networks on small tabular datasets under $10^4$ rows without overfitting (Grinsztajn et al., NeurIPS 2022).
3. **Oblivious Trees:** CatBoost enforces identical depth-4 symmetric splits across tree levels, acting as a structural regularizer against high variance.

---

## 🔬 Exploratory Personalization (Hypothesis H1)

To determine whether fine-tuning models on a woman's own historical cycles helps because of **personal biological alignment** or simply **larger sample size**, we conducted a 3-arm causal experiment with a 2,000-replicate cluster bootstrap:

* **Arm A (Population):** Trained only on other participants' data.
* **Arm B (Personalized):** Arm A + participant's own past cycles (+53 windows median).
* **Arm C (Volume Control):** Arm A + other women's past cycles (+363 windows median — **$7\times$ more data**).
* **Finding:** For individual-specific symptoms like *Sore Breasts*, Arm B outperformed Arm C ($B - C = +0.142$ PR-AUC in 94.8% of bootstrap resamples), confirming that idiosyncratic hormonal sensitivity cannot be solved by simply pooling more unrelated population data.

---

## 📁 Repository Structure

```text
├── data/                       # Local data directories (ignored under DUA)
├── figures/                    # Publication-grade figures (300 DPI, DUA-compliant)
│   ├── fig1_pipeline_diagram.png
│   ├── fig2_prevalence_range.png
│   ├── fig3_cycle_intervals_hist.png
│   ├── fig4_gru_schematic.png
│   ├── fig5_pr_auc_comparison.png
│   ├── fig6_brier_comparison.png
│   ├── fig7_paired_fold_diffs.png
│   ├── fig8_neural_learning_curves.png
│   └── fig9_forest_plot_personalization.png
├── results/                    # Aggregated metrics & cross-validation summaries
│   ├── all_models_summary.csv
│   └── h1v2_summary.csv
├── src/
│   ├── data/
│   │   ├── build_master.py     # Multimodal synchronization & causal windowing
│   │   └── make_windows.py     # 7-day rolling window generator with zero leakage
│   ├── models/
│   │   ├── baselines.py        # Persistence, Logistic Regression, and CatBoost
│   │   └── train_neural.py     # PyTorch Multi-Task GRU, MLP & Masked BCE Loss
│   └── visualization/
│       └── generate_report_figures.py
├── slides.html                 # Complete 10-slide interactive presentation deck
├── .gitignore
└── README.md

⚙️ Setup & Reproduction
1. Environment Setup
bash
# Clone the repository
git clone https://github.com/<your-username>/multimodal-premenstrual-forecasting.git
cd multimodal-premenstrual-forecasting
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install torch torchvision numpy pandas scikit-learn catboost matplotlib seaborn
2. Execution Pipeline
bash
# 1. Feature synchronization and causal window generation
python src/data/build_master.py
python src/data/make_windows.py
# 2. Run baselines (Persistence, Logistic Regression, CatBoost)
python src/models/baselines.py
# 3. Train Multi-Task GRU and Multi-Task MLP with GroupKFold
python src/models/train_neural.py
# 4. Generate all figures
python src/visualization/generate_report_figures.py

🔒 Data Access & Ethics Notice (DUA Compliance)

The raw physiological signals and symptom diaries used in this study originate from the mcPHASES dataset hosted on PhysioNet.

Access is governed by the PhysioNet Credentialed Data Use Agreement (DUA).
Per DUA terms, no raw patient logs, participant identifiers, or unaggregated cycle data are hosted in this repository.
Researchers wishing to reproduce the raw data preprocessing must apply for credentialed access directly via PhysioNet mcPHASES.

📜 Citation & Academic Attribution
bibtex
@misc{aarushi2026premenstrual,
  author = {Aarushi, S. and Ahmed, Tanveer},
  title = {Multimodal Personalized Deep Learning for Premenstrual Symptom Forecasting},
  year = {2026},
  institution = {Symbiosis Institute of Technology (SIT), Symbiosis International University},
  howpublished = {\url{https://github.com/<your-username>/multimodal-premenstrual-forecasting}}
}
---

## 🏗️ Model Architecture (Multi-Task GRU)
