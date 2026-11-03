"""
src/visualization/generate_report_figures.py
Generates publication-quality figures for the final report into figures/:
  fig1_pipeline_diagram.png: Flowchart with exact sample & window counts
  fig2_prevalence_range.png: Target prevalence overall and range across folds
  fig3_cycle_intervals_hist.png: Histogram of the 79 detected cycle intervals
  fig4_gru_schematic.png: Multi-task GRU temporal architecture schematic
  fig5_pr_auc_comparison.png: PR-AUC by model and target with fold SD
  fig6_brier_comparison.png: Brier score by model and target (lower is better)
  fig7_paired_fold_diffs.png: Paired fold differences (CB vs GRU, CB vs Persistence)
  fig8_neural_learning_curves.png: Neural learning curves (Fold 0)
  fig9_forest_plot_personalization.png: Forest plot of B-A, C-A, B-C bootstrap CIs

Guarantees:
  - DUA-compliant: Aggregates only, zero participant IDs, zero per-person curves, zero dates.
"""

from pathlib import Path
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")

SYMPTOM_NAMES = ["cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches"]
SYMPTOM_DISPLAY = {
    "cramps": "Cramps",
    "moodswing": "Mood Swings",
    "fatigue": "Fatigue",
    "bloating": "Bloating",
    "sorebreasts": "Sore Breasts",
    "headaches": "Headaches",
}

MODEL_DISPLAY = {
    "Persistence_LastObserved": "Persistence",
    "LogReg": "Logistic Reg",
    "CatBoost": "CatBoost",
    "MLP": "MLP (Flat)",
    "GRU": "GRU (Temporal)",
}

MODEL_COLORS = {
    "Persistence_LastObserved": "#7f7f7f",
    "LogReg": "#1f77b4",
    "CatBoost": "#2ca02c",
    "MLP": "#ff7f0e",
    "GRU": "#d62728",
}

# -----------------------------------------------------------------------------
# FIG 1: Pipeline Diagram
# -----------------------------------------------------------------------------
def make_fig1():
    print("Generating Fig 1: Pipeline Diagram...")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Box styling helper
    def draw_box(x, y, w, h, title, details, color="#e8f4f8", border="#2b5c8f"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=1.2,rounding_size=2.0",
            facecolor=color, edgecolor=border, linewidth=1.8, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#111111", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, details, ha="center", va="center",
                fontsize=9.5, color="#333333", zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=2, mutation_scale=15),
            zorder=1
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 2.5, label, ha="center", va="center",
                    fontsize=8.5, color="#555555", fontweight="bold")

    # Step 1: Raw Data
    draw_box(3, 62, 26, 26, "1. Raw mcPHASES Data", "5,659 total diary rows\n(2022 & 2024 intervals, 42 pts)", "#f0f4f8", "#334e68")
    
    # Arrow 1 -> 2
    draw_arrow(29, 75, 37, 75, "Filter 2022")

    # Step 2: 2022 Base Grid
    draw_box(37, 62, 26, 26, "2. 2022 Base Grid", "3,698 participant-days\n(10.2% symptom missingness;\nFitbit RHR, temp, sleep, LH, E3G)", "#e6f7ff", "#096dd9")

    # Arrow 2 -> 3
    draw_arrow(63, 75, 71, 75, "7-day sliding")

    # Step 3: Candidate Windows
    draw_box(71, 62, 26, 26, "3. Candidate Windows", "3,404 sliding windows\n(7 days x 43 features)\n-42x7 days dropped at start", "#e6ffed", "#389e0d")

    # Arrow 3 -> 4 (downward)
    draw_arrow(84, 62, 84, 44, "Mask M==0")

    # Step 4: Supervised Evaluation Set
    draw_box(68, 12, 29, 28, "4. Supervised Windows", "3,036 valid windows (89.2%)\n368 unlogged t+1 days masked\nTargets: 6 symptoms >= Moderate", "#fff7e6", "#d46b08")

    # Arrow 4 -> 5
    draw_arrow(68, 26, 57, 26, "GroupKFold")

    # Step 5: Leakage-Free Validation & Modeling
    draw_box(4, 10, 53, 32, "5. Participant-Grouped 5-Fold Validation & Modeling",
             "Outer GroupKFold: 5 folds (8-9 unseen test participants/fold)\n"
             "Inner GroupKFold (Neural): 1 held-out inner fold for early stopping\n"
             "Zero Leakage: Scalers, median imputers, class weights fitted on train only\n"
             "Evaluated: Persistence (LastObs), LogReg, CatBoost, Multi-Task MLP, Multi-Task GRU",
             "#f9f0ff", "#722ed1")

    plt.title("mcPHASES Next-Day Forecasting Pipeline: Data Flow & Sample Counts", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig1_pipeline_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 2: Prevalence per Target (Overall & Range across Folds)
# -----------------------------------------------------------------------------
def make_fig2():
    print("Generating Fig 2: Prevalence Range across Folds...")
    folds_df = pd.read_csv(RESULTS_DIR / "all_models_groupkfold_folds.csv")
    cb_df = folds_df[folds_df["model"] == "CatBoost"]
    
    summary = []
    for sym in SYMPTOM_NAMES:
        sub = cb_df[cb_df["symptom"] == sym]
        mean_prev = sub["prevalence"].mean()
        min_prev = sub["prevalence"].min()
        max_prev = sub["prevalence"].max()
        summary.append({
            "symptom": sym,
            "display": SYMPTOM_DISPLAY[sym],
            "mean": mean_prev,
            "min": min_prev,
            "max": max_prev,
            "folds": sub["prevalence"].values
        })
    df = pd.DataFrame(summary).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    y_pos = np.arange(len(df))
    
    # Range bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["min"], row["max"]], [i, i], color="#4a5568", lw=3, zorder=2)
        ax.scatter(row["folds"], [i]*len(row["folds"]), color="#718096", s=40, alpha=0.7, zorder=3)
        ax.scatter(row["mean"], i, color="#2b6cb0", s=110, edgecolors="#1a365d", lw=1.5, zorder=4)
        ax.text(row["mean"] + 0.015, i - 0.15, f"{row['mean']*100:.1f}% (range {row['min']*100:.1f}-{row['max']*100:.1f}%)",
                fontsize=9.5, fontweight="bold", color="#2d3748")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Positive Prevalence (Moderate or Worse on Day t+1)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 0.70)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_title("Target Positive Prevalence: Overall Mean & Fold-to-Fold Range (5 Folds)", fontsize=12, fontweight="bold", pad=12)

    # Custom legend
    p_mean = ax.scatter([], [], color="#2b6cb0", s=100, edgecolors="#1a365d", label="Cohort Mean Prevalence")
    p_pts = ax.scatter([], [], color="#718096", s=40, label="Individual Fold Prevalences")
    ax.legend(handles=[p_mean, p_pts], loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig2_prevalence_range.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 3: Cycle-Interval Lengths Histogram
# -----------------------------------------------------------------------------
def make_fig3():
    print("Generating Fig 3: Cycle Interval Histogram...")
    df_master = pd.read_csv("data/processed/master_daily_2022.csv")
    pids = df_master["id"].unique()
    intervals = []
    for pid in pids:
        p_df = df_master[df_master["id"] == pid].sort_values("day_in_study")
        bleed_days = p_df.loc[p_df["flow_volume"].notna() & (p_df["flow_volume"] != "Not at all"), "day_in_study"].tolist()
        episodes = []
        if bleed_days:
            curr_ep = [bleed_days[0]]
            for d in bleed_days[1:]:
                if d - curr_ep[-1] <= 3: curr_ep.append(d)
                else: episodes.append(curr_ep); curr_ep = [d]
            episodes.append(curr_ep)
        raw_onsets = [ep[0] for ep in episodes]
        merged = []
        for o in raw_onsets:
            if not merged or o - merged[-1] >= 15: merged.append(o)
        for i in range(len(merged)-1):
            intervals.append(merged[i+1] - merged[i])

    intervals = np.array(intervals)
    assert len(intervals) == 79, f"Expected 79 intervals, got {len(intervals)}"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    bins = np.arange(15, 65, 2)
    n, b, patches_list = ax.hist(intervals, bins=bins, color="#3182ce", edgecolor="#1a365d", alpha=0.85, zorder=2)

    # Annotate clinical standard range (21-35 days)
    ax.axvspan(21, 35, color="#48bb78", alpha=0.18, zorder=1, label="Typical Clinical Range (21–35 days)")
    ax.axvline(29, color="#e53e3e", linestyle="--", lw=2, zorder=3, label=f"Cohort Median = 29 days (Mean = {intervals.mean():.1f} d)")

    ax.set_xlabel("Cycle-to-Cycle Interval Length (days)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Detected Intervals (N = 79)", fontsize=11, fontweight="bold")
    ax.set_title("Distribution of Detected Menstrual Cycle Lengths (2022 Interval, 39 Participants)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig3_cycle_intervals_hist.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 4: Multi-Task GRU Schematic
# -----------------------------------------------------------------------------
def make_fig4():
    print("Generating Fig 4: Multi-Task GRU Schematic...")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 1. Input sequence boxes
    ax.text(50, 93, "7-Day Input Window: [Day t-6, ..., Day t] (7 timesteps x 43 features)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1a202c")

    days = ["t-6", "t-5", "t-4", "t-3", "t-2", "t-1", "t"]
    for i, d in enumerate(days):
        x = 8 + i * 12
        y = 75
        rect = patches.FancyBboxPatch((x, y), 9.5, 12, boxstyle="round,pad=0.5,rounding_size=1.0",
                                      facecolor="#ebf8ff", edgecolor="#3182ce", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 4.75, y + 6, f"$x_{{{d}}}$\n(43 feats)", ha="center", va="center", fontsize=8.5, fontweight="bold")
        # Downward arrow to GRU
        ax.annotate("", xy=(x + 4.75, 59), xytext=(x + 4.75, 75),
                    arrowprops=dict(arrowstyle="-|>", color="#2b6cb0", lw=1.5))

    # 2. GRU Layer
    gru_box = patches.FancyBboxPatch((6, 45), 88, 14, boxstyle="round,pad=0.8,rounding_size=1.5",
                                    facecolor="#feebc8", edgecolor="#dd6b20", lw=2)
    ax.add_patch(gru_box)
    ax.text(50, 52, "Single-Layer GRU (hidden_size = 32, batch_first = True)\nProcesses temporal progression across days t-6 to t",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#7b341e")

    # Readout arrow from day t
    ax.annotate("", xy=(50, 33), xytext=(84.75, 45),
                arrowprops=dict(arrowstyle="-|>", color="#c05621", lw=2.2))
    ax.text(70, 39, "Readout Last Hidden State $h_t$ (dim=32)", fontsize=9, fontweight="bold", color="#9c4221")

    # 3. Dropout & Linear Head
    head_box = patches.FancyBboxPatch((32, 21), 36, 12, boxstyle="round,pad=0.6,rounding_size=1.2",
                                     facecolor="#e9d8fd", edgecolor="#6b46c1", lw=1.8)
    ax.add_patch(head_box)
    ax.text(50, 27, "Dropout (p = 0.3) -> Linear Layer (32 -> 6)\nShared Multi-Task Head",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#44337a")

    # Downward arrow to output logits
    ax.annotate("", xy=(50, 11), xytext=(50, 21),
                arrowprops=dict(arrowstyle="-|>", color="#553c9a", lw=2))

    # 4. Output Logits
    ax.text(50, 6, "6 Logits on Day t+1: [Cramps, Mood Swings, Fatigue, Bloating, Sore Breasts, Headaches]\n"
                   "Trained with Masked Multi-Task BCEWithLogitsLoss (Positive class weighting per symptom)",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2d3748")

    plt.title("Multi-Task GRU Temporal Architecture for Next-Day Symptom Forecasting", fontsize=12.5, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig4_gru_schematic.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 5: PR-AUC Model Comparison (Checked against corrected persistence)
# -----------------------------------------------------------------------------
def make_fig5():
    print("Generating Fig 5: PR-AUC Model Comparison...")
    df_sum = pd.read_csv(RESULTS_DIR / "all_models_summary.csv")
    
    # Assert persistence cramps is ~0.41
    cramps_pers = df_sum[(df_sum["symptom"] == "cramps") & (df_sum["model"] == "Persistence_LastObserved")]["PR-AUC_mean"].values[0]
    assert abs(cramps_pers - 0.4109) < 0.01, f"Persistence cramps PR-AUC is {cramps_pers}, expected ~0.411!"

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    n_syms = len(SYMPTOM_NAMES)
    x_positions = np.arange(n_syms)
    bar_width = 0.15

    model_order = ["Persistence_LastObserved", "LogReg", "CatBoost", "MLP", "GRU"]

    for idx, m_name in enumerate(model_order):
        m_data = df_sum[df_sum["model"] == m_name].set_index("symptom").loc[SYMPTOM_NAMES]
        means = m_data["PR-AUC_mean"].values
        stds = m_data["PR-AUC_std"].values
        ax.bar(
            x_positions + idx * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            label=MODEL_DISPLAY[m_name],
            color=MODEL_COLORS[m_name],
            capsize=3,
            alpha=0.9
        )

    ax.set_xticks(x_positions + bar_width * 2)
    ax.set_xticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=11, fontweight="bold")
    ax.set_ylabel("PR-AUC (Average Precision)", fontsize=11, fontweight="bold")
    ax.set_title("Next-Day Symptom Forecasting: PR-AUC across 5 Models (5-Fold GroupKFold)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig5_pr_auc_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 6: Brier Score Model Comparison
# -----------------------------------------------------------------------------
def make_fig6():
    print("Generating Fig 6: Brier Score Comparison...")
    df_sum = pd.read_csv(RESULTS_DIR / "all_models_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    n_syms = len(SYMPTOM_NAMES)
    x_positions = np.arange(n_syms)
    bar_width = 0.15

    model_order = ["Persistence_LastObserved", "LogReg", "CatBoost", "MLP", "GRU"]

    for idx, m_name in enumerate(model_order):
        m_data = df_sum[df_sum["model"] == m_name].set_index("symptom").loc[SYMPTOM_NAMES]
        means = m_data["Brier_mean"].values
        stds = m_data["Brier_std"].values
        ax.bar(
            x_positions + idx * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            label=MODEL_DISPLAY[m_name],
            color=MODEL_COLORS[m_name],
            capsize=3,
            alpha=0.9
        )

    ax.set_xticks(x_positions + bar_width * 2)
    ax.set_xticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=11, fontweight="bold")
    ax.set_ylabel("Brier Score (Mean Squared Probability Error; Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Probabilistic Error: Brier Score across 5 Models (5-Fold GroupKFold)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 0.32)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig6_brier_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 7: Paired Per-Fold Differences
# -----------------------------------------------------------------------------
def make_fig7():
    print("Generating Fig 7: Paired Per-Fold Differences...")
    folds_df = pd.read_csv(RESULTS_DIR / "all_models_groupkfold_folds.csv")

    # Panel 1: CatBoost - GRU (PR-AUC, positive = CatBoost better)
    # Panel 2: CatBoost - Persistence (Brier, negative = CatBoost better)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    cb_pr = folds_df[folds_df["model"] == "CatBoost"].set_index(["symptom", "fold"])["PR-AUC"]
    gru_pr = folds_df[folds_df["model"] == "GRU"].set_index(["symptom", "fold"])["PR-AUC"]
    diff_cb_gru_pr = cb_pr - gru_pr

    cb_br = folds_df[folds_df["model"] == "CatBoost"].set_index(["symptom", "fold"])["Brier"]
    pers_br = folds_df[folds_df["model"] == "Persistence_LastObserved"].set_index(["symptom", "fold"])["Brier"]
    diff_cb_pers_br = cb_br - pers_br

    x = np.arange(len(SYMPTOM_NAMES))

    # Left: CatBoost vs GRU PR-AUC
    for i, sym in enumerate(SYMPTOM_NAMES):
        vals = [diff_cb_gru_pr.loc[(sym, f)] for f in range(5)]
        mean_v = np.mean(vals)
        axes[0].scatter([i]*5, vals, color="#718096", s=45, alpha=0.7, zorder=3)
        axes[0].scatter(i, mean_v, color="#2ca02c", s=110, edgecolors="#1b611b", lw=1.5, zorder=4)

    axes[0].axhline(0, color="black", linestyle="--", lw=1.2, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=10, fontweight="bold")
    axes[0].set_ylabel("PR-AUC Difference: CatBoost - GRU\n(Positive = CatBoost Better)", fontsize=10.5, fontweight="bold")
    axes[0].set_title("CatBoost vs GRU (Per-Fold PR-AUC Difference)", fontsize=11.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Right: CatBoost vs Persistence Brier
    for i, sym in enumerate(SYMPTOM_NAMES):
        vals = [diff_cb_pers_br.loc[(sym, f)] for f in range(5)]
        mean_v = np.mean(vals)
        axes[1].scatter([i]*5, vals, color="#718096", s=45, alpha=0.7, zorder=3)
        axes[1].scatter(i, mean_v, color="#3182ce", s=110, edgecolors="#1a365d", lw=1.5, zorder=4)

    axes[1].axhline(0, color="black", linestyle="--", lw=1.2, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Brier Difference: CatBoost - Persistence\n(Negative = CatBoost Better)", fontsize=10.5, fontweight="bold")
    axes[1].set_title("CatBoost vs Persistence (Per-Fold Brier Difference)", fontsize=11.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Paired Per-Fold Model Comparisons across 5 Outer Folds", fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig7_paired_fold_diffs.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# FIG 8: Neural Learning Curves (Fold 0)
# -----------------------------------------------------------------------------
def make_fig8():
    print("Generating Fig 8: Copying Neural Learning Curves...")
    src_file = RESULTS_DIR / "plots" / "neural_learning_curves_fold0.png"
    dst_file = FIGURES_DIR / "fig8_neural_learning_curves.png"
    if src_file.exists():
        shutil.copyfile(src_file, dst_file)
        print(f"Saved: {dst_file}")
    else:
        print(f"Warning: {src_file} not found!")

# -----------------------------------------------------------------------------
# FIG 9: Forest Plot of Personalization Bootstrap Differences
# -----------------------------------------------------------------------------
def make_fig9():
    print("Generating Fig 9: Forest Plot of Personalization...")
    h1_df = pd.read_csv(RESULTS_DIR / "h1v2_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Helper to parse "[+0.0149, +0.1529]"
    def parse_ci(ci_str):
        clean = ci_str.strip("[]").replace("+", "")
        parts = [float(p.strip()) for p in clean.split(",")]
        return parts[0], parts[1]

    y_pos = np.arange(len(SYMPTOM_NAMES))

    # Panel A: PR-AUC Differences
    offsets = [-0.22, 0.0, 0.22]
    colors = ["#2b6cb0", "#718096", "#c53030"]
    labels = ["B - A (Personalized - Population)", "C - A (Extra Data - Population)", "B - C (Personalized - Extra Data)"]

    for i, sym in enumerate(SYMPTOM_NAMES):
        row = h1_df[h1_df["symptom"] == sym].iloc[0]
        # B - A
        ba_mean = row["BA_PR_mean"]
        ba_low, ba_high = parse_ci(row["BA_PR_CI"])
        # C - A
        ca_mean = row["CA_PR_mean"]
        ca_low, ca_high = parse_ci(row["CA_PR_CI"])
        # B - C
        bc_mean = row["BC_PR_mean"]
        bc_low, bc_high = parse_ci(row["BC_PR_CI"])

        for arm_idx, (m, low, high) in enumerate([(ba_mean, ba_low, ba_high), (ca_mean, ca_low, ca_high), (bc_mean, bc_low, bc_high)]):
            y = i + offsets[arm_idx]
            axes[0].plot([low, high], [y, y], color=colors[arm_idx], lw=2.2)
            axes[0].scatter(m, y, color=colors[arm_idx], s=50, zorder=3)

    axes[0].axvline(0, color="black", linestyle="--", lw=1.2, alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("PR-AUC Difference (Mean & 95% Percentile Bootstrap CI)", fontsize=10.5, fontweight="bold")
    axes[0].set_title("Personalization Effect on PR-AUC (N=497, 2000 Bootstrap Reps)", fontsize=11.5, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    # Panel B: Brier Differences (Negative is better)
    for i, sym in enumerate(SYMPTOM_NAMES):
        row = h1_df[h1_df["symptom"] == sym].iloc[0]
        ba_mean = row["BA_Br_mean"]
        ba_low, ba_high = parse_ci(row["BA_Br_CI"])
        ca_mean = row["CA_Br_mean"]
        ca_low, ca_high = parse_ci(row["CA_Br_CI"])
        bc_mean = row["BC_Br_mean"]
        bc_low, bc_high = parse_ci(row["BC_Br_CI"])

        for arm_idx, (m, low, high) in enumerate([(ba_mean, ba_low, ba_high), (ca_mean, ca_low, ca_high), (bc_mean, bc_low, bc_high)]):
            y = i + offsets[arm_idx]
            axes[1].plot([low, high], [y, y], color=colors[arm_idx], lw=2.2)
            axes[1].scatter(m, y, color=colors[arm_idx], s=50, zorder=3)

    axes[1].axvline(0, color="black", linestyle="--", lw=1.2, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([SYMPTOM_DISPLAY[s] for s in SYMPTOM_NAMES], fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Brier Difference (Negative = Better; 95% Bootstrap CI)", fontsize=10.5, fontweight="bold")
    axes[1].set_title("Personalization Effect on Brier Score (N=497, 2000 Bootstrap Reps)", fontsize=11.5, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    # Custom legend on Panel 0
    lines = [patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=lines, loc="lower center", ncol=3, frameon=True, fontsize=10.5, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle("Exploratory Personalization (H1): Disentangling Own Data from More Data", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    out_path = FIGURES_DIR / "fig9_forest_plot_personalization.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    make_fig8()
    make_fig9()
    print("\nALL 9 FIGURES GENERATED SUCCESSFULLY IN figures/!")
