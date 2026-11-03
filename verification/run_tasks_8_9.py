"""
verification/run_tasks_8_9.py
Computes:
TASK 8: Paired fold-level comparisons from results/all_models_groupkfold_folds.csv
TASK 9: Biological cycle counts and distributions on 2022 data.
Strictly read-only, aggregate outputs only.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

def run_task8():
    print("=== TASK 8: PAIRED FOLD-LEVEL SUMMARIES ===")
    df = pd.read_csv("results/all_models_groupkfold_folds.csv")
    symptoms = df["symptom"].unique()
    
    # We want:
    # 1. CatBoost vs GRU in PR-AUC
    # 2. CatBoost vs Persistence_LastObserved in PR-AUC
    # 3. CatBoost vs Persistence_LastObserved in Brier
    # (mean and std of paired difference, # folds CatBoost beats opponent out of 5)
    
    records = []
    for sym in symptoms:
        sub = df[df["symptom"] == sym]
        
        cb = sub[sub["model"] == "CatBoost"].sort_values("fold")
        gru = sub[sub["model"] == "GRU"].sort_values("fold")
        pers = sub[sub["model"] == "Persistence_LastObserved"].sort_values("fold")
        
        # 1. CatBoost vs GRU (PR-AUC)
        diff_cb_gru_pr = cb["PR-AUC"].values - gru["PR-AUC"].values
        wins_cb_gru_pr = int(np.sum(diff_cb_gru_pr > 0))
        mean_cb_gru_pr = float(np.mean(diff_cb_gru_pr))
        std_cb_gru_pr = float(np.std(diff_cb_gru_pr, ddof=1))
        
        # 2. CatBoost vs Persistence (PR-AUC)
        diff_cb_pers_pr = cb["PR-AUC"].values - pers["PR-AUC"].values
        wins_cb_pers_pr = int(np.sum(diff_cb_pers_pr > 0))
        mean_cb_pers_pr = float(np.mean(diff_cb_pers_pr))
        std_cb_pers_pr = float(np.std(diff_cb_pers_pr, ddof=1))
        
        # 3. CatBoost vs Persistence (Brier) - NOTE: lower Brier is better, so win means cb < pers (diff_brier = pers - cb > 0)
        # Or define diff as CB - Pers. Let's record both win count (CB < Pers) and diff (CB - Pers)
        diff_brier = cb["Brier"].values - pers["Brier"].values # negative means CB has lower (better) error
        wins_cb_pers_brier = int(np.sum(cb["Brier"].values < pers["Brier"].values))
        mean_diff_brier = float(np.mean(diff_brier))
        std_diff_brier = float(np.std(diff_brier, ddof=1))
        
        records.append({
            "symptom": sym,
            "cb_v_gru_pr_wins": f"{wins_cb_gru_pr}/5",
            "cb_v_gru_pr_diff": f"{mean_cb_gru_pr:+.4f} ± {std_cb_gru_pr:.4f}",
            "cb_v_pers_pr_wins": f"{wins_cb_pers_pr}/5",
            "cb_v_pers_pr_diff": f"{mean_cb_pers_pr:+.4f} ± {std_cb_pers_pr:.4f}",
            "cb_v_pers_brier_wins": f"{wins_cb_pers_brier}/5",
            "cb_v_pers_brier_diff": f"{mean_diff_brier:+.4f} ± {std_diff_brier:.4f}",
        })
        
    df_t8 = pd.DataFrame(records)
    print(df_t8.to_string(index=False))
    return df_t8

def run_task9():
    print("\n=== TASK 9: CYCLE COUNTS & DISTRIBUTIONS ===")
    df = pd.read_csv("data/processed/master_daily_2022.csv")
    
    # Extract unique onsets per participant
    # In master table: days where days_since_onset == 0
    onsets_df = df[df["days_since_onset"] == 0].sort_values(["id", "day_in_study"])
    total_detected_onsets = len(onsets_df)
    
    # Group by participant
    onsets_per_p = onsets_df.groupby("id")["day_in_study"].apply(list).to_dict()
    
    # Total participants
    all_pids = df["id"].unique()
    n_total_p = len(all_pids)
    
    p_with_ge1_onset = sum(1 for pid in all_pids if len(onsets_per_p.get(pid, [])) >= 1)
    p_with_ge2_onsets = sum(1 for pid in all_pids if len(onsets_per_p.get(pid, [])) >= 2)
    
    # Compute all onset-to-onset intervals
    intervals = []
    p_with_ge1_interval = 0
    for pid in all_pids:
        o_list = onsets_per_p.get(pid, [])
        if len(o_list) >= 2:
            p_with_ge1_interval += 1
            for k in range(len(o_list) - 1):
                interval_len = o_list[k+1] - o_list[k]
                intervals.append(interval_len)
                
    intervals = np.array(intervals)
    total_intervals = len(intervals)
    
    med_len = float(np.median(intervals))
    mean_len = float(np.mean(intervals))
    q25 = float(np.percentile(intervals, 25))
    q75 = float(np.percentile(intervals, 75))
    iqr_len = q75 - q25
    
    lt20 = int(np.sum(intervals < 20))
    gt45 = int(np.sum(intervals > 45))
    
    print(f"Total detected onsets: {total_detected_onsets}")
    print(f"Total onset-to-onset intervals: {total_intervals}")
    print(f"Participants with >= 1 onset: {p_with_ge1_onset}/{n_total_p}")
    print(f"Participants with >= 2 onsets: {p_with_ge2_onsets}/{n_total_p}")
    print(f"Participants with >= 1 interval: {p_with_ge1_interval}/{n_total_p}")
    print(f"Median interval length: {med_len:.1f} days")
    print(f"Mean interval length: {mean_len:.2f} days")
    print(f"IQR: [{q25:.1f}, {q75:.1f}] (width: {iqr_len:.1f} days)")
    print(f"Intervals < 20 days: {lt20}")
    print(f"Intervals > 45 days: {gt45}")

if __name__ == "__main__":
    run_task8()
    run_task9()
