"""
verification/test_onset_causality.py
Tests whether any cycle onset feature at day t uses future information (rows > t).
We take the original causal logic from build_master.py and for every row t,
run it on the truncated history df[day_in_study <= t], then compare against the stored master table.
Strictly read-only, aggregate outputs only.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("mcphases_data")
MASTER_PATH = Path("data/processed/master_daily_2022.csv")

def detect_causal_cycle_onsets_single(p_df_truncated: pd.DataFrame, target_t: int):
    """
    Runs the exact logic from build_master.py lines 34-110 on a dataframe containing ONLY days <= target_t.
    """
    p_df = p_df_truncated.sort_values("day_in_study").copy()
    p_df["bleeding"] = p_df["flow_volume"].notna() & (p_df["flow_volume"] != "Not at all")
    
    bleed_days = p_df.loc[p_df["bleeding"], "day_in_study"].tolist()
    
    # Step 1: Bridge gaps <= 2 days into episodes
    episodes = []
    if bleed_days:
        curr_ep = [bleed_days[0]]
        for d in bleed_days[1:]:
            if d - curr_ep[-1] <= 3:  # gap of <= 2 unlogged/non-bleed days
                curr_ep.append(d)
            else:
                episodes.append(curr_ep)
                curr_ep = [d]
        episodes.append(curr_ep)
        
    # Step 2: Extract raw onsets
    raw_onsets = [ep[0] for ep in episodes]
    
    # Step 3: Merge onsets starting < 15 days after previous onset
    merged_onsets = []
    for onset in raw_onsets:
        if not merged_onsets:
            merged_onsets.append(onset)
        else:
            if onset - merged_onsets[-1] >= 15:
                merged_onsets.append(onset)
                
    # Step 4: Causal alignment at target_t
    past_onsets = [o for o in merged_onsets if o <= target_t]
    
    if not past_onsets:
        c_id = 0
        days_since = np.nan
        prior_len = np.nan
        has_prior = 0
    else:
        curr_onset = past_onsets[-1]
        c_id = len(past_onsets)
        days_since = target_t - curr_onset
        if len(past_onsets) >= 2:
            prior_len = curr_onset - past_onsets[-2]
            has_prior = 1
        else:
            prior_len = np.nan
            has_prior = 0
            
    # Cyclic calculations
    tau_pop = 29.0
    ratio_pop = np.clip((0.0 if np.isnan(days_since) else days_since) / tau_pop, 0.0, 1.0)
    theta_pop = 2.0 * np.pi * ratio_pop
    sin_pop = np.nan if np.isnan(days_since) else np.sin(theta_pop)
    cos_pop = np.nan if np.isnan(days_since) else np.cos(theta_pop)
    
    return {
        "cycle_id": c_id,
        "days_since_onset": days_since,
        "cycle_length_prior": prior_len,
        "has_prior_cycle": has_prior,
        "cycle_sin_pop": sin_pop,
        "cycle_cos_pop": cos_pop,
    }

def run_causality_test():
    print("=== TASK 3: CAUSALITY TRUNCATION TEST ===")
    df_raw = pd.read_csv(DATA_DIR / "hormones_and_selfreport.csv")
    df_raw_2022 = df_raw[df_raw["study_interval"] == 2022].copy()
    df_master = pd.read_csv(MASTER_PATH)
    
    print(f"Total master rows to test: {len(df_master)}")
    
    features_to_test = [
        "cycle_id",
        "days_since_onset",
        "cycle_length_prior",
        "has_prior_cycle",
        "cycle_sin_pop",
        "cycle_cos_pop",
    ]
    
    mismatch_counts = {f: 0 for f in features_to_test}
    max_abs_diffs = {f: 0.0 for f in features_to_test}
    
    # We test ALL 3698 rows
    total_tested = 0
    
    # Pre-group raw by id
    grouped_raw = {pid: grp.sort_values("day_in_study").copy() for pid, grp in df_raw_2022.groupby("id")}
    
    for idx, row in df_master.iterrows():
        pid = row["id"]
        t = int(row["day_in_study"])
        p_raw = grouped_raw[pid]
        p_truncated = p_raw[p_raw["day_in_study"] <= t]
        
        recomputed = detect_causal_cycle_onsets_single(p_truncated, t)
        
        for f in features_to_test:
            val_stored = row[f]
            val_recomp = recomputed[f]
            
            # Check NaN equality
            if np.isnan(val_stored) and np.isnan(val_recomp):
                continue
            elif np.isnan(val_stored) != np.isnan(val_recomp):
                mismatch_counts[f] += 1
                max_abs_diffs[f] = max(max_abs_diffs[f], 999.0)
            else:
                diff = abs(val_stored - val_recomp)
                if diff > 1e-5:
                    mismatch_counts[f] += 1
                    max_abs_diffs[f] = max(max_abs_diffs[f], float(diff))
                    
        total_tested += 1
        if total_tested % 1000 == 0:
            print(f"  Tested {total_tested}/{len(df_master)} rows...")
            
    print(f"\nCompleted testing {total_tested} rows.")
    results_table = []
    all_pass = True
    for f in features_to_test:
        status = "PASS" if mismatch_counts[f] == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {f:20s}: rows={total_tested} | mismatches={mismatch_counts[f]} | max_diff={max_abs_diffs[f]:.6f} | status={status}")
        results_table.append({
            "feature": f,
            "rows_tested": total_tested,
            "mismatches": mismatch_counts[f],
            "max_abs_diff": max_abs_diffs[f],
            "status": status
        })
    print(f"Overall Causality Test: {'PASS' if all_pass else 'FAIL'}")
    return results_table

if __name__ == "__main__":
    run_causality_test()
