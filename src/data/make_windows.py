"""
src/data/make_windows.py
Extracts consecutive 7-day input windows (W=7) and next-day (t+1) targets.
Implements:
1. Participant-boundary and consecutive-day verification.
2. GroupKFold (5-fold) by participant id for unseen-participant evaluation.
3. Forward-chaining partition (earlier cycles -> final cycle) for personalization.
4. Causal feature matrix extraction without phase or future targets.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA_PATH = Path("data/processed/master_daily_2022.csv")
OUTPUT_DIR = Path("data/processed")

CORE_SYMPTOMS = ["cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches"]
ALL_SYMPTOMS = [
    "cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches",
    "appetite", "exerciselevel", "sleepissue", "stress", "foodcravings", "indigestion"
]

# Exact feature list per day (D = 41 features)
DAILY_FEATURES = [
    # 1. Self-report symptoms (12 ordinals + 12 indicators = 24)
    *[f"{s}_ord" for s in ALL_SYMPTOMS],
    *[f"{s}_was_missing" for s in ALL_SYMPTOMS],
    # 2. Hormones (2 features + 2 indicators = 4)
    "lh", "lh_was_missing", "estrogen", "estrogen_was_missing",
    # 3. Wearables (4 features + 4 indicators = 8)
    "rhr", "rhr_was_missing",
    "sleep_minutesasleep", "sleep_minutesawake", "sleep_efficiency", "sleep_timeinbed", "sleep_was_missing",
    "temp_nightly", "temp_sd_baseline", "temp_was_missing",
    # 4. Cycle & calendar context (5 features)
    "days_since_onset", "has_prior_cycle", "cycle_sin_pop", "cycle_cos_pop", "is_weekend"
]


def create_windows(window_size: int = 7):
    print("=" * 70)
    print("CONSTRUCTING 7-DAY WINDOWS AND DUAL SPLITS")
    print("=" * 70)
    
    assert DATA_PATH.exists(), f"Master table missing at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    
    # Sort strictly by participant and chronological day
    df = df.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    
    # Verify daily feature columns exist
    missing_cols = set(DAILY_FEATURES) - set(df.columns)
    assert len(missing_cols) == 0, f"Missing features in master table: {missing_cols}"
    
    window_records = []
    
    # Group by participant
    for p_id, p_df in df.groupby("id"):
        p_df = p_df.sort_values("day_in_study").reset_index(drop=True)
        days = p_df["day_in_study"].values
        n_days = len(p_df)
        
        if n_days <= window_size:
            continue
            
        for i in range(window_size - 1, n_days - 1):
            # Check window consecutiveness: days[i - window_size + 1 : i + 1]
            win_days = days[i - window_size + 1 : i + 1]
            expected_days = np.arange(win_days[0], win_days[0] + window_size)
            if not np.array_equal(win_days, expected_days):
                # Gap in study days within the 7-day window -> discard
                continue
                
            # Check target day consecutiveness: day[i+1] == day[i] + 1
            if days[i + 1] != days[i] + 1:
                # Target day is not the consecutive next calendar day -> discard
                continue
                
            query_row = p_df.iloc[i]
            target_row = p_df.iloc[i + 1]
            
            # Extract target vector and mask vector
            target_vals = []
            mask_vals = []
            for s in CORE_SYMPTOMS:
                # Target was already constructed in build_master.py
                target_val = query_row[f"target_{s}_t1"]
                mask_val = int(query_row[f"mask_{s}_t1"])
                target_vals.append(target_val)
                mask_vals.append(mask_val)
                
            # Window data slice
            win_df = p_df.iloc[i - window_size + 1 : i + 1]
            feature_matrix = win_df[DAILY_FEATURES].values.astype(np.float32)
            
            # Metadata
            window_records.append({
                "participant_id": p_id,
                "query_day": int(query_row["day_in_study"]),
                "target_day": int(target_row["day_in_study"]),
                "cycle_id": int(query_row["cycle_id"]),
                "target_cycle_id": int(target_row["cycle_id"]),
                "features": feature_matrix,
                "targets": np.array(target_vals, dtype=np.float32),
                "masks": np.array(mask_vals, dtype=np.int32),
            })
            
    print(f"Total valid consecutive 7-day windows constructed: {len(window_records)}")
    
    # -------------------------------------------------------------
    # Partition 1: GroupKFold (5 Folds) by participant id
    # -------------------------------------------------------------
    participant_ids = np.array([r["participant_id"] for r in window_records])
    unique_participants = np.unique(participant_ids)
    print(f"Participants represented in windows: {len(unique_participants)}/42")
    
    gkf = GroupKFold(n_splits=5)
    dummy_x = np.zeros(len(window_records))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(dummy_x, groups=participant_ids)):
        for idx in val_idx:
            window_records[idx]["fold"] = fold
            
    # -------------------------------------------------------------
    # Partition 2: Forward-Chaining (Personalization)
    # -------------------------------------------------------------
    # Identify max cycle_id per participant in the dataset
    max_cycles = {}
    for p_id, p_df in df.groupby("id"):
        max_c = p_df["cycle_id"].max()
        max_cycles[p_id] = max_c
        
    for r in window_records:
        p_id = r["participant_id"]
        target_cycle = r["target_cycle_id"]
        max_c = max_cycles[p_id]
        
        # Eligible for personalization evaluation if participant has >= 2 detected cycles
        if max_c >= 2:
            r["personalization_eligible"] = 1
            if target_cycle == max_c:
                r["fc_split"] = "test"  # final cycle test
            elif target_cycle < max_c and target_cycle >= 1:
                r["fc_split"] = "train" # earlier cycles
            else:
                r["fc_split"] = "unaligned" # prior to cycle 1
        else:
            r["personalization_eligible"] = 0
            r["fc_split"] = "ineligible"

    # Save to disk as an npz / pickle dictionary
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pack into numpy arrays for high-performance dataloading
    X = np.stack([r["features"] for r in window_records], axis=0) # (N, 7, 41)
    Y = np.stack([r["targets"] for r in window_records], axis=0)  # (N, 6)
    M = np.stack([r["masks"] for r in window_records], axis=0)    # (N, 6)
    folds = np.array([r["fold"] for r in window_records], dtype=np.int32)
    p_ids = np.array([r["participant_id"] for r in window_records])
    q_days = np.array([r["query_day"] for r in window_records], dtype=np.int32)
    t_days = np.array([r["target_day"] for r in window_records], dtype=np.int32)
    fc_splits = np.array([r["fc_split"] for r in window_records])
    pers_eligible = np.array([r["personalization_eligible"] for r in window_records], dtype=np.int32)
    
    np.savez_compressed(
        OUTPUT_DIR / "windows_dataset.npz",
        X=X, Y=Y, M=M,
        folds=folds,
        p_ids=p_ids,
        q_days=q_days,
        t_days=t_days,
        fc_splits=fc_splits,
        pers_eligible=pers_eligible,
        feature_names=np.array(DAILY_FEATURES),
        target_names=np.array(CORE_SYMPTOMS)
    )
    print(f"Saved window dataset to: {OUTPUT_DIR / 'windows_dataset.npz'}")
    
    # -------------------------------------------------------------
    # Report Statistics per Split and Target
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET SPLIT & TARGET DISTRIBUTION REPORT")
    print("=" * 70)
    
    # GroupKFold summary
    print("\n--- GroupKFold (Unseen Participants) ---")
    for f in range(5):
        f_idx = np.where(folds == f)[0]
        f_pids = np.unique(p_ids[f_idx])
        print(f"Fold {f}: {len(f_idx):4d} windows | {len(f_pids):2d} participants (IDs: {list(f_pids)[:4]}...)")
        
    # Forward chaining summary
    print("\n--- Within-Person Forward Chaining (Personalization) ---")
    n_fc_train = (fc_splits == "train").sum()
    n_fc_test = (fc_splits == "test").sum()
    n_fc_inelig = (fc_splits == "ineligible").sum()
    n_fc_unaligned = (fc_splits == "unaligned").sum()
    print(f"FC Train (Earlier cycles): {n_fc_train:4d} windows")
    print(f"FC Test  (Final cycle):    {n_fc_test:4d} windows")
    print(f"FC Ineligible (<2 cycles): {n_fc_inelig:4d} windows")
    print(f"FC Pre-cycle 1:            {n_fc_unaligned:4d} windows")
    
    # Target prevalence in window dataset
    print("\n--- Target Prevalence in Windowed Dataset (N = {}) ---".format(len(window_records)))
    for s_idx, sym in enumerate(CORE_SYMPTOMS):
        m_s = M[:, s_idx] == 1
        pos = (Y[m_s, s_idx] == 1).sum()
        tot = m_s.sum()
        unsupervised = len(window_records) - tot
        pct = (pos / tot) * 100 if tot > 0 else 0
        print(f"[{sym:11s}] Supervised: {tot:4d} | Pos (>=3): {int(pos):4d} ({pct:5.1f}%) | Neg: {tot - int(pos):4d} | Masked/Missing: {unsupervised:4d}")


if __name__ == "__main__":
    create_windows()
