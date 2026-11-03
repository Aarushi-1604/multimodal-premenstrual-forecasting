"""
src/data/build_master.py
Constructs the master daily table for the 2022 interval of mcPHASES.
Includes causal cycle onset detection, strictly causal cycle features,
ordinal symptom mapping, wearable joins with verified day semantics,
and next-day (t+1) targets with valid supervision masks.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("mcphases_data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "master_daily_2022.csv"

ORDINAL_SYMPTOM_MAP = {
    "Not at all": 0,
    "Very Low/Little": 1,
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5,
}

CORE_SYMPTOMS = ["cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches"]
ALL_SYMPTOMS = [
    "cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches",
    "appetite", "exerciselevel", "sleepissue", "stress", "foodcravings", "indigestion"
]


def detect_causal_cycle_onsets(df_core: pd.DataFrame) -> pd.DataFrame:
    """
    Biological cycle onset detection (validated logic):
    - bleeding = flow_volume.notna() & (flow_volume != 'Not at all')
    - bridge gaps <= 2 days within an episode
    - merge any episode starting < 15 days after previous episode start
    
    Returns a dataframe of causal cycle metadata for every (id, day_in_study).
    At day t, ONLY onsets <= t are used to define cycle_id and days_since_onset.
    cycle_length_prior uses only the elapsed duration between onsets already completed before day t.
    """
    records = []
    
    for participant_id, p_df in df_core.groupby("id"):
        p_df = p_df.sort_values("day_in_study").copy()
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
            
        # Step 2: Extract raw onsets (start of each episode)
        raw_onsets = [ep[0] for ep in episodes]
        
        # Step 3: Merge onsets starting < 15 days after previous onset
        merged_onsets = []
        for onset in raw_onsets:
            if not merged_onsets:
                merged_onsets.append(onset)
            else:
                if onset - merged_onsets[-1] >= 15:
                    merged_onsets.append(onset)
                # else: ignore as spotting / secondary bleed within same cycle
                
        # Step 4: Causal alignment for every day t in participant timeline
        # Calculate historical cycle lengths strictly from past onsets
        for _, row in p_df.iterrows():
            t = row["day_in_study"]
            past_onsets = [o for o in merged_onsets if o <= t]
            
            if not past_onsets:
                # Prior to first detected bleed in study
                c_id = 0
                days_since = np.nan
                prior_len = np.nan
                has_prior = 0
            else:
                curr_onset = past_onsets[-1]
                c_id = len(past_onsets)  # 1-indexed count of observed cycles
                days_since = t - curr_onset
                
                if len(past_onsets) >= 2:
                    # Prior cycle length is elapsed time between previous two onsets
                    prior_len = curr_onset - past_onsets[-2]
                    has_prior = 1
                else:
                    prior_len = np.nan
                    has_prior = 0
                    
            records.append({
                "id": participant_id,
                "day_in_study": t,
                "bleeding_active": int(row["bleeding"]),
                "cycle_id": c_id,
                "days_since_onset": days_since,
                "cycle_length_prior": prior_len,
                "has_prior_cycle": has_prior,
            })
            
    df_cycle_meta = pd.DataFrame(records)
    return df_cycle_meta


def build_master_table():
    print("=" * 70)
    print("BUILDING MASTER DAILY TABLE (mcPHASES 2022)")
    print("=" * 70)
    
    # 1. Base grid: hormones_and_selfreport (2022)
    core_path = DATA_DIR / "hormones_and_selfreport.csv"
    assert core_path.exists(), f"Missing file: {core_path}"
    df_core = pd.read_csv(core_path)
    df_base = df_core[df_core["study_interval"] == 2022].copy()
    
    n_base = len(df_base)
    print(f"[Base Grid] Loaded {n_base} rows, {df_base['id'].nunique()} participants.")
    assert df_base.duplicated(subset=["id", "day_in_study"]).sum() == 0, "Duplicate keys in base grid!"
    
    # 2. Map Symptoms to Ordinal Scale
    print("\n[Encoding Symptoms] Mapping text to ordinal {0..5}...")
    for sym in ALL_SYMPTOMS:
        assert sym in df_base.columns, f"Symptom column missing: {sym}"
        # Validate that all non-null values are covered in ORDINAL_SYMPTOM_MAP
        non_null_vals = df_base[sym].dropna().unique()
        unmapped = set(non_null_vals) - set(ORDINAL_SYMPTOM_MAP.keys())
        assert len(unmapped) == 0, f"Unmapped values found in {sym}: {unmapped}"
        
        df_base[f"{sym}_ord"] = df_base[sym].map(ORDINAL_SYMPTOM_MAP)
        df_base[f"{sym}_was_missing"] = df_base[sym].isna().astype(int)

    # 3. Clean Hormone features (LH, Estrogen; drop PdG)
    print("[Hormones] Parsing lh and estrogen...")
    df_base["lh"] = pd.to_numeric(df_base["lh"], errors="coerce")
    df_base["estrogen"] = pd.to_numeric(df_base["estrogen"], errors="coerce")
    df_base["lh_was_missing"] = df_base["lh"].isna().astype(int)
    df_base["estrogen_was_missing"] = df_base["estrogen"].isna().astype(int)

    # 4. Generate Causal Cycle Features
    print("[Cycle Features] Computing causal onset detection and phase encodings...")
    df_cycle_features = detect_causal_cycle_onsets(df_base)
    
    # Merge cycle features (cardinality 1:1)
    df_base = pd.merge(df_base, df_cycle_features, on=["id", "day_in_study"], how="left", validate="one_to_one")
    assert len(df_base) == n_base, "Row multiplication during cycle feature merge!"

    # Compute causal cyclic features:
    # (a) Population-prior based: tau_pop = 29.0 days
    tau_pop = 29.0
    ratio_pop = np.clip(df_base["days_since_onset"].fillna(0) / tau_pop, 0.0, 1.0)
    theta_pop = 2.0 * np.pi * ratio_pop
    df_base["cycle_sin_pop"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.sin(theta_pop))
    df_base["cycle_cos_pop"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.cos(theta_pop))

    # (b) Prior-cycle based (fallback to tau_pop if missing or biologically aberrant < 15 or > 65)
    valid_prior_len = df_base["cycle_length_prior"].between(15, 65)
    tau_effective = np.where(valid_prior_len, df_base["cycle_length_prior"], tau_pop)
    ratio_prior = np.clip(df_base["days_since_onset"].fillna(0) / tau_effective, 0.0, 1.0)
    theta_prior = 2.0 * np.pi * ratio_prior
    df_base["cycle_sin_prior"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.sin(theta_prior))
    df_base["cycle_cos_prior"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.cos(theta_prior))

    # 5. Join Resting Heart Rate
    print("\n[Merging Resting Heart Rate]...")
    df_rhr = pd.read_csv(DATA_DIR / "resting_heart_rate.csv")
    df_rhr_2022 = df_rhr[df_rhr["study_interval"] == 2022].copy()
    assert df_rhr_2022.duplicated(subset=["id", "day_in_study"]).sum() == 0, "Duplicate keys in RHR!"
    
    # Value parsing: 0.0 is sensor failure, convert to NaN
    df_rhr_2022["rhr"] = pd.to_numeric(df_rhr_2022["value"], errors="coerce")
    zeros_count = (df_rhr_2022["rhr"] == 0.0).sum()
    print(f"  RHR zero values treated as invalid/missing: {zeros_count}")
    df_rhr_2022.loc[df_rhr_2022["rhr"] == 0.0, "rhr"] = np.nan
    df_rhr_2022["rhr_was_missing"] = df_rhr_2022["rhr"].isna().astype(int)
    
    rhr_keep = df_rhr_2022[["id", "day_in_study", "rhr", "rhr_was_missing"]]
    df_base = pd.merge(df_base, rhr_keep, on=["id", "day_in_study"], how="left", validate="one_to_one")
    assert len(df_base) == n_base, "Row multiplication during RHR merge!"
    print(f"  Rows after RHR merge: {len(df_base)} (matched: {df_base['rhr'].notna().sum()}/{n_base})")

    # 6. Join Sleep (Main Sleep using sleep_end_day_in_study)
    print("\n[Merging Sleep (Main Sleep)]...")
    df_sleep = pd.read_csv(DATA_DIR / "sleep.csv")
    df_sleep_2022 = df_sleep[
        (df_sleep["study_interval"] == 2022) & 
        (df_sleep["mainsleep"].isin([True, 1, "True", "TRUE"]))
    ].copy()
    
    # Map sleep_end_day_in_study -> day_in_study
    df_sleep_2022["day_in_study"] = df_sleep_2022["sleep_end_day_in_study"]
    assert df_sleep_2022.duplicated(subset=["id", "day_in_study"]).sum() == 0, "Duplicate keys in sleep_end_day_in_study!"
    
    sleep_cols = ["minutesasleep", "minutesawake", "efficiency", "timeinbed"]
    for sc in sleep_cols:
        df_sleep_2022[f"sleep_{sc}"] = pd.to_numeric(df_sleep_2022[sc], errors="coerce")
        
    df_sleep_2022["sleep_was_missing"] = 0
    sleep_keep = df_sleep_2022[["id", "day_in_study"] + [f"sleep_{sc}" for sc in sleep_cols] + ["sleep_was_missing"]]
    
    df_base = pd.merge(df_base, sleep_keep, on=["id", "day_in_study"], how="left", validate="one_to_one")
    assert len(df_base) == n_base, "Row multiplication during Sleep merge!"
    df_base["sleep_was_missing"] = df_base["sleep_was_missing"].fillna(1).astype(int)
    print(f"  Rows after Sleep merge: {len(df_base)} (sleep logged: {(df_base['sleep_was_missing'] == 0).sum()}/{n_base})")

    # 7. Join Computed Temperature using sleep_end_day_in_study
    print("\n[Merging Computed Temperature]...")
    df_temp = pd.read_csv(DATA_DIR / "computed_temperature.csv")
    df_temp_2022 = df_temp[df_temp["study_interval"] == 2022].copy()
    
    df_temp_2022["day_in_study"] = df_temp_2022["sleep_end_day_in_study"]
    assert df_temp_2022.duplicated(subset=["id", "day_in_study"]).sum() == 0, "Duplicate keys in computed_temperature!"
    
    df_temp_2022["temp_nightly"] = pd.to_numeric(df_temp_2022["nightly_temperature"], errors="coerce")
    df_temp_2022["temp_sd_baseline"] = pd.to_numeric(df_temp_2022["baseline_relative_nightly_standard_deviation"], errors="coerce")
    df_temp_2022["temp_was_missing"] = 0
    
    temp_keep = df_temp_2022[["id", "day_in_study", "temp_nightly", "temp_sd_baseline", "temp_was_missing"]]
    df_base = pd.merge(df_base, temp_keep, on=["id", "day_in_study"], how="left", validate="one_to_one")
    assert len(df_base) == n_base, "Row multiplication during Temperature merge!"
    df_base["temp_was_missing"] = df_base["temp_was_missing"].fillna(1).astype(int)
    print(f"  Rows after Temp merge: {len(df_base)} (temp logged: {(df_base['temp_was_missing'] == 0).sum()}/{n_base})")

    # 8. Construct Next-Day Targets (t+1) with Strict Supervision Masks
    print("\n[Constructing Next-Day Targets (t+1)]...")
    # Sort strictly by participant and chronological day
    df_base = df_base.sort_values(["id", "day_in_study"]).reset_index(drop=True)
    
    for sym in CORE_SYMPTOMS:
        # Shift symptom ordinal by -1 grouped by id
        shifted_ord = df_base.groupby("id")[f"{sym}_ord"].shift(-1)
        shifted_day = df_base.groupby("id")["day_in_study"].shift(-1)
        
        # Valid next day means day_{t+1} exists AND day_{t+1} == day_t + 1 (consecutive) AND target symptom was logged (not NaN)
        is_consecutive_next_day = (shifted_day == (df_base["day_in_study"] + 1))
        is_target_logged = shifted_ord.notna()
        
        mask = (is_consecutive_next_day & is_target_logged).astype(int)
        
        # Fixed Moderate+ threshold (ord >= 3: Moderate, High, Very High)
        target = np.where(mask == 1, (shifted_ord >= 3).astype(int), np.nan)
        
        df_base[f"target_{sym}_t1"] = target
        df_base[f"mask_{sym}_t1"] = mask

    # 9. Clean Final Columns Selection
    selected_cols = [
        "id", "day_in_study", "is_weekend", "phase", "flow_volume",
        "bleeding_active", "cycle_id", "days_since_onset", "cycle_length_prior", "has_prior_cycle",
        "cycle_sin_pop", "cycle_cos_pop", "cycle_sin_prior", "cycle_cos_prior",
        "lh", "lh_was_missing", "estrogen", "estrogen_was_missing",
        "rhr", "rhr_was_missing",
        "sleep_minutesasleep", "sleep_minutesawake", "sleep_efficiency", "sleep_timeinbed", "sleep_was_missing",
        "temp_nightly", "temp_sd_baseline", "temp_was_missing"
    ]
    # Add ordinal symptom features & indicators
    for sym in ALL_SYMPTOMS:
        selected_cols.extend([f"{sym}_ord", f"{sym}_was_missing"])
    # Add targets and masks
    for sym in CORE_SYMPTOMS:
        selected_cols.extend([f"target_{sym}_t1", f"mask_{sym}_t1"])

    df_master = df_base[selected_cols].copy()
    
    # Save output (both parquet and feather/csv if desired, parquet is standard & fast)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_master.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully wrote master table to: {OUTPUT_FILE}")
    print(f"Final shape: {df_master.shape}")

    # 10. Summary Target Statistics Report
    print("\n" + "=" * 70)
    print("TARGET CLASS DISTRIBUTION SUMMARY (Moderate+ Threshold >= 3 at t+1)")
    print("=" * 70)
    for sym in CORE_SYMPTOMS:
        mask = df_master[f"mask_{sym}_t1"] == 1
        pos = (df_master.loc[mask, f"target_{sym}_t1"] == 1).sum()
        neg = (df_master.loc[mask, f"target_{sym}_t1"] == 0).sum()
        tot_supervised = pos + neg
        unsupervised = len(df_master) - tot_supervised
        pos_rate = (pos / tot_supervised) * 100 if tot_supervised > 0 else 0
        print(f"[{sym:11s}] Supervised: {tot_supervised:4d} | Pos (>=3): {pos:4d} ({pos_rate:5.1f}%) | Neg (<3): {neg:4d} | Unsupervised/Missing t+1: {unsupervised:4d}")

    return df_master


if __name__ == "__main__":
    build_master_table()
