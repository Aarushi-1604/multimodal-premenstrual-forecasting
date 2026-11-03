"""
Step S1 Read-Only Inspection Script
Checks schema, string categories, value counts, join keys, duplicates,
and numeric parseability across 2022 tables without printing participant-level records.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("mcphases_data")

def run_inspection():
    print("=" * 70)
    print("1. HORMONES AND SELF-REPORT (2022 Interval)")
    print("=" * 70)
    
    df_core = pd.read_csv(DATA_DIR / "hormones_and_selfreport.csv")
    df_core_2022 = df_core[df_core["study_interval"] == 2022].copy()
    print(f"Total rows in hormones_and_selfreport (2022): {len(df_core_2022)}")
    print(f"Unique participants: {df_core_2022['id'].nunique()}")
    
    # Check key uniqueness
    dup_core = df_core_2022.duplicated(subset=["id", "day_in_study"]).sum()
    print(f"Duplicates on (id, day_in_study): {dup_core}")
    
    # Value counts for all self-report symptoms & flow
    symptom_cols = [
        "cramps", "moodswing", "fatigue", "bloating", "sorebreasts", "headaches",
        "appetite", "exerciselevel", "sleepissue", "stress", "foodcravings", "indigestion"
    ]
    
    print("\n--- Value distributions for symptom columns (2022) ---")
    for col in symptom_cols:
        if col in df_core_2022.columns:
            vc = df_core_2022[col].value_counts(dropna=False)
            print(f"\n[{col}] total non-null: {df_core_2022[col].notna().sum()}/{len(df_core_2022)} ({df_core_2022[col].notna().mean()*100:.1f}%)")
            for val, cnt in vc.items():
                print(f"   {repr(val)}: {cnt} ({cnt/len(df_core_2022)*100:.2f}%)")
                
    print("\n--- Flow Volume distribution (2022) ---")
    vc_flow = df_core_2022["flow_volume"].value_counts(dropna=False)
    for val, cnt in vc_flow.items():
        print(f"   {repr(val)}: {cnt} ({cnt/len(df_core_2022)*100:.2f}%)")
        
    print("\n--- Phase distribution (2022) ---")
    vc_phase = df_core_2022["phase"].value_counts(dropna=False)
    for val, cnt in vc_phase.items():
        print(f"   {repr(val)}: {cnt} ({cnt/len(df_core_2022)*100:.2f}%)")
        
    # Hormones parsing check
    print("\n--- Hormone Numeric Verification ---")
    for h in ["lh", "estrogen", "pdg"]:
        non_null = df_core_2022[h].dropna()
        numeric_series = pd.to_numeric(df_core_2022[h], errors="coerce")
        unparseable = non_null.shape[0] - numeric_series.dropna().shape[0]
        print(f"   {h}: non-null count={non_null.shape[0]}, unparseable text count={unparseable}")
        if numeric_series.dropna().shape[0] > 0:
            print(f"      min={numeric_series.min():.2f}, median={numeric_series.median():.2f}, max={numeric_series.max():.2f}")

    print("\n" + "=" * 70)
    print("2. RESTING HEART RATE (2022 Interval)")
    print("=" * 70)
    df_rhr = pd.read_csv(DATA_DIR / "resting_heart_rate.csv")
    df_rhr_2022 = df_rhr[df_rhr["study_interval"] == 2022].copy()
    print(f"Total rows (2022): {len(df_rhr_2022)}")
    print(f"Unique participants: {df_rhr_2022['id'].nunique()}")
    dup_rhr = df_rhr_2022.duplicated(subset=["id", "day_in_study"]).sum()
    print(f"Duplicates on (id, day_in_study): {dup_rhr}")
    rhr_num = pd.to_numeric(df_rhr_2022["value"], errors="coerce")
    print(f"RHR 'value': non-null={df_rhr_2022['value'].notna().sum()}, unparseable={df_rhr_2022['value'].dropna().shape[0] - rhr_num.dropna().shape[0]}")
    print(f"RHR stats: min={rhr_num.min():.1f}, median={rhr_num.median():.1f}, max={rhr_num.max():.1f}")
    
    # Overlap with core
    core_keys = set(zip(df_core_2022["id"], df_core_2022["day_in_study"]))
    rhr_keys = set(zip(df_rhr_2022["id"], df_rhr_2022["day_in_study"]))
    overlap_rhr = len(core_keys.intersection(rhr_keys))
    print(f"RHR overlap with core (id, day_in_study): {overlap_rhr}/{len(core_keys)} ({overlap_rhr/len(core_keys)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("3. SLEEP TABLE (2022 Interval) & DATE ALIGNMENT")
    print("=" * 70)
    df_sleep = pd.read_csv(DATA_DIR / "sleep.csv")
    df_sleep_2022 = df_sleep[df_sleep["study_interval"] == 2022].copy()
    print(f"Total rows (2022): {len(df_sleep_2022)}")
    print(f"Unique participants: {df_sleep_2022['id'].nunique()}")
    
    print("\nSleep mainsleep values:")
    print(df_sleep_2022["mainsleep"].value_counts(dropna=False))
    
    # Filter for main sleep
    # Handle boolean or string or int
    is_main = df_sleep_2022["mainsleep"].isin([True, 1, "True", "TRUE", "true"])
    df_main_sleep = df_sleep_2022[is_main].copy()
    print(f"Rows with mainsleep == True: {len(df_main_sleep)}")
    
    # Inspect relationship between sleep_start_day_in_study and sleep_end_day_in_study
    diff_days = df_main_sleep["sleep_end_day_in_study"] - df_main_sleep["sleep_start_day_in_study"]
    print("\nDistribution of (sleep_end_day_in_study - sleep_start_day_in_study):")
    print(diff_days.value_counts(dropna=False))
    
    # Duplicate check on sleep_start_day vs sleep_end_day
    dup_start = df_main_sleep.duplicated(subset=["id", "sleep_start_day_in_study"]).sum()
    dup_end = df_main_sleep.duplicated(subset=["id", "sleep_end_day_in_study"]).sum()
    print(f"Main sleep duplicates on (id, sleep_start_day_in_study): {dup_start}")
    print(f"Main sleep duplicates on (id, sleep_end_day_in_study): {dup_end}")
    
    # Overlap comparison
    sleep_start_keys = set(zip(df_main_sleep["id"], df_main_sleep["sleep_start_day_in_study"]))
    sleep_end_keys = set(zip(df_main_sleep["id"], df_main_sleep["sleep_end_day_in_study"]))
    print(f"Main sleep overlap with core using sleep_start_day_in_study: {len(core_keys.intersection(sleep_start_keys))}/{len(core_keys)} ({len(core_keys.intersection(sleep_start_keys))/len(core_keys)*100:.1f}%)")
    print(f"Main sleep overlap with core using sleep_end_day_in_study:   {len(core_keys.intersection(sleep_end_keys))}/{len(core_keys)} ({len(core_keys.intersection(sleep_end_keys))/len(core_keys)*100:.1f}%)")
    
    # Sample timestamp inspection format without printing full id/row
    print("\nSleep timestamp formats (first 3 string lengths and sample time of day):")
    for tcol in ["sleep_start_timestamp", "sleep_end_timestamp"]:
        sample_times = df_main_sleep[tcol].dropna().astype(str).str.extract(r'(\d{2}:\d{2}:\d{2})')[0].head(5).tolist()
        print(f"   {tcol} times sample: {sample_times}")

    print("\n" + "=" * 70)
    print("4. COMPUTED TEMPERATURE (2022 Interval)")
    print("=" * 70)
    df_temp = pd.read_csv(DATA_DIR / "computed_temperature.csv")
    df_temp_2022 = df_temp[df_temp["study_interval"] == 2022].copy()
    print(f"Total rows (2022): {len(df_temp_2022)}")
    print(f"Unique participants: {df_temp_2022['id'].nunique()}")
    
    diff_temp_days = df_temp_2022["sleep_end_day_in_study"] - df_temp_2022["sleep_start_day_in_study"]
    print("\nDistribution of (sleep_end_day_in_study - sleep_start_day_in_study) in computed_temperature:")
    print(diff_temp_days.value_counts(dropna=False))
    
    dup_temp_start = df_temp_2022.duplicated(subset=["id", "sleep_start_day_in_study"]).sum()
    dup_temp_end = df_temp_2022.duplicated(subset=["id", "sleep_end_day_in_study"]).sum()
    print(f"Duplicates on (id, sleep_start_day_in_study): {dup_temp_start}")
    print(f"Duplicates on (id, sleep_end_day_in_study): {dup_temp_end}")
    
    temp_num = pd.to_numeric(df_temp_2022["nightly_temperature"], errors="coerce")
    print(f"Nightly temperature: non-null={df_temp_2022['nightly_temperature'].notna().sum()}, unparseable={df_temp_2022['nightly_temperature'].dropna().shape[0] - temp_num.dropna().shape[0]}")
    print(f"Nightly temperature stats: min={temp_num.min():.2f}, median={temp_num.median():.2f}, max={temp_num.max():.2f}")
    
    temp_start_keys = set(zip(df_temp_2022["id"], df_temp_2022["sleep_start_day_in_study"]))
    temp_end_keys = set(zip(df_temp_2022["id"], df_temp_2022["sleep_end_day_in_study"]))
    print(f"Temperature overlap with core using sleep_start_day_in_study: {len(core_keys.intersection(temp_start_keys))}/{len(core_keys)} ({len(core_keys.intersection(temp_start_keys))/len(core_keys)*100:.1f}%)")
    print(f"Temperature overlap with core using sleep_end_day_in_study:   {len(core_keys.intersection(temp_end_keys))}/{len(core_keys)} ({len(core_keys.intersection(temp_end_keys))/len(core_keys)*100:.1f}%)")

if __name__ == "__main__":
    run_inspection()
