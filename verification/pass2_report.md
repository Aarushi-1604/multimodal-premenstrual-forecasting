# VERIFICATION PASS 2 REPORT: AUDIT EVIDENCE & AGGREGATE TESTS

**Project:** `C:\Users\aarus\Desktop\DeepLearningProject`  
**Date:** October 19, 2026  
**Scope:** Strictly read-only on all existing models, datasets, code, and results. All experiments and tests executed in isolated verification scripts (`verification/`). Redacted aggregate outputs only (DUA compliant, zero participant IDs, zero raw dates).

---

## 1. Feature Inventory (`windows_dataset.npz` -> `feature_names`, 43 features)

### Complete 43-Feature Table in $X$

| # | Feature Name | Source Table & Column | Built in File:Line | Uses only data $\le t$? | Is Missingness Indicator? |
|---|---|---|---|---|---|
| 0 | `cramps_ord` | `hormones_and_selfreport.csv` -> `cramps` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 1 | `moodswing_ord` | `hormones_and_selfreport.csv` -> `moodswing` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 2 | `fatigue_ord` | `hormones_and_selfreport.csv` -> `fatigue` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 3 | `bloating_ord` | `hormones_and_selfreport.csv` -> `bloating` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 4 | `sorebreasts_ord` | `hormones_and_selfreport.csv` -> `sorebreasts` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 5 | `headaches_ord` | `hormones_and_selfreport.csv` -> `headaches` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 6 | `appetite_ord` | `hormones_and_selfreport.csv` -> `appetite` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 7 | `exerciselevel_ord` | `hormones_and_selfreport.csv` -> `exerciselevel` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 8 | `sleepissue_ord` | `hormones_and_selfreport.csv` -> `sleepissue` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 9 | `stress_ord` | `hormones_and_selfreport.csv` -> `stress` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 10 | `foodcravings_ord` | `hormones_and_selfreport.csv` -> `foodcravings` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 11 | `indigestion_ord` | `hormones_and_selfreport.csv` -> `indigestion` | `build_master.py:73` | Yes | No (Ordinal $0\dots5$) |
| 12 | `cramps_was_missing` | Derived from missingness of `cramps` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 13 | `moodswing_was_missing` | Derived from missingness of `moodswing` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 14 | `fatigue_was_missing` | Derived from missingness of `fatigue` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 15 | `bloating_was_missing` | Derived from missingness of `bloating` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 16 | `sorebreasts_was_missing` | Derived from missingness of `sorebreasts` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 17 | `headaches_was_missing` | Derived from missingness of `headaches` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 18 | `appetite_was_missing` | Derived from missingness of `appetite` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 19 | `exerciselevel_was_missing` | Derived from missingness of `exerciselevel` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 20 | `sleepissue_was_missing` | Derived from missingness of `sleepissue` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 21 | `stress_was_missing` | Derived from missingness of `stress` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 22 | `foodcravings_was_missing` | Derived from missingness of `foodcravings` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 23 | `indigestion_was_missing` | Derived from missingness of `indigestion` | `build_master.py:74` | Yes | Yes (Binary 0/1) |
| 24 | `lh` | `hormones_and_selfreport.csv` -> `lh` | `build_master.py:78` | Yes | No (Numeric $\text{mIU/mL}$) |
| 25 | `lh_was_missing` | Derived from missingness of `lh` | `build_master.py:80` | Yes | Yes (Binary 0/1) |
| 26 | `estrogen` | `hormones_and_selfreport.csv` -> `estrogen` | `build_master.py:79` | Yes | No (Numeric $\text{pg/mL}$) |
| 27 | `estrogen_was_missing` | Derived from missingness of `estrogen` | `build_master.py:81` | Yes | Yes (Binary 0/1) |
| 28 | `rhr` | `resting_heart_rate.csv` -> `value` | `build_master.py:182-185` | Yes | No (Numeric bpm) |
| 29 | `rhr_was_missing` | Derived from missingness or $0.0$ in `rhr` | `build_master.py:186` | Yes | Yes (Binary 0/1) |
| 30 | `sleep_minutesasleep` | `sleep.csv` -> `minutesasleep` | `build_master.py:207` | Yes | No (Numeric min) |
| 31 | `sleep_minutesawake` | `sleep.csv` -> `minutesawake` | `build_master.py:207` | Yes | No (Numeric min) |
| 32 | `sleep_efficiency` | `sleep.csv` -> `efficiency` | `build_master.py:207` | Yes | No (Numeric %) |
| 33 | `sleep_timeinbed` | `sleep.csv` -> `timeinbed` | `build_master.py:207` | Yes | No (Numeric min) |
| 34 | `sleep_was_missing` | Derived from unlogged nocturnal sleep | `build_master.py:214` | Yes | Yes (Binary 0/1) |
| 35 | `temp_nightly` | `computed_temperature.csv` -> `nightly_temperature` | `build_master.py:225` | Yes | No (Numeric °C) |
| 36 | `temp_sd_baseline` | `computed_temperature.csv` -> `baseline_relative_nightly_standard_deviation` | `build_master.py:226` | Yes | No (Numeric SD ratio) |
| 37 | `temp_was_missing` | Derived from missing temperature | `build_master.py:231` | Yes | Yes (Binary 0/1) |
| 38 | `days_since_onset` | Causal elapsed days since most recent bleed onset $\le t$ | `build_master.py:93, 108` | Yes | No (Count days) |
| 39 | `has_prior_cycle` | Binary indicator: $\ge 2$ historical onsets observed $\le t$ | `build_master.py:98, 110` | Yes | No (Binary 0/1) |
| 40 | `cycle_sin_pop` | Trigonometric cyclic sine using $\tau_{\text{pop}}=29.0$ | `build_master.py:164` | Yes | No (Trigonometric scalar) |
| 41 | `cycle_cos_pop` | Trigonometric cyclic cosine using $\tau_{\text{pop}}=29.0$ | `build_master.py:165` | Yes | No (Trigonometric scalar) |
| 42 | `is_weekend` | `hormones_and_selfreport.csv` -> `is_weekend` | `build_master.py:25` | Yes | No (Binary 0/1) |

### Explicit Verification Answers:
- **Is any feature named or derived from `phase`?** **NO (PASS).** `phase` is completely absent from $X$. Checked via `any('phase' in f.lower() for f in feature_names) == False`.
- **Are `lh`/`estrogen`/`pdg` inputs?** **`lh` and `estrogen` ARE inputs** (along with missingness flags). **`pdg` is NOT an input** (`pdg` is 100% missing in 2022 and was dropped).
- **Which column feeds temperature and what unit?** **`temp_nightly`** feeds absolute nightly temperature in **degrees Celsius (°C)** (mean 33.90°C, std 0.95°C, range 25.70°C to 36.43°C from Fitbit dermal sensor), and **`temp_sd_baseline`** feeds `baseline_relative_nightly_standard_deviation` (unitless standard deviation ratio, mean 0.60, std 0.28).
- **Is resting-HR 0 converted to NaN before any statistic is computed?** **YES (PASS).** Handled in `src/data/build_master.py` lines 181–186:
  ```python
  df_rhr_2022["rhr"] = pd.to_numeric(df_rhr_2022["value"], errors="coerce")
  df_rhr_2022.loc[df_rhr_2022["rhr"] == 0.0, "rhr"] = np.nan
  df_rhr_2022["rhr_was_missing"] = df_rhr_2022["rhr"].isna().astype(int)
  ```
  All 782 zero values are replaced with `np.nan` prior to merging and imputation.

---

## 2. Target Mask $M$ & Reproduction Checks

### a) 30-Row Fold-by-Target Reproduction Table

Exact rule selecting evaluation rows in `results/all_models_groupkfold_folds.csv`:
$$\text{Row condition for Fold } k \text{ and Target } j: \quad (\text{folds} == k) \ \& \ (M[:, j] == 1)$$

| Fold | Symptom Target | Calculated $n_{\text{test}}$ | Result CSV $n_{\text{test}}$ | Calculated $n_{\text{pos}}$ | Result CSV $n_{\text{pos}}$ | Status |
|---|---|---|---|---|---|---|
| 0 | cramps | 654 | 654 | 148 | 148 | **PASS** |
| 0 | moodswing | 654 | 654 | 226 | 226 | **PASS** |
| 0 | fatigue | 654 | 654 | 387 | 387 | **PASS** |
| 0 | bloating | 654 | 654 | 223 | 223 | **PASS** |
| 0 | sorebreasts | 654 | 654 | 117 | 117 | **PASS** |
| 0 | headaches | 654 | 654 | 191 | 191 | **PASS** |
| 1 | cramps | 586 | 586 | 102 | 102 | **PASS** |
| 1 | moodswing | 586 | 586 | 273 | 273 | **PASS** |
| 1 | fatigue | 586 | 586 | 317 | 317 | **PASS** |
| 1 | bloating | 586 | 586 | 289 | 289 | **PASS** |
| 1 | sorebreasts | 586 | 586 | 75 | 75 | **PASS** |
| 1 | headaches | 586 | 586 | 169 | 169 | **PASS** |
| 2 | cramps | 601 | 601 | 37 | 37 | **PASS** |
| 2 | moodswing | 601 | 601 | 90 | 90 | **PASS** |
| 2 | fatigue | 601 | 601 | 352 | 352 | **PASS** |
| 2 | bloating | 601 | 601 | 76 | 76 | **PASS** |
| 2 | sorebreasts | 601 | 601 | 18 | 18 | **PASS** |
| 2 | headaches | 601 | 601 | 100 | 100 | **PASS** |
| 3 | cramps | 594 | 594 | 145 | 145 | **PASS** |
| 3 | moodswing | 594 | 594 | 175 | 175 | **PASS** |
| 3 | fatigue | 594 | 594 | 308 | 308 | **PASS** |
| 3 | bloating | 594 | 594 | 213 | 213 | **PASS** |
| 3 | sorebreasts | 594 | 594 | 118 | 118 | **PASS** |
| 3 | headaches | 594 | 594 | 188 | 188 | **PASS** |
| 4 | cramps | 601 | 601 | 49 | 49 | **PASS** |
| 4 | moodswing | 601 | 601 | 140 | 140 | **PASS** |
| 4 | fatigue | 601 | 601 | 338 | 338 | **PASS** |
| 4 | bloating | 601 | 601 | 137 | 137 | **PASS** |
| 4 | sorebreasts | 601 | 601 | 27 | 27 | **PASS** |
| 4 | headaches | 601 | 601 | 122 | 122 | **PASS** |

### b) Values Held by $Y$ where $M == 0$
In `data/processed/windows_dataset.npz`:
- For all 6 targets, whenever $M[:, j] == 0$, **$Y[:, j]$ is strictly `NaN`**.
- Count where $M == 0$ and $Y$ is NaN: **368 per target** (100% of masked positions).
- Count where $M == 0$ and $Y == 0$: **0 per target**.
- Count where $M == 0$ and $Y == 1$: **0 per target**.

### c) Code Excerpts Where Mask $M$ is Used

**1. Masked BCE Loss Function (`src/models/train_neural.py` L103-L111):**
```python
def compute_masked_loss(logits, targets, masks, pos_weights):
    # Sanitize unlogged targets (where mask == 0, target is NaN) so BCE doesn't propagate NaN
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    loss_matrix = bce(logits, safe_targets) # (batch, 6)
    masked_loss = loss_matrix * masks
    denom = masks.sum() + 1e-7
    return masked_loss.sum() / denom
```

**2. Row Selection for LogReg / CatBoost (`src/models/baselines.py` L80-L85):**
```python
            # Supervised masks for this specific symptom
            train_sup = train_mask & (M[:, s_idx] == 1)
            test_sup = test_mask & (M[:, s_idx] == 1)
            
            X_tr, y_tr = X_flat[train_sup], Y[train_sup, s_idx]
            X_te, y_te = X_flat[test_sup], Y[test_sup, s_idx]
```

**3. Metric Computation (`src/models/train_neural.py` L402-L410):**
```python
            for s_idx, sym in enumerate(target_names):
                t_mask = te_masks[:, s_idx] == 1
                y_true = te_targets[t_mask, s_idx]
                y_prob = te_probs[t_mask, s_idx]
                y_pred = (y_prob >= 0.5).astype(int)

                res = eval_predictions(y_true, y_pred, y_prob)
```

**4. Persistence Baseline Row Selection (`src/models/train_neural.py` L163-L166):**
```python
        for f in range(5):
            test_mask = (folds == f) & (M[:, s_idx] == 1)
            y_te = Y[test_mask, s_idx]
            win_sym_vals = X[test_mask, :, sym_feat_idx] # (N_te, 7)
```

### d) Synthetic Unit Test for Masked Loss Invariance
- Implemented and run in `verification/run_pass2_checks.py`.
- Tensors with random logits, targets, and binary masks were generated. Targets at $M == 0$ were modified from `NaN` to arbitrary constants ($999.0$).
- Output: `loss_diff = 0.0`, `max_grad_diff = 0.0`.
- **Status:** **PASS**.

### e) Code Context for `fillna(0)` / `nan_to_num` / `np.where(isnan)`

1. **`src/models/baselines.py` L101-L105:**
   ```python
   # Raw day t value (unimputed, if missing -> 0)
   day_t_vals = X_flat[test_sup, day_t_feature_idx]
   y_pred_persist = np.where(np.isnan(day_t_vals), 0, (day_t_vals >= 3)).astype(int)
   y_prob_persist = y_pred_persist.astype(float)
   ```
   *Modifies:* Only the original S4 persistence prediction. (Superceded in S5 by Last Observed Value forward-fill).
2. **`src/models/train_neural.py` L103-L107:**
   ```python
   # Sanitize unlogged targets (where mask == 0, target is NaN) so BCE doesn't propagate NaN
   safe_targets = torch.nan_to_num(targets, nan=0.0)
   bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
   ```
   *Modifies:* Tensor copy of targets $Y$ solely at masked positions $M == 0$ before computing binary cross entropy.
3. **`src/data/build_master.py` L160-L165:**
   ```python
   tau_pop = 29.0
   ratio_pop = np.clip(df_base["days_since_onset"].fillna(0) / tau_pop, 0.0, 1.0)
   theta_pop = 2.0 * np.pi * ratio_pop
   df_base["cycle_sin_pop"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.sin(theta_pop))
   df_base["cycle_cos_pop"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.cos(theta_pop))
   ```
   *Modifies:* Intermediate series for computing trigonometric features. When `days_since_onset` is NaN (pre-cycle-1), outputs are restored to `np.nan`.
4. **`src/data/build_master.py` L168-L173:**
   ```python
   tau_effective = np.where(valid_prior_len, df_base["cycle_length_prior"], tau_pop)
   ratio_prior = np.clip(df_base["days_since_onset"].fillna(0) / tau_effective, 0.0, 1.0)
   theta_prior = 2.0 * np.pi * ratio_prior
   df_base["cycle_sin_prior"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.sin(theta_prior))
   df_base["cycle_cos_prior"] = np.where(df_base["days_since_onset"].isna(), np.nan, np.cos(theta_prior))
   ```
   *Modifies:* Intermediate series for prior-cycle trigonometric features.

---

## 3. Cycle-Onset Feature Causality (Highest Priority)

### a) Onset Detection Function (`src/data/build_master.py` L34-L114)
```python
def detect_causal_cycle_onsets(df_core: pd.DataFrame) -> pd.DataFrame:
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
                
        # Step 4: Causal alignment for every day t in participant timeline
        for _, row in p_df.iterrows():
            t = row["day_in_study"]
            past_onsets = [o for o in merged_onsets if o <= t]
            
            if not past_onsets:
                c_id = 0
                days_since = np.nan
                prior_len = np.nan
                has_prior = 0
            else:
                curr_onset = past_onsets[-1]
                c_id = len(past_onsets)
                days_since = t - curr_onset
                if len(past_onsets) >= 2:
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
    return pd.DataFrame(records)
```

**Causality Analysis:**
- In Step 4, row $t$ only inspects `past_onsets = [o for o in merged_onsets if o <= t]`.
- Does an onset date $o \le t$ depend on bleeding days after $t$?
  - An episode's onset is defined as its *first* bleeding day: `raw_onsets = [ep[0] for ep in episodes]`. Thus, whether an episode continues bleeding on day $t+1$ or $t+2$ never alters the start day `ep[0]`.
  - The 15-day merge rule only merges an onset if it is within 15 days of the *previous* onset; it never moves earlier.
  - Future onsets ($> t$) are filtered out (`o <= t`).
  - No future cycle length is ever used.

### b) Shift(-1) Target Construction (`src/data/build_master.py` L240-L255)
```python
    for sym in CORE_SYMPTOMS:
        # Shift symptom ordinal by -1 grouped by id
        shifted_ord = df_base.groupby("id")[f"{sym}_ord"].shift(-1)
        shifted_day = df_base.groupby("id")["day_in_study"].shift(-1)
        
        # Valid next day means day_{t+1} exists AND day_{t+1} == day_t + 1 (consecutive) AND target symptom was logged (not NaN)
        is_consecutive_next_day = (shifted_day == (df_base["day_in_study"] + 1))
        is_target_logged = shifted_ord.notna()
        
        mask = (is_consecutive_next_day & is_target_logged).astype(int)
        target = np.where(mask == 1, (shifted_ord >= 3).astype(int), np.nan)
        
        df_base[f"target_{sym}_t1"] = target
        df_base[f"mask_{sym}_t1"] = mask
```
**Confirmation:**
The `.shift(-1)` lines produce strictly `target_<sym>_t1` and `mask_<sym>_t1`. They are **never** included in feature lists or input matrices.

### c) Truncation Test (`verification/test_onset_causality.py`)
Tested across **all 3,698 participant-day rows** by completely truncating the input data to $t' \le t$ and recomputing all onset features from scratch:

| Feature Name | Rows Tested | Mismatches | Max Absolute Diff | Status |
|---|---|---|---|---|
| `cycle_id` | 3,698 | 0 | 0.000000 | **PASS** |
| `days_since_onset` | 3,698 | 0 | 0.000000 | **PASS** |
| `cycle_length_prior` | 3,698 | 0 | 0.000000 | **PASS** |
| `has_prior_cycle` | 3,698 | 0 | 0.000000 | **PASS** |
| `cycle_sin_pop` | 3,698 | 0 | 0.000000 | **PASS** |
| `cycle_cos_pop` | 3,698 | 0 | 0.000000 | **PASS** |

**Overall Causality Test:** **PASS (0 mismatches across 3,698 rows).**

---

## 4. `cycle_*_pop` Provenance

- **Is $\tau_{\text{pop}}$ a hard-coded constant or computed from data?**
  It is a **hard-coded constant** of **$29.0$ days** (`src/data/build_master.py` line 161: `tau_pop = 29.0`).
- **Provenance:**
  In the clinical literature and gynecology guidelines (e.g. FIGO menstrual standards), normal adult menstrual cycle length is 28–30 days. In Phase 1 EDA on the 2022 cohort, the empirical cohort median cycle length was observed to be 29.0 days (mean 30.5).
- **Leakage Status:**
  Because $\tau_{\text{pop}} = 29.0$ is an external scalar constant used purely for sinusoidal scaling ($2\pi \cdot \text{day} / 29.0$), it does not leak individual test-participant cycle lengths or test-period outcomes.

---

## 5. Preprocessing Scope & Fitting Invariants

### Exact Lines Fitting Scalers & Imputers

1. **Classical Baselines (`src/models/baselines.py` L87-L95):**
   ```python
   # 1. Fit Imputer and Scaler on training fold ONLY
   imputer = SimpleImputer(strategy="median")
   X_tr_imp = imputer.fit_transform(X_tr)
   X_te_imp = imputer.transform(X_te)
   
   scaler = StandardScaler()
   X_tr_scaled = scaler.fit_transform(X_tr_imp)
   X_te_scaled = scaler.transform(X_te_imp)
   ```
   *Passed Array:* `X_tr` ($X_{\text{flat}}[\text{train\_mask} \ \& \ M]$) containing **only outer-train windows**.
2. **Neural Models (`src/models/train_neural.py` L227-L233):**
   ```python
   # Preprocessing: Fit Imputer & Scaler on INNER TRAIN ONLY
   X_inner_tr = X[inner_train_idx].reshape(-1, D)
   imputer = SimpleImputer(strategy="median")
   imputer.fit(X_inner_tr)

   X_inner_tr_imp = imputer.transform(X_inner_tr)
   scaler = StandardScaler()
   scaler.fit(X_inner_tr_imp)
   ```
   *Passed Array:* `X_inner_tr` containing **only inner-train windows** (a strict subset of outer-train, disjoint from inner-validation and outer-test). Outer test participants are **never seen** during fitting.
3. **Class Weights / Positive Weights (`src/models/train_neural.py` L250-L260):**
   ```python
   pos_weights_list = []
   for s in range(6):
       m_s = M_tr[:, s] == 1
       n_pos = np.sum(Y_tr[m_s, s] == 1)
       n_neg = np.sum(Y_tr[m_s, s] == 0)
       pw = n_neg / max(n_pos, 1)
       pos_weights_list.append(pw)
   pos_weights_tensor = torch.tensor(pos_weights_list, dtype=torch.float32).to(device)
   ```
   *Passed Array:* `Y_tr` and `M_tr` computed **exclusively on inner-train**.

### Are any normalizations computed in `make_windows.py` or `build_master.py` before split?
- In `build_master.py` and `make_windows.py`, **NO standard scaling, min-max scaling, or mean-centering is performed**.
- Features stored in `windows_dataset.npz` are raw physical units (bpm, °C, ordinal integers $0\dots5$, minutes). Scaling is strictly deferred to fold-specific training splits.

---

## 6. Early Stopping, Inner Split, Hyperparameters, and CatBoost

### a) Configuration Values (`results/reproducibility_metadata.json`)
```json
{
  "cv_scheme": "5-fold GroupKFold by participant ID",
  "inner_cv_scheme": "Nested 5-fold GroupKFold on outer-train for early stopping",
  "n_samples": 3404,
  "seq_len": 7,
  "feat_dim": 43,
  "random_seed": 42,
  "models": {
    "MLP": {
      "in_features": 301,
      "hidden": [64, 32],
      "dropout": 0.3,
      "lr": 0.001,
      "weight_decay": 0.001,
      "max_epochs": 60,
      "patience": 12
    },
    "GRU": {
      "in_features": 43,
      "hidden_size": 32,
      "layers": 1,
      "dropout": 0.3,
      "lr": 0.001,
      "weight_decay": 0.001,
      "max_epochs": 60,
      "patience": 12
    }
  }
}
```

### b) Inner Train/Validation Split Code (`src/models/train_neural.py` L211-L223)
```python
        # Create inner participant-level split for early stopping
        outer_train_pids = p_ids[outer_train_idx]
        gkf_inner = GroupKFold(n_splits=5)
        # 1 inner fold for validation (~20% of training participants)
        inner_tr_sub, inner_val_sub = next(gkf_inner.split(outer_train_idx, groups=outer_train_pids))
        
        inner_train_idx = outer_train_idx[inner_tr_sub]
        inner_val_idx = outer_train_idx[inner_val_sub]

        # Verify inner participant disjointness
        inner_tr_p = set(p_ids[inner_train_idx])
        inner_val_p = set(p_ids[inner_val_idx])
        assert len(inner_tr_p.intersection(inner_val_p)) == 0, "Inner leak detected!"
```
**Confirmation:** Grouped by participant ID (`outer_train_pids`), drawn strictly from outer-train participants. Overlap is asserted to be 0.

### c) Early Stopping Metric & Test Set Isolation
- **Metric monitored:** `avg_val_loss` (multi-task masked BCE loss computed on `inner_val_idx`).
- **Patience:** 12 epochs.
- **Test set isolation:** Outer test fold is completely untouched during epoch iterations. Model state with lowest inner validation loss is loaded (`model.load_state_dict(best_model_state)`), and then evaluated once on outer test.

### d) CatBoost Configuration (`src/models/baselines.py` L135-L144)
```python
            cb = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos,
                random_seed=SEED,
                verbose=0
            )
            cb.fit(X_tr_imp, y_tr)
```
- `eval_set` and `early_stopping_rounds` were **NOT** used in CatBoost (fixed 250 trees, depth 4). There is zero test leak in CatBoost.

### e) Hyperparameter Search
- **No automated grid search, random search, or post-hoc test-set tuning loops exist.** Fixed default regularization parameters were used across all folds.

---

## 7. Persistence Baseline & Decision Thresholds

### a) Definitions
- **`q_days` (`make_windows.py:101`):** The final study day $t$ of the consecutive 7-day input window $[t-6 \dots t]$.
- **`t_days` (`make_windows.py:102`):** The target evaluation study day, strictly verified to be $t+1$.
- **`pers_eligible` (`make_windows.py:142`):** Binary flag indicating whether the participant has $\ge 2$ observed menstrual cycles (1 for 39 participants, 0 for 3 participants).
- **`fc_splits` (`make_windows.py:144-149`):** Forward-chaining partition label (`train` for earlier cycles $1\dots K_i-1$, `test` for the participant's final cycle $K_i$).

### b) Corrected Persistence Prediction Code (`src/models/train_neural.py` L168-L177)
```python
            y_pred = np.zeros(len(y_te), dtype=int)
            for i in range(len(y_te)):
                row_vals = win_sym_vals[i]
                observed = row_vals[~np.isnan(row_vals)]
                if len(observed) > 0:
                    y_pred[i] = int(observed[-1] >= 3)
                else:
                    y_pred[i] = 0 # fallback if no observation in 7 days
                    
            y_prob = y_pred.astype(float)
```
- Operates within participant on historical window days $[t-6 \dots t]$.
- Carries forward the last non-NaN survey response (`observed[-1] >= 3`).
- Outputs a binary label $\{0, 1\}$ cast to float $\{0.0, 1.0\}$.

### c) Decision Thresholds & Class Weighting
- **Threshold for F1 / Precision / Recall:** Fixed at **$0.5$** for all models across all folds.
- **Class Weights:**
  - LogReg used `class_weight='balanced'`.
  - CatBoost used `scale_pos_weight = N_neg / N_pos`.
  - MLP & GRU used `pos_weight = N_neg / N_pos` in BCEWithLogitsLoss.
- **Brier Score Calculation:** Computed via `brier_score_loss(y_true, y_prob)`. Since models were trained with loss weighting, their uncalibrated output probabilities reflect that positive shift.

---

## 8. Paired Fold-Level Summaries (`results/all_models_groupkfold_folds.csv`)

Computed directly across the 5 outer folds (no folds dropped or reweighted):

| Target Symptom | CatBoost vs GRU (PR-AUC Wins) | CatBoost vs GRU PR-AUC Paired Diff ($\Delta$) | CatBoost vs Persistence (PR-AUC Wins) | CatBoost vs Persistence PR-AUC Paired Diff ($\Delta$) | CatBoost vs Persistence (Brier Wins) | CatBoost vs Persistence Brier Paired Diff ($\Delta$) |
|---|---|---|---|---|---|---|
| **cramps** | 3 / 5 | $+0.0061 \pm 0.0502$ | 5 / 5 | $+0.1447 \pm 0.0700$ | 4 / 5 | $-0.0116 \pm 0.0151$ |
| **moodswing** | 3 / 5 | $+0.0104 \pm 0.0321$ | 5 / 5 | $+0.1292 \pm 0.0401$ | 4 / 5 | $-0.0503 \pm 0.0362$ |
| **fatigue** | 4 / 5 | $+0.0267 \pm 0.0287$ | 5 / 5 | $+0.1176 \pm 0.0277$ | 5 / 5 | $-0.0819 \pm 0.0223$ |
| **bloating** | 4 / 5 | $+0.0091 \pm 0.0251$ | 5 / 5 | $+0.1233 \pm 0.0562$ | 5 / 5 | $-0.0401 \pm 0.0221$ |
| **sorebreasts**| 4 / 5 | $+0.0596 \pm 0.1050$ | 4 / 5 | $+0.1104 \pm 0.1050$ | 4 / 5 | $+0.0039 \pm 0.0350$ |
| **headaches** | 3 / 5 | $+0.0240 \pm 0.0444$ | 5 / 5 | $+0.1823 \pm 0.0458$ | 5 / 5 | $-0.0456 \pm 0.0202$ |

*(Note on Brier: A negative difference $\Delta(\text{CB} - \text{Pers}) < 0$ denotes lower / superior mean squared probability error for CatBoost).*

---

## 9. Biological Cycle Counts & Distributions (2022 Interval)

Computed using the finalized cycle detection on `data/processed/master_daily_2022.csv`:

- **Total detected bleeding onsets:** **121**
- **Total onset-to-onset intervals:** **79**
- **Participants with $\ge 1$ onset:** **42 / 42** (100.0%)
- **Participants with $\ge 2$ onsets:** **39 / 42** (92.9%)
- **Participants with $\ge 1$ complete interval:** **39 / 42** (92.9%)
- **Cycle interval length median:** **29.0 days**
- **Cycle interval length mean:** **30.54 days**
- **Cycle interval IQR:** **[27.0, 33.0] days** (width: 6.0 days)
- **Short outlier intervals ($< 20$ days):** **2** (genuine biological irregularity)
- **Long outlier intervals ($> 45$ days):** **3** (genuine biological irregularity)

---

## Final Verification Summary Table

| Item | Status | One-Line Reason | Evidence Location |
|---|---|---|---|
| **Task 1: Feature Inventory** | **PASS** | 43 daily features confirmed; `phase` absent; resting HR 0 converted to NaN before merge; temperature is absolute nightly °C + baseline SD ratio. | `build_master.py:70-85, 182-186, 225-226` |
| **Task 2a: Target Reproduction** | **PASS** | All 30 combinations of fold $\times$ target reproduced $n_{\text{test}}$ and $n_{\text{pos}}$ to exact integer counts. | `verification/run_pass2_checks.py` |
| **Task 2b: Target Mask Values** | **PASS** | $Y$ is strictly `NaN` wherever $M == 0$ (368 positions per target); zero unmasked false zeros. | `verification/run_pass2_checks.py` |
| **Task 2c: Mask Implementation** | **PASS** | Mask is applied element-wise in loss, sample selection, and metric evaluation. | `train_neural.py:103-111, 402-410`; `baselines.py:80-85` |
| **Task 2d: Loss Invariance Test** | **PASS** | Masked BCE loss and autograd gradients are invariant ($\Delta = 0.0$) to values at $M == 0$. | `verification/run_pass2_checks.py` |
| **Task 2e: NaN/Fill Audits** | **PASS** | Code inspection confirms `nan_to_num` operates solely on unlogged targets; `fillna(0)` in cyclic math restores NaNs. | `train_neural.py:105`; `build_master.py:162, 170` |
| **Task 3: Onset Causality** | **PASS** | Truncation test across all 3,698 rows confirmed 0 mismatches between full series and past-only truncated series. | `verification/test_onset_causality.py` |
| **Task 4: `cycle_*_pop` Provenance**| **PASS** | Population cycle length $\tau_{\text{pop}}=29.0$ is a scalar constant based on clinical literature/median. | `build_master.py:161` |
| **Task 5: Preprocessing Scope** | **PASS** | Imputers and scalers are fit strictly on training splits (inner-train for neural); outer test is purely transformed. | `train_neural.py:227-233`; `baselines.py:87-95` |
| **Task 6: Early Stopping & Isolation** | **PASS** | Early stopping monitors inner-validation loss only; outer test is strictly evaluated once on frozen checkpoint. | `train_neural.py:211-223, 362-370` |
| **Task 7: Persistence & Thresholds** | **PASS** | Corrected persistence implements forward-fill within participant history; fixed 0.5 decision threshold. | `train_neural.py:168-177` |
| **Task 8: Paired Fold Comparisons** | **PASS** | Full paired fold differences and win counts computed without dropping any folds. | `verification/run_tasks_8_9.py` |
| **Task 9: Cycle Distributions** | **PASS** | 121 onsets, 79 intervals across 39 participants; median 29.0 days, mean 30.54 days, IQR 27–33. | `verification/run_tasks_8_9.py` |

---

### Files Created During Verification
1. `verification/run_pass2_checks.py` (Unit tests, inventory, mask checks)
2. `verification/test_onset_causality.py` (3,698-row truncation causality test)
3. `verification/run_tasks_8_9.py` (Fold-level paired analysis & cycle distributions)
4. `verification/pass2_report.md` (Saved markdown verification report)
