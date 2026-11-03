"""
verification/h1_personalization.py
Tests Hypothesis H1: Personalized vs Population forecasting using CatBoost.
Evaluated strictly on final-cycle windows (fc_splits == 'test' & pers_eligible == 1).
Models compared:
  - Population model: Trained on all windows of folds != k.
  - Personalized (weight 1): Population training rows + own-train rows (fc_splits == 'train') passing the gap rule.
  - Personalized (weight 5): Same as above, with own-train rows assigned sample_weight = 5.0.

Gap rule: For any participant in fold k, drop any own-train window whose lookback [t-6..t] or target day t+1
falls on or after the first lookback day of that participant's test cycle.
Assert that for every added window: target_day < first_test_lookback_day.
"""

import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from catboost import CatBoostClassifier

DATA_PATH = Path("data/processed/windows_dataset.npz")
RESULTS_DIR = Path("results")
SEED = 42

def run_h1():
    t0 = time.time()
    print("=" * 75)
    print("TESTING HYPOTHESIS H1: PERSONALIZED VS POPULATION (CATBOOST)")
    print("=" * 75)
    
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
    data = np.load(DATA_PATH)
    X = data["X"]          # (N, 7, 43)
    Y = data["Y"]          # (N, 6)
    M = data["M"]          # (N, 6)
    folds = data["folds"]  # (N,)
    p_ids = data["p_ids"]  # (N,)
    fc_splits = data["fc_splits"]
    pers_eligible = data["pers_eligible"]
    q_days = data["q_days"]
    t_days = data["t_days"]
    target_names = list(data["target_names"])
    
    N, W, D = X.shape
    X_flat = X.reshape(N, W * D)
    
    # -------------------------------------------------------------
    # Step 0: Definitions and Counts
    # -------------------------------------------------------------
    print("\n--- STEP 0: SPLIT SUMMARY ---")
    unq, cnt = np.unique(fc_splits, return_counts=True)
    for u, c in zip(unq, cnt):
        print(f"  {u:12s}: {c:4d} windows")
    n_elig_p = len(np.unique(p_ids[pers_eligible == 1]))
    print(f"Participants with pers_eligible == 1: {n_elig_p} / 42")
    
    # Pre-calculate first test lookback day per eligible participant
    first_test_lookback = {}
    eligible_pids = np.unique(p_ids[pers_eligible == 1])
    for pid in eligible_pids:
        p_test_mask = (p_ids == pid) & (fc_splits == "test")
        test_q = q_days[p_test_mask]
        first_test_lookback[pid] = np.min(test_q - 6)

    # Calculate distribution of own-train windows added per participant
    own_train_dist = []
    for pid in eligible_pids:
        p_tr_mask = (p_ids == pid) & (fc_splits == "train")
        valid_p_tr = t_days[p_tr_mask] < first_test_lookback[pid]
        own_train_dist.append(int(np.sum(valid_p_tr)))

    print("\nDistribution of own-train windows added per participant (passing gap rule):")
    print(f"  Min: {min(own_train_dist)} | Median: {int(np.median(own_train_dist))} | Max: {max(own_train_dist)}")

    # -------------------------------------------------------------
    # Step 1: Run Fold x Target Evaluation
    # -------------------------------------------------------------
    fold_records = []
    
    for s_idx, sym in enumerate(target_names):
        print(f"\n>>> Running Target: {sym.upper()} <<<")
        for f in range(5):
            # 1. EVAL rows: fc_splits=='test' & folds==k & M[:,s_idx]==1 & pers_eligible==1
            eval_mask = (fc_splits == "test") & (folds == f) & (M[:, s_idx] == 1) & (pers_eligible == 1)
            eval_idx = np.where(eval_mask)[0]
            
            n_eval = len(eval_idx)
            y_eval = Y[eval_idx, s_idx]
            n_pos = int(np.sum(y_eval == 1))
            
            # 2. POPULATION training rows: folds != k & M[:, s_idx] == 1
            pop_tr_mask = (folds != f) & (M[:, s_idx] == 1)
            pop_tr_idx = np.where(pop_tr_mask)[0]
            
            # Sanity check (a): no eval row is in pop_tr
            assert len(set(eval_idx).intersection(set(pop_tr_idx))) == 0, "Eval row in pop train set!"
            # Sanity check (b): no fold-k participant is in pop_tr
            fold_k_pids = set(p_ids[folds == f])
            pop_tr_pids = set(p_ids[pop_tr_idx])
            assert len(fold_k_pids.intersection(pop_tr_pids)) == 0, "Fold-k participant in pop train set!"
            
            # 3. PERSONALIZED training rows: pop_tr_idx + own_tr_idx
            # Own-train windows: fold-k participants with fc_splits=='train' & M[:, s_idx]==1 & passing gap rule
            own_candidates_mask = (folds == f) & (fc_splits == "train") & (M[:, s_idx] == 1) & (pers_eligible == 1)
            own_candidates_idx = np.where(own_candidates_mask)[0]
            
            # Filter and strictly ASSERT gap rule for all added windows
            valid_own_idx = []
            for idx in own_candidates_idx:
                pid = p_ids[idx]
                target_day = t_days[idx]
                first_lookback = first_test_lookback[pid]
                if target_day < first_lookback:
                    valid_own_idx.append(idx)

            valid_own_idx = np.array(valid_own_idx, dtype=int)
            # Explicit assertion on every added window
            for idx in valid_own_idx:
                pid = p_ids[idx]
                first_lookback = first_test_lookback[pid]
                assert t_days[idx] < first_lookback, f"Target day violated for added window of participant <ID>!"
                assert q_days[idx] < first_lookback, f"Lookback day violated for added window of participant <ID>!"
                assert q_days[idx] - 6 < first_lookback, f"Lookback start violated for added window of participant <ID>!"

            pers_tr_idx = np.concatenate([pop_tr_idx, valid_own_idx])
            
            # Sanity check (a) for personalized train
            assert len(set(eval_idx).intersection(set(pers_tr_idx))) == 0, "Eval row in pers train set!"
            
            # Weights for personalized models
            # pop rows weight = 1.0, own rows weight = 1.0 (w1) or 5.0 (w5)
            weights_w1 = np.ones(len(pers_tr_idx), dtype=float)
            weights_w5 = np.ones(len(pers_tr_idx), dtype=float)
            weights_w5[len(pop_tr_idx):] = 5.0 # weight own rows by 5
            
            # 4. Preprocessing fitted on training rows ONLY for each model
            # For population model:
            imp_pop = SimpleImputer(strategy="median")
            X_pop_tr_imp = imp_pop.fit_transform(X_flat[pop_tr_idx])
            X_eval_imp_pop = imp_pop.transform(X_flat[eval_idx])
            
            # For personalized model:
            imp_pers = SimpleImputer(strategy="median")
            X_pers_tr_imp = imp_pers.fit_transform(X_flat[pers_tr_idx])
            X_eval_imp_pers = imp_pers.transform(X_flat[eval_idx])
            
            # Positive scale weights (from respective training set labels)
            y_pop_tr = Y[pop_tr_idx, s_idx]
            pos_weight_pop = float(np.sum(y_pop_tr == 0) / max(np.sum(y_pop_tr == 1), 1))
            
            y_pers_tr = Y[pers_tr_idx, s_idx]
            pos_weight_pers = float(np.sum(y_pers_tr == 0) / max(np.sum(y_pers_tr == 1), 1))
            
            # --- Model A: Population CatBoost ---
            cb_pop = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=pos_weight_pop,
                random_seed=SEED,
                verbose=0
            )
            cb_pop.fit(X_pop_tr_imp, y_pop_tr)
            probs_pop = cb_pop.predict_proba(X_eval_imp_pop)[:, 1]
            
            # --- Model B: Personalized CatBoost (w1) ---
            cb_pers_w1 = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=pos_weight_pers,
                random_seed=SEED,
                verbose=0
            )
            cb_pers_w1.fit(X_pers_tr_imp, y_pers_tr, sample_weight=weights_w1)
            probs_pers_w1 = cb_pers_w1.predict_proba(X_eval_imp_pers)[:, 1]
            
            # --- Model C: Personalized CatBoost (w5) ---
            cb_pers_w5 = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=pos_weight_pers,
                random_seed=SEED,
                verbose=0
            )
            cb_pers_w5.fit(X_pers_tr_imp, y_pers_tr, sample_weight=weights_w5)
            probs_pers_w5 = cb_pers_w5.predict_proba(X_eval_imp_pers)[:, 1]
            
            # Metric function
            def calc_metrics(y_true, y_prob):
                low_pos_flag = int(n_pos < 10)
                if n_pos < 10:
                    pr_auc = np.nan
                    roc_auc = np.nan
                else:
                    pr_auc = float(average_precision_score(y_true, y_prob))
                    try:
                        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
                    except ValueError:
                        roc_auc = np.nan
                brier = float(brier_score_loss(y_true, y_prob))
                return pr_auc, roc_auc, brier, low_pos_flag
            
            pr_pop, roc_pop, brier_pop, flag_pop = calc_metrics(y_eval, probs_pop)
            pr_w1, roc_w1, brier_w1, flag_w1 = calc_metrics(y_eval, probs_pers_w1)
            pr_w5, roc_w5, brier_w5, flag_w5 = calc_metrics(y_eval, probs_pers_w5)
            
            for m_name, pr, roc, brier, flg in [
                ("population", pr_pop, roc_pop, brier_pop, flag_pop),
                ("personalized_w1", pr_w1, roc_w1, brier_w1, flag_w1),
                ("personalized_w5", pr_w5, roc_w5, brier_w5, flag_w5),
            ]:
                fold_records.append({
                    "fold": f,
                    "symptom": sym,
                    "model": m_name,
                    "n_eval": n_eval,
                    "n_pos": n_pos,
                    "low_pos_flag": flg,
                    "PR-AUC": pr,
                    "ROC-AUC": roc,
                    "Brier": brier,
                })
                
    df_folds = pd.DataFrame(fold_records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_folds.to_csv(RESULTS_DIR / "h1_personalization_folds.csv", index=False)
    print(f"\nSaved fold results to: {RESULTS_DIR / 'h1_personalization_folds.csv'}")

    # -------------------------------------------------------------
    # Step 2: Summary Outputs & Paired Differences
    # -------------------------------------------------------------
    summary_records = []
    print("\n" + "=" * 85)
    print("STEP 2: SUMMARY TABLE & PAIRED DIFFERENCES (PERSONALIZED_W1 vs POPULATION)")
    print("=" * 85)
    
    for sym in target_names:
        sub = df_folds[df_folds["symptom"] == sym]
        
        pop = sub[sub["model"] == "population"].set_index("fold")
        w1 = sub[sub["model"] == "personalized_w1"].set_index("fold")
        w5 = sub[sub["model"] == "personalized_w5"].set_index("fold")
        
        # Paired differences (w1 - pop) where PR-AUC is non-null
        valid_pr = pop["PR-AUC"].notna() & w1["PR-AUC"].notna()
        diff_pr = (w1.loc[valid_pr, "PR-AUC"] - pop.loc[valid_pr, "PR-AUC"]).values
        wins_pr = int(np.sum(diff_pr > 0))
        mean_diff_pr = float(np.mean(diff_pr)) if len(diff_pr) > 0 else np.nan
        std_diff_pr = float(np.std(diff_pr, ddof=1)) if len(diff_pr) > 1 else np.nan
        
        diff_brier = (w1["Brier"] - pop["Brier"]).values # negative is better for w1
        wins_brier = int(np.sum(diff_brier < 0)) # w1 lower brier
        mean_diff_brier = float(np.mean(diff_brier))
        std_diff_brier = float(np.std(diff_brier, ddof=1))
        
        summary_records.append({
            "symptom": sym,
            "pop_pr_mean": pop["PR-AUC"].mean(),
            "pop_pr_std": pop["PR-AUC"].std(),
            "w1_pr_mean": w1["PR-AUC"].mean(),
            "w1_pr_std": w1["PR-AUC"].std(),
            "w5_pr_mean": w5["PR-AUC"].mean(),
            "w5_pr_std": w5["PR-AUC"].std(),
            "diff_pr_mean": mean_diff_pr,
            "diff_pr_std": std_diff_pr,
            "pr_wins": f"{wins_pr}/{len(diff_pr)}",
            "pop_brier_mean": pop["Brier"].mean(),
            "w1_brier_mean": w1["Brier"].mean(),
            "w5_brier_mean": w5["Brier"].mean(),
            "diff_brier_mean": mean_diff_brier,
            "diff_brier_std": std_diff_brier,
            "brier_wins": f"{wins_brier}/5",
        })
        
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(RESULTS_DIR / "h1_personalization_summary.csv", index=False)
    print(f"Saved summary to: {RESULTS_DIR / 'h1_personalization_summary.csv'}")

    # Total eval rows and positives per target
    print("\nTotal Evaluation Rows and Positives per Target Overall (Final Cycles Subset):")
    for sym in target_names:
        sub = df_folds[(df_folds["symptom"] == sym) & (df_folds["model"] == "population")]
        tot_eval = sub["n_eval"].sum()
        tot_pos = sub["n_pos"].sum()
        pct = (tot_pos / tot_eval) * 100 if tot_eval > 0 else 0
        print(f"  {sym:12s}: Total Eval = {tot_eval} | Positives = {tot_pos:3d} ({pct:5.1f}%)")

    runtime = time.time() - t0
    print(f"\nTotal Execution Runtime: {runtime:.2f} seconds.")
    return df_folds, df_summary, runtime

if __name__ == "__main__":
    run_h1()
