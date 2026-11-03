"""
verification/h1_v2.py
Tests Hypothesis H1 v2: Separating "Own Data" from "More Data"
Arms compared for participant p in fold k on target j:
  A (Population): Trained on folds != k & M[:, j] == 1 (1 model per fold).
  B (Personalized): A's training rows + ONLY p's own fc 'train' windows (passing gap rule & M[:, j] == 1).
  C (Extra-data control): A's training rows + fc 'train' windows of all OTHER test-fold participants (excluding p).

Strict asserts:
  - p's own windows never in C
  - no fc 'test' window of anyone is in any training set
  - no eval row in any training set
  - gap rule: t-6, t, and t+1 strictly < first lookback day of its owner's test cycle.

Cluster bootstrap over participants: 2000 replicates, seed 42.
Outputs:
  - results/h1v2_summary.csv
  - results/h1v2_predictions_LOCAL.npz
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

def run_h1_v2():
    t0 = time.time()
    print("=" * 75)
    print("TESTING HYPOTHESIS H1 v2: OWN DATA vs MORE DATA (ARMS A, B, C)")
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
    
    elig_pids = np.unique(p_ids[pers_eligible == 1])
    print(f"Total eligible participants: {len(elig_pids)} / 42")
    
    # -------------------------------------------------------------------------
    # 1. Counts and Distributions
    # -------------------------------------------------------------------------
    print("\n--- 1. COUNTS & DISTRIBUTIONS ---")
    # All 671 fc-test windows vs supervised (M==1)
    test_total_per_p = []
    test_sup_per_p = []
    for pid in elig_pids:
        p_all_test = (p_ids == pid) & (fc_splits == "test")
        p_sup_test = p_all_test & (M[:, 0] == 1) # M is identical across targets
        test_total_per_p.append(int(np.sum(p_all_test)))
        test_sup_per_p.append(int(np.sum(p_sup_test)))
        
    print(f"Total fc-test windows per participant: min={min(test_total_per_p)}, median={int(np.median(test_total_per_p))}, max={max(test_total_per_p)}")
    print(f"Supervised eval windows per participant: min={min(test_sup_per_p)}, median={int(np.median(test_sup_per_p))}, max={max(test_sup_per_p)}")
    print(f"Participants with < 5 total fc-test windows: {sum(1 for c in test_total_per_p if c < 5)}")
    print(f"Participants with < 5 supervised eval windows: {sum(1 for c in test_sup_per_p if c < 5)}")
    
    for j, sym in enumerate(target_names):
        test_mask = (fc_splits == "test") & (pers_eligible == 1)
        m0_count = int(np.sum(test_mask & (M[:, j] == 0)))
        print(f"  {sym:12s}: M==0 on test windows = {m0_count} / {test_mask.sum()} ({m0_count/test_mask.sum()*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Precompute gap rule & valid own-train windows per participant
    # -------------------------------------------------------------------------
    first_test_lookback = {}
    for pid in elig_pids:
        p_test_mask = (p_ids == pid) & (fc_splits == "test")
        test_q = q_days[p_test_mask]
        first_test_lookback[pid] = np.min(test_q - 6)

    valid_own_train = {}
    for pid in elig_pids:
        p_tr_mask = (p_ids == pid) & (fc_splits == "train")
        # Gap rule condition: target day t+1 < first lookback day of owner's test cycle
        valid_indices = []
        for idx in np.where(p_tr_mask)[0]:
            flb = first_test_lookback[pid]
            # strict assertion on all potential own-train windows
            if t_days[idx] < flb:
                assert q_days[idx] < flb, f"Lookback end violated for participant <ID>!"
                assert q_days[idx] - 6 < flb, f"Lookback start violated for participant <ID>!"
                valid_indices.append(idx)
        valid_own_train[pid] = np.array(valid_indices, dtype=int)

    # -------------------------------------------------------------------------
    # 2. Added windows per participant for B and for C
    # -------------------------------------------------------------------------
    print("\n--- 2. ADDED WINDOWS PER PARTICIPANT FOR B AND C ---")
    b_added_counts = []
    c_added_counts = []
    for pid in elig_pids:
        k = folds[p_ids == pid][0]
        other_pids = [other for other in elig_pids if other != pid and folds[p_ids == other][0] == k]
        
        b_indices = valid_own_train[pid]
        b_added_counts.append(len(b_indices))
        
        c_indices = []
        for other in other_pids:
            c_indices.extend(valid_own_train[other])
        c_added_counts.append(len(c_indices))

    print(f"Added windows for Arm B (own data): min={min(b_added_counts)}, median={int(np.median(b_added_counts))}, max={max(b_added_counts)}")
    print(f"Added windows for Arm C (extra other data): min={min(c_added_counts)}, median={int(np.median(c_added_counts))}, max={max(c_added_counts)}")

    # -------------------------------------------------------------------------
    # Training & Evaluation of Arms A, B, C
    # -------------------------------------------------------------------------
    print("\n--- Training Arms A, B, C across all 6 targets ---")
    
    # We will record predictions for every supervised eval window (total 497 windows)
    # Stored indexed by window index in dataset
    eval_window_indices = np.where((fc_splits == "test") & (pers_eligible == 1) & (M[:, 0] == 1))[0]
    n_eval_total = len(eval_window_indices)
    print(f"Total supervised eval windows pooled across all participants: {n_eval_total}")
    
    # Prediction dictionaries
    preds_A = np.zeros((n_eval_total, 6), dtype=np.float32)
    preds_B = np.zeros((n_eval_total, 6), dtype=np.float32)
    preds_C = np.zeros((n_eval_total, 6), dtype=np.float32)
    targets_eval = Y[eval_window_indices] # (497, 6)
    pids_eval = p_ids[eval_window_indices]
    folds_eval = folds[eval_window_indices]

    # Map each eval index to its position in [0 .. n_eval_total - 1]
    eval_pos_map = {idx: pos for pos, idx in enumerate(eval_window_indices)}

    for s_idx, sym in enumerate(target_names):
        print(f"\nProcessing Target {s_idx+1}/6: {sym.upper()}...")
        for f in range(5):
            # 1. Arm A (Population model for fold f)
            pop_tr_mask = (folds != f) & (M[:, s_idx] == 1)
            pop_tr_idx = np.where(pop_tr_mask)[0]
            
            # Assertions for Arm A:
            fold_f_test_windows = np.where((folds == f) & (fc_splits == "test"))[0]
            assert len(set(p_ids[folds == f]).intersection(set(p_ids[pop_tr_idx]))) == 0, "Fold-f participant in pop train!"
            fold_f_eval_indices = np.where((folds == f) & (fc_splits == "test") & (pers_eligible == 1) & (M[:, s_idx] == 1))[0]
            assert len(set(fold_f_eval_indices).intersection(set(pop_tr_idx))) == 0, "Eval row in pop train!"

            imp_A = SimpleImputer(strategy="median")
            X_A_tr = imp_A.fit_transform(X_flat[pop_tr_idx])
            y_A_tr = Y[pop_tr_idx, s_idx]
            pw_A = float(np.sum(y_A_tr == 0) / max(np.sum(y_A_tr == 1), 1))

            cb_A = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=pw_A,
                random_seed=SEED,
                thread_count=4,
                verbose=0
            )
            cb_A.fit(X_A_tr, y_A_tr)

            # Get eligible participants in this fold
            fold_elig_pids = [pid for pid in elig_pids if folds[p_ids == pid][0] == f]

            for pid in fold_elig_pids:
                p_eval_mask = (p_ids == pid) & (fc_splits == "test") & (M[:, s_idx] == 1)
                p_eval_idx = np.where(p_eval_mask)[0]
                
                if len(p_eval_idx) == 0:
                    continue
                    
                # Predictions for Arm A
                X_p_eval_A = imp_A.transform(X_flat[p_eval_idx])
                prob_A = cb_A.predict_proba(X_p_eval_A)[:, 1]
                for i_pos, glob_idx in enumerate(p_eval_idx):
                    preds_A[eval_pos_map[glob_idx], s_idx] = prob_A[i_pos]

                # 2. Arm B (Personalized: A + p's own valid train)
                own_tr_idx = [idx for idx in valid_own_train[pid] if M[idx, s_idx] == 1]
                own_tr_idx = np.array(own_tr_idx, dtype=int)
                
                b_tr_idx = np.concatenate([pop_tr_idx, own_tr_idx])
                
                # Strict Assertions for Arm B:
                assert len(set(p_eval_idx).intersection(set(b_tr_idx))) == 0, "Eval row in Arm B train!"
                assert np.all(fc_splits[own_tr_idx] == "train"), "Non-train window added to Arm B!"
                assert len(set(fold_f_test_windows).intersection(set(b_tr_idx))) == 0, "Fold-f test window in Arm B train!"
                for idx in own_tr_idx:
                    flb = first_test_lookback[pid]
                    assert t_days[idx] < flb and q_days[idx] < flb and (q_days[idx] - 6) < flb, "Gap rule violated in Arm B!"

                imp_B = SimpleImputer(strategy="median")
                X_B_tr = imp_B.fit_transform(X_flat[b_tr_idx])
                y_B_tr = Y[b_tr_idx, s_idx]
                pw_B = float(np.sum(y_B_tr == 0) / max(np.sum(y_B_tr == 1), 1))

                cb_B = CatBoostClassifier(
                    iterations=250,
                    depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=pw_B,
                    random_seed=SEED,
                    thread_count=4,
                    verbose=0
                )
                cb_B.fit(X_B_tr, y_B_tr)
                
                X_p_eval_B = imp_B.transform(X_flat[p_eval_idx])
                prob_B = cb_B.predict_proba(X_p_eval_B)[:, 1]
                for i_pos, glob_idx in enumerate(p_eval_idx):
                    preds_B[eval_pos_map[glob_idx], s_idx] = prob_B[i_pos]

                # 3. Arm C (Extra-data control: A + valid train of OTHER test fold participants)
                other_pids = [other for other in fold_elig_pids if other != pid]
                c_other_idx = []
                for other in other_pids:
                    c_other_idx.extend([idx for idx in valid_own_train[other] if M[idx, s_idx] == 1])
                c_other_idx = np.array(c_other_idx, dtype=int)
                
                c_tr_idx = np.concatenate([pop_tr_idx, c_other_idx])
                
                # Strict Assertions for Arm C:
                assert len(set(p_eval_idx).intersection(set(c_tr_idx))) == 0, "Eval row in Arm C train!"
                assert not any(p_ids[idx] == pid for idx in c_tr_idx), "p's own windows present in Arm C!"
                assert np.all(fc_splits[c_other_idx] == "train"), "Non-train window added to Arm C!"
                assert len(set(fold_f_test_windows).intersection(set(c_tr_idx))) == 0, "Fold-f test window in Arm C train!"
                for idx in c_other_idx:
                    o_pid = p_ids[idx]
                    flb = first_test_lookback[o_pid]
                    assert t_days[idx] < flb and q_days[idx] < flb and (q_days[idx] - 6) < flb, "Gap rule violated in Arm C!"

                imp_C = SimpleImputer(strategy="median")
                X_C_tr = imp_C.fit_transform(X_flat[c_tr_idx])
                y_C_tr = Y[c_tr_idx, s_idx]
                pw_C = float(np.sum(y_C_tr == 0) / max(np.sum(y_C_tr == 1), 1))

                cb_C = CatBoostClassifier(
                    iterations=250,
                    depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=pw_C,
                    random_seed=SEED,
                    thread_count=4,
                    verbose=0
                )
                cb_C.fit(X_C_tr, y_C_tr)

                X_p_eval_C = imp_C.transform(X_flat[p_eval_idx])
                prob_C = cb_C.predict_proba(X_p_eval_C)[:, 1]
                for i_pos, glob_idx in enumerate(p_eval_idx):
                    preds_C[eval_pos_map[glob_idx], s_idx] = prob_C[i_pos]

    # Save local predictions
    np.savez_compressed(
        RESULTS_DIR / "h1v2_predictions_LOCAL.npz",
        preds_A=preds_A,
        preds_B=preds_B,
        preds_C=preds_C,
        targets=targets_eval,
        pids=pids_eval,
        folds=folds_eval,
        eval_indices=eval_window_indices
    )
    print(f"\nSaved local window predictions to: {RESULTS_DIR / 'h1v2_predictions_LOCAL.npz'}")

    # -------------------------------------------------------------------------
    # 3. Pooled Metrics per Target for A, B, C
    # -------------------------------------------------------------------------
    print("\n--- 3. POOLED METRICS PER TARGET ---")
    pooled_records = []
    for s_idx, sym in enumerate(target_names):
        y_true = targets_eval[:, s_idx]
        p_A = preds_A[:, s_idx]
        p_B = preds_B[:, s_idx]
        p_C = preds_C[:, s_idx]
        
        n_eval = len(y_true)
        n_pos = int(np.sum(y_true == 1))
        
        def calc_metrics(prob):
            pr = float(average_precision_score(y_true, prob))
            roc = float(roc_auc_score(y_true, prob))
            brier = float(brier_score_loss(y_true, prob))
            return pr, roc, brier
            
        pr_A, roc_A, br_A = calc_metrics(p_A)
        pr_B, roc_B, br_B = calc_metrics(p_B)
        pr_C, roc_C, br_C = calc_metrics(p_C)
        
        pooled_records.append({
            "symptom": sym,
            "n_eval": n_eval,
            "n_pos": n_pos,
            "A_PR_AUC": pr_A,
            "B_PR_AUC": pr_B,
            "C_PR_AUC": pr_C,
            "A_ROC_AUC": roc_A,
            "B_ROC_AUC": roc_B,
            "C_ROC_AUC": roc_C,
            "A_Brier": br_A,
            "B_Brier": br_B,
            "C_Brier": br_C,
        })
        
    df_pooled = pd.DataFrame(pooled_records)
    print(df_pooled.to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. Cluster Bootstrap over Participants (2000 replicates, seed 42)
    # -------------------------------------------------------------------------
    print("\n--- 4. CLUSTER BOOTSTRAP OVER PARTICIPANTS (2000 REPLICATES) ---")
    rng = np.random.default_rng(SEED)
    B_reps = 2000
    
    # Unique participants present in the eval set (those with >=1 supervised eval window)
    unique_eval_pids = np.unique(pids_eval)
    n_p = len(unique_eval_pids)
    print(f"Participants in bootstrap cluster: {n_p}")
    
    # Pre-index windows by participant for fast resampling
    p_window_indices = {pid: np.where(pids_eval == pid)[0] for pid in unique_eval_pids}
    
    bootstrap_records = []
    
    for s_idx, sym in enumerate(target_names):
        y_all = targets_eval[:, s_idx]
        pA_all = preds_A[:, s_idx]
        pB_all = preds_B[:, s_idx]
        pC_all = preds_C[:, s_idx]
        
        diff_BA_pr = np.zeros(B_reps, dtype=float)
        diff_CA_pr = np.zeros(B_reps, dtype=float)
        diff_BC_pr = np.zeros(B_reps, dtype=float)
        
        diff_BA_br = np.zeros(B_reps, dtype=float)
        diff_CA_br = np.zeros(B_reps, dtype=float)
        diff_BC_br = np.zeros(B_reps, dtype=float)
        
        valid_reps = 0
        while valid_reps < B_reps:
            # Resample participants with replacement
            sampled_pids = rng.choice(unique_eval_pids, size=n_p, replace=True)
            # Concatenate all their windows
            sampled_idx = np.concatenate([p_window_indices[pid] for pid in sampled_pids])
            
            y_b = y_all[sampled_idx]
            # Need at least 1 positive and 1 negative to evaluate PR-AUC
            if np.sum(y_b == 1) == 0 or np.sum(y_b == 0) == 0:
                continue
                
            pA_b = pA_all[sampled_idx]
            pB_b = pB_all[sampled_idx]
            pC_b = pC_all[sampled_idx]
            
            prA = average_precision_score(y_b, pA_b)
            prB = average_precision_score(y_b, pB_b)
            prC = average_precision_score(y_b, pC_b)
            
            brA = brier_score_loss(y_b, pA_b)
            brB = brier_score_loss(y_b, pB_b)
            brC = brier_score_loss(y_b, pC_b)
            
            diff_BA_pr[valid_reps] = prB - prA
            diff_CA_pr[valid_reps] = prC - prA
            diff_BC_pr[valid_reps] = prB - prC
            
            # For Brier: improvement means lower error, but diff is B - A
            diff_BA_br[valid_reps] = brB - brA
            diff_CA_br[valid_reps] = brC - brA
            diff_BC_br[valid_reps] = brB - brC
            
            valid_reps += 1

        def summarize_diff(arr, is_brier=False):
            mean_val = float(np.mean(arr))
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            if is_brier:
                # improvement for Brier is diff < 0
                imp_rate = float(np.mean(arr < 0))
            else:
                # improvement for PR-AUC is diff > 0
                imp_rate = float(np.mean(arr > 0))
            return mean_val, ci_low, ci_high, imp_rate

        ba_pr_m, ba_pr_l, ba_pr_h, ba_pr_imp = summarize_diff(diff_BA_pr, is_brier=False)
        ca_pr_m, ca_pr_l, ca_pr_h, ca_pr_imp = summarize_diff(diff_CA_pr, is_brier=False)
        bc_pr_m, bc_pr_l, bc_pr_h, bc_pr_imp = summarize_diff(diff_BC_pr, is_brier=False)

        ba_br_m, ba_br_l, ba_br_h, ba_br_imp = summarize_diff(diff_BA_br, is_brier=True)
        ca_br_m, ca_br_l, ca_br_h, ca_br_imp = summarize_diff(diff_CA_br, is_brier=True)
        bc_br_m, bc_br_l, bc_br_h, bc_br_imp = summarize_diff(diff_BC_br, is_brier=True)

        bootstrap_records.append({
            "symptom": sym,
            # B - A (Own vs Pop)
            "BA_PR_mean": ba_pr_m, "BA_PR_CI": f"[{ba_pr_l:+.4f}, {ba_pr_h:+.4f}]", "BA_PR_imp_share": ba_pr_imp,
            "BA_Br_mean": ba_br_m, "BA_Br_CI": f"[{ba_br_l:+.4f}, {ba_br_h:+.4f}]", "BA_Br_imp_share": ba_br_imp,
            # C - A (More Data vs Pop)
            "CA_PR_mean": ca_pr_m, "CA_PR_CI": f"[{ca_pr_l:+.4f}, {ca_pr_h:+.4f}]", "CA_PR_imp_share": ca_pr_imp,
            "CA_Br_mean": ca_br_m, "CA_Br_CI": f"[{ca_br_l:+.4f}, {ca_br_h:+.4f}]", "CA_Br_imp_share": ca_br_imp,
            # B - C (Own Data vs More Data)
            "BC_PR_mean": bc_pr_m, "BC_PR_CI": f"[{bc_pr_l:+.4f}, {bc_pr_h:+.4f}]", "BC_PR_imp_share": bc_pr_imp,
            "BC_Br_mean": bc_br_m, "BC_Br_CI": f"[{bc_br_l:+.4f}, {bc_br_h:+.4f}]", "BC_Br_imp_share": bc_br_imp,
        })

    df_boot = pd.DataFrame(bootstrap_records)
    
    # Combine pooled + bootstrap into results/h1v2_summary.csv
    df_combined = pd.merge(df_pooled, df_boot, on="symptom")
    df_combined.to_csv(RESULTS_DIR / "h1v2_summary.csv", index=False)
    print(f"\nSaved combined summary to: {RESULTS_DIR / 'h1v2_summary.csv'}")

    runtime = time.time() - t0
    print(f"\nTotal H1 v2 Runtime: {runtime:.2f} seconds.")
    return df_combined, runtime

if __name__ == "__main__":
    run_h1_v2()
