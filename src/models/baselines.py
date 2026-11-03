"""
src/models/baselines.py
Evaluates population-level baselines across 5-fold GroupKFold:
1. Persistence Baseline (Naive last observed value at day t)
2. Population Prior Baseline (Training set empirical mean)
3. L2-Regularized Logistic Regression (Flattened 7-day window)
4. CatBoost Classifier (Flattened 7-day window)

Metrics: PR-AUC (Average Precision), ROC-AUC, F1, Precision, Recall, Brier Score.
Preprocessing (Imputer, Scaler) fitted strictly on training folds.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from catboost import CatBoostClassifier

DATA_PATH = Path("data/processed/windows_dataset.npz")
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "baselines_groupkfold_results.csv"

SEED = 42

def load_data():
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"
    data = np.load(DATA_PATH)
    X = data["X"]          # (N, 7, 43)
    Y = data["Y"]          # (N, 6)
    M = data["M"]          # (N, 6)
    folds = data["folds"]  # (N,)
    feature_names = data["feature_names"]
    target_names = data["target_names"]
    return X, Y, M, folds, feature_names, target_names


def run_baselines():
    print("=" * 70)
    print("STEP S4: EVALUATING POPULATION BASELINES (5-FOLD GROUPKFOLD)")
    print("=" * 70)
    
    X, Y, M, folds, feature_names, target_names = load_data()
    N, W, D = X.shape
    X_flat = X.reshape(N, W * D)  # Flatten 7-day history to 301 features
    
    # Identify indices of the 6 target symptoms at day t (last step in window, index 6)
    # The symptom ordinals are the first 12 features in DAILY_FEATURES
    sym_to_idx = {name: i for i, name in enumerate(feature_names)}
    
    results = []

    for s_idx, sym_name in enumerate(target_names):
        print(f"\n>>> TARGET SYMPTOM: {sym_name.upper()} <<<")
        
        # Get index of day-t ordinal for this symptom: step 6 * D + feature_idx
        day_t_feature_idx = (W - 1) * D + sym_to_idx[f"{sym_name}_ord"]
        
        fold_metrics = {
            "Persistence": [],
            "Pop_Prior": [],
            "LogReg": [],
            "CatBoost": [],
        }

        for f in range(5):
            train_mask = (folds != f)
            test_mask = (folds == f)
            
            # Supervised masks for this specific symptom
            train_sup = train_mask & (M[:, s_idx] == 1)
            test_sup = test_mask & (M[:, s_idx] == 1)
            
            X_tr, y_tr = X_flat[train_sup], Y[train_sup, s_idx]
            X_te, y_te = X_flat[test_sup], Y[test_sup, s_idx]
            
            # 1. Fit Imputer and Scaler on training fold ONLY
            imputer = SimpleImputer(strategy="median")
            X_tr_imp = imputer.fit_transform(X_tr)
            X_te_imp = imputer.transform(X_te)
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_imp)
            X_te_scaled = scaler.transform(X_te_imp)
            
            pos_prev = np.mean(y_te)

            # ---------------------------------------------------------
            # Baseline 1: Persistence (State at day t >= 3)
            # ---------------------------------------------------------
            # Raw day t value (unimputed, if missing -> 0)
            day_t_vals = X_flat[test_sup, day_t_feature_idx]
            y_pred_persist = np.where(np.isnan(day_t_vals), 0, (day_t_vals >= 3)).astype(int)
            y_prob_persist = y_pred_persist.astype(float)
            
            # ---------------------------------------------------------
            # Baseline 2: Population Prior (Training set mean)
            # ---------------------------------------------------------
            train_mean = np.mean(y_tr)
            y_prob_prior = np.full_like(y_te, fill_value=train_mean, dtype=float)
            y_pred_prior = (y_prob_prior >= 0.5).astype(int)
            
            # ---------------------------------------------------------
            # Baseline 3: Logistic Regression (L2 penalty)
            # ---------------------------------------------------------
            lr = LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="liblinear",
                random_state=SEED,
                class_weight="balanced"
            )
            lr.fit(X_tr_scaled, y_tr)
            y_prob_lr = lr.predict_proba(X_te_scaled)[:, 1]
            y_pred_lr = (y_prob_lr >= 0.5).astype(int)
            
            # ---------------------------------------------------------
            # Baseline 4: CatBoost Classifier
            # ---------------------------------------------------------
            # Calculate scale_pos_weight
            neg_count = np.sum(y_tr == 0)
            pos_count = np.sum(y_tr == 1)
            scale_pos = neg_count / max(pos_count, 1)
            
            cb = CatBoostClassifier(
                iterations=250,
                depth=4,
                learning_rate=0.05,
                scale_pos_weight=scale_pos,
                random_seed=SEED,
                verbose=0
            )
            # CatBoost handles NaNs naturally, but we use X_tr_imp for consistency
            cb.fit(X_tr_imp, y_tr)
            y_prob_cb = cb.predict_proba(X_te_imp)[:, 1]
            y_pred_cb = (y_prob_cb >= 0.5).astype(int)

            # Metric evaluation function
            def eval_preds(y_true, y_pred, y_prob):
                pr_auc = average_precision_score(y_true, y_prob)
                try:
                    roc_auc = roc_auc_score(y_true, y_prob)
                except ValueError:
                    roc_auc = np.nan
                f1 = f1_score(y_true, y_pred, zero_division=0)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                brier = brier_score_loss(y_true, y_prob)
                return {
                    "PR-AUC": pr_auc,
                    "ROC-AUC": roc_auc,
                    "F1": f1,
                    "Precision": prec,
                    "Recall": rec,
                    "Brier": brier,
                }

            res_persist = eval_preds(y_te, y_pred_persist, y_prob_persist)
            res_prior = eval_preds(y_te, y_pred_prior, y_prob_prior)
            res_lr = eval_preds(y_te, y_pred_lr, y_prob_lr)
            res_cb = eval_preds(y_te, y_pred_cb, y_prob_cb)

            for m_name, m_dict in [
                ("Persistence", res_persist),
                ("Pop_Prior", res_prior),
                ("LogReg", res_lr),
                ("CatBoost", res_cb),
            ]:
                record = {
                    "symptom": sym_name,
                    "fold": f,
                    "model": m_name,
                    "prevalence": pos_prev,
                    "n_test": len(y_te),
                    **m_dict,
                }
                results.append(record)
                fold_metrics[m_name].append(m_dict)

        # Print summary across folds for this symptom
        print(f"{'Model':14s} | {'PR-AUC (mean±std)':18s} | {'ROC-AUC':14s} | {'F1':14s} | {'Precision':12s} | {'Recall':12s}")
        print("-" * 90)
        for m_name in ["Persistence", "Pop_Prior", "LogReg", "CatBoost"]:
            m_list = fold_metrics[m_name]
            pr_vals = [d["PR-AUC"] for d in m_list]
            roc_vals = [d["ROC-AUC"] for d in m_list]
            f1_vals = [d["F1"] for d in m_list]
            p_vals = [d["Precision"] for d in m_list]
            r_vals = [d["Recall"] for d in m_list]
            print(
                f"{m_name:14s} | {np.mean(pr_vals):.3f} ± {np.std(pr_vals):.3f}    | "
                f"{np.mean(roc_vals):.3f} ± {np.std(roc_vals):.3f} | "
                f"{np.mean(f1_vals):.3f} ± {np.std(f1_vals):.3f} | "
                f"{np.mean(p_vals):.3f} ± {np.std(p_vals):.3f}  | "
                f"{np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}"
            )

    df_results = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"\nDetailed fold-by-fold results saved to: {RESULTS_FILE}")
    return df_results


if __name__ == "__main__":
    run_baselines()
