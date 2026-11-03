"""
src/models/train_neural.py
Implements and evaluates:
1. Corrected Persistence baseline (Last Observed Value in window)
2. MLP (flattened 7x43 features, 301 -> 64 -> 32 -> 6)
3. GRU (sequence 7x43, single layer hidden size 32, dropout 0.3, 32 -> 6)

Outer evaluation: 5-fold GroupKFold by participant ID.
Inner validation: Inner GroupKFold on outer training participants ONLY for model selection / early stopping.
Zero leakage: Scalers, imputers, class weights fitted on inner-train only.
Loss: Multi-task masked BCEWithLogitsLoss.
"""

from pathlib import Path
import json
import random
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

DATA_PATH = Path("data/processed/windows_dataset.npz")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WindowDataset(Dataset):
    def __init__(self, X_seq, X_flat, Y, M):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_flat = torch.tensor(X_flat, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_flat[idx], self.Y[idx], self.M[idx]

class MultiTaskMLP(nn.Module):
    def __init__(self, in_features=301, num_targets=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_targets),
        )

    def forward(self, x_flat):
        return self.net(x_flat)

class MultiTaskGRU(nn.Module):
    def __init__(self, in_features=43, hidden_size=32, num_targets=6, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_targets)

    def forward(self, x_seq):
        # x_seq: (batch, seq_len=7, feat_dim=43)
        out, h_n = self.gru(x_seq)
        # h_n: (1, batch, hidden_size)
        last_hidden = h_n.squeeze(0)
        dropped = self.drop(last_hidden)
        logits = self.head(dropped)
        return logits

def compute_masked_loss(logits, targets, masks, pos_weights):
    # Sanitize unlogged targets (where mask == 0, target is NaN) so BCE doesn't propagate NaN
    safe_targets = torch.nan_to_num(targets, nan=0.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    loss_matrix = bce(logits, safe_targets) # (batch, 6)
    masked_loss = loss_matrix * masks
    denom = masks.sum() + 1e-7
    return masked_loss.sum() / denom

def eval_predictions(y_true, y_pred, y_prob):
    pr_auc = average_precision_score(y_true, y_prob)
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
        else:
            roc_auc = np.nan
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

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP S5: MULTI-TASK NEURAL EXPERIMENTS & CORRECTED PERSISTENCE")
    print("=" * 75)

    data = np.load(DATA_PATH)
    X = data["X"]          # (N, 7, 43)
    Y = data["Y"]          # (N, 6)
    M = data["M"]          # (N, 6)
    folds = data["folds"]  # (N,)
    p_ids = data["p_ids"]  # (N,)
    feature_names = list(data["feature_names"])
    target_names = list(data["target_names"])

    N, W, D = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Total windows: {N} | Seq len: {W} | Daily features: {D}")

    # =========================================================================
    # PART 1: EVALUATE CORRECTED LAST-OBSERVED-VALUE PERSISTENCE
    # =========================================================================
    print("\n--- Evaluating Corrected Persistence (Last Observed Value in Window) ---")
    persist_records = []
    for s_idx, sym in enumerate(target_names):
        sym_feat_idx = feature_names.index(f"{sym}_ord")
        for f in range(5):
            test_mask = (folds == f) & (M[:, s_idx] == 1)
            y_te = Y[test_mask, s_idx]
            win_sym_vals = X[test_mask, :, sym_feat_idx] # (N_te, 7)
            
            y_pred = np.zeros(len(y_te), dtype=int)
            for i in range(len(y_te)):
                row_vals = win_sym_vals[i]
                observed = row_vals[~np.isnan(row_vals)]
                if len(observed) > 0:
                    y_pred[i] = int(observed[-1] >= 3)
                else:
                    y_pred[i] = 0 # fallback if no observation in 7 days
                    
            y_prob = y_pred.astype(float)
            res = eval_predictions(y_te, y_pred, y_prob)
            n_pos = int(np.sum(y_te == 1))
            persist_records.append({
                "symptom": sym,
                "fold": f,
                "model": "Persistence_LastObserved",
                "prevalence": float(np.mean(y_te)),
                "n_test": len(y_te),
                "n_pos": n_pos,
                "low_pos_warning": int(n_pos < 30),
                **res
            })

    df_persist = pd.DataFrame(persist_records)
    df_persist.to_csv(RESULTS_DIR / "corrected_persistence_results.csv", index=False)
    print(f"Saved corrected persistence results to: {RESULTS_DIR / 'corrected_persistence_results.csv'}")

    # =========================================================================
    # PART 2: NEURAL MODELS (MLP & GRU) WITH NESTED EARLY STOPPING
    # =========================================================================
    neural_records = []
    training_histories = {"MLP": {}, "GRU": {}}

    for f in range(5):
        print(f"\n==================== OUTER FOLD {f} ====================")
        outer_train_idx = np.where(folds != f)[0]
        outer_test_idx = np.where(folds == f)[0]

        # Verify participant disjointness
        train_p = set(p_ids[outer_train_idx])
        test_p = set(p_ids[outer_test_idx])
        assert len(train_p.intersection(test_p)) == 0, "Outer leak detected!"

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

        # Preprocessing: Fit Imputer & Scaler on INNER TRAIN ONLY
        # Flattened features for fitting:
        X_inner_tr = X[inner_train_idx].reshape(-1, D)
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_inner_tr)

        X_inner_tr_imp = imputer.transform(X_inner_tr)
        scaler = StandardScaler()
        scaler.fit(X_inner_tr_imp)

        def transform_features(X_raw):
            n_sub, w_sub, d_sub = X_raw.shape
            flat_2d = X_raw.reshape(-1, d_sub)
            imp_2d = imputer.transform(flat_2d)
            scaled_2d = scaler.transform(imp_2d)
            scaled_seq = scaled_2d.reshape(n_sub, w_sub, d_sub)
            scaled_flat = scaled_2d.reshape(n_sub, w_sub * d_sub)
            return scaled_seq, scaled_flat

        X_tr_seq, X_tr_flat = transform_features(X[inner_train_idx])
        X_val_seq, X_val_flat = transform_features(X[inner_val_idx])
        X_te_seq, X_te_flat = transform_features(X[outer_test_idx])

        Y_tr, M_tr = Y[inner_train_idx], M[inner_train_idx]
        Y_val, M_val = Y[inner_val_idx], M[inner_val_idx]
        Y_te, M_te = Y[outer_test_idx], M[outer_test_idx]

        # Compute positive weights per target on INNER TRAIN ONLY
        pos_weights_list = []
        for s in range(6):
            m_s = M_tr[:, s] == 1
            n_pos = np.sum(Y_tr[m_s, s] == 1)
            n_neg = np.sum(Y_tr[m_s, s] == 0)
            pw = n_neg / max(n_pos, 1)
            pos_weights_list.append(pw)
        pos_weights_tensor = torch.tensor(pos_weights_list, dtype=torch.float32).to(device)

        # Datasets & Loaders
        ds_tr = WindowDataset(X_tr_seq, X_tr_flat, Y_tr, M_tr)
        ds_val = WindowDataset(X_val_seq, X_val_flat, Y_val, M_val)
        ds_te = WindowDataset(X_te_seq, X_te_flat, Y_te, M_te)

        loader_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=128, shuffle=False)
        loader_te = DataLoader(ds_te, batch_size=128, shuffle=False)

        # Train both models
        models = {
            "MLP": MultiTaskMLP(in_features=W * D, num_targets=6, dropout=0.3).to(device),
            "GRU": MultiTaskGRU(in_features=D, hidden_size=32, num_targets=6, dropout=0.3).to(device),
        }

        for model_name, model in models.items():
            set_seed(SEED + f * 100)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            
            best_val_loss = float("inf")
            best_model_state = None
            history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": []}

            epochs = 60
            patience = 12
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                model.train()
                tr_loss_accum = 0.0
                tr_batches = 0
                for x_seq_b, x_flat_b, y_b, m_b in loader_tr:
                    x_seq_b = x_seq_b.to(device)
                    x_flat_b = x_flat_b.to(device)
                    y_b = y_b.to(device)
                    m_b = m_b.to(device)

                    optimizer.zero_grad()
                    if model_name == "MLP":
                        logits = model(x_flat_b)
                    else:
                        logits = model(x_seq_b)

                    loss = compute_masked_loss(logits, y_b, m_b, pos_weights_tensor)
                    loss.backward()
                    optimizer.step()

                    tr_loss_accum += loss.item()
                    tr_batches += 1

                avg_tr_loss = tr_loss_accum / max(tr_batches, 1)

                # Validation phase
                model.eval()
                val_loss_accum = 0.0
                val_batches = 0
                all_val_probs = []
                all_val_targets = []
                all_val_masks = []

                with torch.no_grad():
                    for x_seq_b, x_flat_b, y_b, m_b in loader_val:
                        x_seq_b = x_seq_b.to(device)
                        x_flat_b = x_flat_b.to(device)
                        y_b = y_b.to(device)
                        m_b = m_b.to(device)

                        if model_name == "MLP":
                            logits = model(x_flat_b)
                        else:
                            logits = model(x_seq_b)

                        loss = compute_masked_loss(logits, y_b, m_b, pos_weights_tensor)
                        val_loss_accum += loss.item()
                        val_batches += 1

                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_val_probs.append(probs)
                        all_val_targets.append(y_b.cpu().numpy())
                        all_val_masks.append(m_b.cpu().numpy())

                avg_val_loss = val_loss_accum / max(val_batches, 1)
                cat_val_probs = np.vstack(all_val_probs)
                cat_val_targets = np.vstack(all_val_targets)
                cat_val_masks = np.vstack(all_val_masks)

                # Compute mean PR-AUC across valid validation targets
                pr_aucs = []
                for s in range(6):
                    v_mask = cat_val_masks[:, s] == 1
                    if np.sum(cat_val_targets[v_mask, s] == 1) > 0:
                        pr = average_precision_score(cat_val_targets[v_mask, s], cat_val_probs[v_mask, s])
                        pr_aucs.append(pr)
                mean_val_pr = float(np.mean(pr_aucs)) if pr_aucs else 0.0

                history["epoch"].append(epoch)
                history["train_loss"].append(avg_tr_loss)
                history["val_loss"].append(avg_val_loss)
                history["val_pr_auc"].append(mean_val_pr)

                # Early stopping on inner validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Save fold history for plotting
            training_histories[model_name][f"fold_{f}"] = history

            # Load best model checkpoint for evaluation on OUTER TEST
            model.load_state_dict(best_model_state)
            model.eval()

            all_te_probs = []
            all_te_targets = []
            all_te_masks = []
            with torch.no_grad():
                for x_seq_b, x_flat_b, y_b, m_b in loader_te:
                    x_seq_b = x_seq_b.to(device)
                    x_flat_b = x_flat_b.to(device)

                    if model_name == "MLP":
                        logits = model(x_flat_b)
                    else:
                        logits = model(x_seq_b)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_te_probs.append(probs)
                    all_te_targets.append(y_b.numpy())
                    all_te_masks.append(m_b.numpy())

            te_probs = np.vstack(all_te_probs)
            te_targets = np.vstack(all_te_targets)
            te_masks = np.vstack(all_te_masks)

            # Evaluate each target independently using the test mask
            for s_idx, sym in enumerate(target_names):
                t_mask = te_masks[:, s_idx] == 1
                y_true = te_targets[t_mask, s_idx]
                y_prob = te_probs[t_mask, s_idx]
                y_pred = (y_prob >= 0.5).astype(int)

                res = eval_predictions(y_true, y_pred, y_prob)
                n_pos = int(np.sum(y_true == 1))

                rec = {
                    "symptom": sym,
                    "fold": f,
                    "model": model_name,
                    "prevalence": float(np.mean(y_true)),
                    "n_test": len(y_true),
                    "n_pos": n_pos,
                    "low_pos_warning": int(n_pos < 30),
                    **res
                }
                neural_records.append(rec)

    df_neural = pd.DataFrame(neural_records)
    df_neural.to_csv(RESULTS_DIR / "neural_groupkfold_results.csv", index=False)
    print(f"\nSaved neural fold results to: {RESULTS_DIR / 'neural_groupkfold_results.csv'}")

    # =========================================================================
    # PART 3: AGGREGATE SUMMARY & CONSOLIDATED COMPARISON TABLE
    # =========================================================================
    # Load classical baselines
    df_base = pd.read_csv(RESULTS_DIR / "baselines_groupkfold_results.csv")
    # Filter out original persistence, keep LogReg and CatBoost
    df_base_filtered = df_base[df_base["model"].isin(["LogReg", "CatBoost"])].copy()
    
    # Add n_pos and low_pos_warning to classical baselines for consistency
    pos_map = df_persist.set_index(["symptom", "fold"])["n_pos"].to_dict()
    df_base_filtered["n_pos"] = df_base_filtered.apply(lambda r: pos_map.get((r["symptom"], r["fold"]), np.nan), axis=1)
    df_base_filtered["low_pos_warning"] = (df_base_filtered["n_pos"] < 30).astype(int)

    # Combine all 5 models
    df_all_folds = pd.concat([df_persist, df_base_filtered, df_neural], ignore_index=True)
    df_all_folds.to_csv(RESULTS_DIR / "all_models_groupkfold_folds.csv", index=False)

    # Compute Summary (Mean +/- Std)
    summary_rows = []
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall", "Brier"]
    
    for (sym, model_name), grp in df_all_folds.groupby(["symptom", "model"]):
        row = {
            "symptom": sym,
            "model": model_name,
            "prevalence_mean": grp["prevalence"].mean(),
            "n_test_mean": grp["n_test"].mean(),
            "folds_with_low_pos": grp["low_pos_warning"].sum(),
        }
        for m in metrics:
            row[f"{m}_mean"] = grp[m].mean()
            row[f"{m}_std"] = grp[m].std()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    # Sort order
    model_order = ["Persistence_LastObserved", "LogReg", "CatBoost", "MLP", "GRU"]
    df_summary["model_rank"] = df_summary["model"].map(lambda m: model_order.index(m) if m in model_order else 99)
    df_summary = df_summary.sort_values(["symptom", "model_rank"]).drop(columns=["model_rank"])
    df_summary.to_csv(RESULTS_DIR / "all_models_summary.csv", index=False)
    print(f"Saved complete multi-model summary to: {RESULTS_DIR / 'all_models_summary.csv'}")

    # Display console table
    print("\n" + "=" * 95)
    print(f"{'Symptom':12s} | {'Model':26s} | {'PR-AUC (mean±std)':18s} | {'ROC-AUC':14s} | {'F1':14s} | {'Brier':10s}")
    print("-" * 95)
    for _, r in df_summary.iterrows():
        print(
            f"{r['symptom']:12s} | {r['model']:26s} | "
            f"{r['PR-AUC_mean']:.3f} ± {r['PR-AUC_std']:.3f}    | "
            f"{r['ROC-AUC_mean']:.3f} ± {r['ROC-AUC_std']:.3f} | "
            f"{r['F1_mean']:.3f} ± {r['F1_std']:.3f} | "
            f"{r['Brier_mean']:.3f}"
        )

    # =========================================================================
    # PART 4: GENERATE REPORT-READY PLOTS
    # =========================================================================
    print("\n--- Generating Publication Plots ---")
    
    # Plot 1: PR-AUC Comparison Across Models and Symptoms
    plt.figure(figsize=(12, 6))
    symptoms = list(target_names)
    n_syms = len(symptoms)
    x_positions = np.arange(n_syms)
    bar_width = 0.16
    
    colors = {
        "Persistence_LastObserved": "#7f7f7f",
        "LogReg": "#1f77b4",
        "CatBoost": "#2ca02c",
        "MLP": "#ff7f0e",
        "GRU": "#d62728",
    }
    labels = {
        "Persistence_LastObserved": "Persistence (Last Obs)",
        "LogReg": "Logistic Reg",
        "CatBoost": "CatBoost",
        "MLP": "MLP (Flat)",
        "GRU": "GRU (Temporal)",
    }

    for idx, m_name in enumerate(model_order):
        m_data = df_summary[df_summary["model"] == m_name].set_index("symptom").loc[symptoms]
        means = m_data["PR-AUC_mean"].values
        stds = m_data["PR-AUC_std"].values
        plt.bar(
            x_positions + idx * bar_width,
            means,
            yerr=stds,
            width=bar_width,
            label=labels.get(m_name, m_name),
            color=colors.get(m_name, "blue"),
            capsize=3,
            alpha=0.9
        )

    plt.xticks(x_positions + bar_width * 2, [s.capitalize() for s in symptoms], fontsize=11, fontweight="bold")
    plt.ylabel("PR-AUC (Average Precision)", fontsize=12)
    plt.title("Prospectively Validated PR-AUC on Unseen Participants (5-Fold GroupKFold)", fontsize=13, fontweight="bold")
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_auc_model_comparison.png", dpi=300)
    plt.close()

    # Plot 2: Training & Validation Curves (Fold 0 as representative)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m_name, color in [("MLP", "#ff7f0e"), ("GRU", "#d62728")]:
        h = training_histories[m_name]["fold_0"]
        axes[0].plot(h["epoch"], h["train_loss"], label=f"{m_name} Train", color=color, linestyle="--")
        axes[0].plot(h["epoch"], h["val_loss"], label=f"{m_name} Val", color=color, linewidth=2)
        axes[1].plot(h["epoch"], h["val_pr_auc"], label=f"{m_name} Val PR-AUC", color=color, linewidth=2)

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Masked BCE Loss", fontsize=11)
    axes[0].set_title("Training vs Inner-Validation Loss (Fold 0)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Mean Validation PR-AUC", fontsize=11)
    axes[1].set_title("Validation PR-AUC Across Epochs (Fold 0)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "neural_learning_curves_fold0.png", dpi=300)
    plt.close()
    print(f"Saved publication plots to {PLOTS_DIR}")

    # =========================================================================
    # PART 5: REPRODUCIBILITY NOTE
    # =========================================================================
    reproducibility_meta = {
        "random_seed": SEED,
        "device": str(device),
        "dataset_path": str(DATA_PATH),
        "n_samples": int(N),
        "seq_len": int(W),
        "feat_dim": int(D),
        "cv_scheme": "5-fold GroupKFold by participant ID",
        "inner_cv_scheme": "Nested 5-fold GroupKFold on outer-train for early stopping",
        "models": {
            "MLP": {"in_features": W * D, "hidden": [64, 32], "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3, "max_epochs": 60, "patience": 12},
            "GRU": {"in_features": D, "hidden_size": 32, "layers": 1, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3, "max_epochs": 60, "patience": 12},
        }
    }
    with open(RESULTS_DIR / "reproducibility_metadata.json", "w") as f:
        json.dump(reproducibility_meta, f, indent=2)
    print(f"Saved reproducibility metadata to: {RESULTS_DIR / 'reproducibility_metadata.json'}")

if __name__ == "__main__":
    main()
