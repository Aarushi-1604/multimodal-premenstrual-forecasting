"""
verification/run_pass2_checks.py
Collects all quantitative outputs, tables, and checks for Pass 2 Verification.
Strictly read-only on existing project files. Outputs purely aggregate statistics.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(".").resolve()
VERIF_DIR = PROJECT_ROOT / "verification"

def run_task1():
    print("=== TASK 1: FEATURE INVENTORY ===")
    data = np.load("data/processed/windows_dataset.npz")
    feature_names = list(data["feature_names"])
    print(f"Total features in X: {len(feature_names)}")
    # We will generate the detailed markdown table in the report
    has_phase = any("phase" in f.lower() for f in feature_names)
    print(f"Any feature named or derived from 'phase': {has_phase}")
    hormones = [f for f in feature_names if f in ["lh", "estrogen", "pdg", "lh_was_missing", "estrogen_was_missing", "pdg_was_missing"]]
    print(f"Hormone features: {hormones}")
    temp_feats = [f for f in feature_names if "temp" in f]
    print(f"Temperature features: {temp_feats}")
    rhr_feats = [f for f in feature_names if "rhr" in f]
    print(f"RHR features: {rhr_feats}")

def run_task2():
    print("\n=== TASK 2: TARGET MASK M CHECKS ===")
    data = np.load("data/processed/windows_dataset.npz")
    Y = data["Y"]
    M = data["M"]
    folds = data["folds"]
    target_names = list(data["target_names"])
    df_folds = pd.read_csv("results/all_models_groupkfold_folds.csv")

    # a) Reproduce n_test and n_pos
    # Let's check across 5 folds and 6 targets
    print("Fold & Target reproduction test:")
    all_pass = True
    rows_task2a = []
    for f in range(5):
        for j, sym in enumerate(target_names):
            # evaluation row condition: folds == f and M[:, j] == 1
            cond = (folds == f) & (M[:, j] == 1)
            n_calc = int(np.sum(cond))
            pos_calc = int(np.sum(Y[cond, j]))

            # lookup in all_models_groupkfold_folds for CatBoost (or any model)
            sub = df_folds[(df_folds["fold"] == f) & (df_folds["symptom"] == sym)]
            n_expected = int(sub["n_test"].iloc[0])
            pos_expected = int(sub["n_pos"].iloc[0])

            status = "PASS" if (n_calc == n_expected and pos_calc == pos_expected) else "FAIL"
            if status == "FAIL":
                all_pass = False
            rows_task2a.append({
                "fold": f,
                "symptom": sym,
                "n_calc": n_calc,
                "n_expected": n_expected,
                "pos_calc": pos_calc,
                "pos_expected": pos_expected,
                "status": status
            })
    print(f"Task 2a all 30 comparisons matched: {all_pass}")

    # b) What value does Y hold where M == 0?
    print("\nValues in Y where M == 0:")
    for j, sym in enumerate(target_names):
        m0 = M[:, j] == 0
        y_m0 = Y[m0, j]
        count_nan = int(np.isnan(y_m0).sum())
        count_zero = int((y_m0 == 0).sum())
        count_other = int((~np.isnan(y_m0) & (y_m0 != 0)).sum())
        print(f"  {sym:12s}: M==0 count = {m0.sum()} | NaN count = {count_nan} | 0 count = {count_zero} | other = {count_other}")

    # d) Synthetic unit test for masked loss invariance
    print("\nRunning synthetic unit test for masked BCE loss...")
    def compute_masked_loss(logits, targets, masks, pos_weights):
        safe_targets = torch.nan_to_num(targets, nan=0.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
        loss_matrix = bce(logits, safe_targets)
        masked_loss = loss_matrix * masks
        denom = masks.sum() + 1e-7
        return masked_loss.sum() / denom

    torch.manual_seed(42)
    B = 16
    logits = torch.randn(B, 6, requires_grad=True)
    targets1 = torch.randint(0, 2, (B, 6)).float()
    masks = torch.randint(0, 2, (B, 6)).float()
    # set NaN where mask == 0 to mimic real dataset
    targets1[masks == 0] = float("nan")
    pos_weights = torch.tensor([1.5, 2.0, 0.8, 1.2, 3.0, 1.1])

    loss1 = compute_masked_loss(logits, targets1, masks, pos_weights)
    loss1.backward()
    grad1 = logits.grad.clone()

    # Now change targets only at M == 0 positions (e.g. from nan to 999.0 or 0.0 or 1.0)
    logits.grad.zero_()
    targets2 = targets1.clone()
    targets2[masks == 0] = 999.0 # completely different values
    loss2 = compute_masked_loss(logits, targets2, masks, pos_weights)
    loss2.backward()
    grad2 = logits.grad.clone()

    loss_diff = torch.abs(loss1 - loss2).item()
    grad_diff = torch.max(torch.abs(grad1 - grad2)).item()
    test_d_pass = (loss_diff == 0.0) and (grad_diff == 0.0)
    print(f"Synthetic Unit Test: loss_diff = {loss_diff}, max_grad_diff = {grad_diff} -> {'PASS' if test_d_pass else 'FAIL'}")

    return rows_task2a

if __name__ == "__main__":
    run_task1()
    run_task2()
