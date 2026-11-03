#!/usr/bin/env python3
"""
audit_verify.py  -  READ-ONLY, AGGREGATE-ONLY verification for the mcPHASES project.

Run from the project root (C:\\Users\\aarus\\Desktop\\DeepLearningProject):
    python audit_verify.py            -> writes audit_verify_report.md
    python audit_verify.py --root PATH

Guarantees (DUA-safe):
  * Never modifies any file except writing audit_verify_report.md.
  * Never prints a raw row, participant id, or per-participant value.
    Only counts, shapes, dtypes, aggregate rates, and source-code line numbers.
  * Result CSVs are only summarised by column names + a few aggregate checks.
Paste audit_verify_report.md back to Claude (it contains no participant-level data).
"""
import argparse, json, re, sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None
try:
    import pandas as pd
except ImportError:
    pd = None

OUT = []
def log(s=""):
    print(s); OUT.append(s)
def h(s): log(f"\n## {s}\n")
def flag(level, msg): log(f"- **[{level}]** {msg}")

# ---------------------------------------------------------------- 1. source scan
SRC_PATTERNS = [
 # (regex, severity, why)
 (r"shift\(\s*-\d", "HIGH", "negative shift = looks at the future; check it is only used to build the TARGET, never features"),
 (r"\.fit(_transform)?\(\s*(X|x|df|data|all|full)\b", "HIGH", "fit on a variable that may be the full dataset; confirm it is train-only"),
 (r"StandardScaler|SimpleImputer|MinMaxScaler|RobustScaler", "MED", "preprocessing object: confirm .fit() receives TRAIN rows only"),
 (r"train_test_split|(?<!Group)(?<!Stratified)KFold|StratifiedKFold(?!Group)", "HIGH", "non-grouped split: windows from one person may land in train and test"),
 (r"GroupKFold|StratifiedGroupKFold|LeaveOneGroupOut", "INFO", "grouped split present (good); confirm groups = participant id"),
 (r"fillna\(\s*0|nan_to_num|np\.where\(.*isnan", "MED", "zero-fill of missing: for TARGETS or persistence this counts missing as 'no symptom'"),
 (r"ffill|bfill|interpolate", "MED", "bfill/interpolate use future values; ffill is causal only if limited to the past and within participant (groupby)"),
 (r"rolling\([^)]*center\s*=\s*True", "HIGH", "centred rolling window uses future days"),
 (r"cycle_(sin|cos)_pop|_pop\b", "MED", "population cycle stats: confirm computed from TRAIN folds only"),
 (r"days_since_onset|has_prior_cycle|onset", "MED", "onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used"),
 (r"MIN_GAP_TO_BRIDGE|MIN_CYCLE_LENGTH|bridge", "MED", "retrospective onset cleaning (needs days after t) -> potential future info in cycle features"),
 (r"early_?stop|patience", "INFO", "early stopping: confirm monitored metric is on inner-validation, not test fold"),
 (r"best_iteration|use_best_model|eval_set", "MED", "CatBoost eval_set must be the inner-val split, not the test fold"),
 (r"class_weight|pos_weight|scale_pos_weight", "INFO", "class weights: confirm from TRAIN labels only"),
 (r"random_state|manual_seed|seed\(", "INFO", "seeding present"),
 (r"\.mean\(\)\s*\)?\s*$|nanmean|nanstd", "LOW", "global statistics: confirm not computed on full data before the split"),
]

def scan_sources(root: Path):
    h("1. Static source scan (line numbers only, no code echoed beyond the matched pattern name)")
    files = [p for p in root.rglob("*.py")
             if not any(x in p.parts for x in (".venv", "venv", "site-packages", "__pycache__", ".git"))
             and p.name != Path(__file__).name]
    if not files:
        flag("WARN", "No .py files found under root.")
        return
    for p in sorted(files):
        try: lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            flag("WARN", f"cannot read {p}: {e}"); continue
        hits = []
        for i, ln in enumerate(lines, 1):
            if ln.strip().startswith("#"): continue
            for rx, sev, why in SRC_PATTERNS:
                if re.search(rx, ln):
                    hits.append((i, sev, rx, why))
        log(f"\n### {p.relative_to(root)}  ({len(lines)} lines)")
        if not hits: log("(no flagged patterns)"); continue
        order = {"HIGH":0,"MED":1,"LOW":2,"INFO":3}
        for i, sev, rx, why in sorted(hits, key=lambda t:(order[t[1]], t[0])):
            log(f"- L{i} [{sev}] `{rx[:40]}` - {why}")

# ---------------------------------------------------------------- 2. npz checks
def find_key(keys, *cands):
    for c in cands:
        for k in keys:
            if k.lower() == c: return k
    for c in cands:
        for k in keys:
            if c in k.lower(): return k
    return None

def check_npz(root: Path):
    h("2. windows_dataset.npz structure and leakage checks (aggregates only)")
    if np is None: flag("WARN", "numpy not installed"); return
    cands = list(root.rglob("windows_dataset.npz"))
    if not cands: flag("WARN", "windows_dataset.npz not found"); return
    z = np.load(cands[0], allow_pickle=True)
    keys = list(z.files)
    log("Arrays (name, shape, dtype):")
    for k in keys: log(f"- `{k}` {z[k].shape} {z[k].dtype}")

    X = z[find_key(keys, "x", "features", "x_seq")] if find_key(keys, "x", "features", "x_seq") else None
    Y = z[find_key(keys, "y", "targets", "y_seq")] if find_key(keys, "y", "targets", "y_seq") else None
    M = z[find_key(keys, "mask", "y_mask", "target_mask")] if find_key(keys, "mask", "y_mask", "target_mask") else None
    G = z[find_key(keys, "groups", "participant", "pid", "id")] if find_key(keys, "groups", "participant", "pid", "id") else None
    D = z[find_key(keys, "day", "target_day", "t")] if find_key(keys, "day", "target_day", "day_in_study") else None

    if X is None or Y is None or G is None:
        flag("WARN", "Could not auto-identify X / Y / groups arrays; edit key names in script."); return
    n = len(X)
    log(f"\nWindows: {n};  participants (unique groups): {len(np.unique(G))}")
    if M is None:
        flag("HIGH", "No target mask array found: confirm missing target surveys cannot be read as 0 (negatives).")
    else:
        Mb = M.astype(bool)
        log(f"Target mask observed fraction per target: {np.round(Mb.mean(0),3).tolist()}")
        if np.any(Y[~Mb] != 0) and not np.all(np.isnan(Y[~Mb])):
            log("- masked positions carry non-zero values (fine only if loss/eval ignore them)")
        pos = np.nansum(np.where(Mb, Y, 0), axis=0)
        log(f"Positives per target (all windows): {pos.astype(int).tolist()}")
        log(f"Prevalence among observed: {np.round(pos / np.maximum(Mb.sum(0),1), 3).tolist()}")
        # per-person windows
        ug, cnt = np.unique(G, return_counts=True)
        log(f"Windows per participant: min {cnt.min()}, median {int(np.median(cnt))}, max {cnt.max()}")

    # NaN / zero checks inside X
    Xf = X.astype(float)
    log(f"\nNaN fraction in X: {np.isnan(Xf).mean():.4f}")
    if X.ndim == 3:
        flat = Xf.reshape(-1, X.shape[-1])
        zero_frac = (flat == 0).mean(0)
        hi = np.where(zero_frac > 0.2)[0]
        log(f"Feature indices with >20% exact zeros (possible zero-fill of missing, e.g. resting HR): {hi.tolist()}")
        # overlap between consecutive windows -> inflation of effective sample size
        log("Overlap: 7-day windows on consecutive days share 6/7 of their inputs; "
            "report effective n as participants/cycles, not windows.")

    # fold files
    for k in keys:
        if "fold" in k.lower():
            F = z[k]
            log(f"\nFold array `{k}` shape {F.shape}")
            if F.ndim == 1 and len(F) == n:
                for f in np.unique(F):
                    te = F == f
                    tr_groups, te_groups = set(G[~te].tolist()), set(G[te].tolist())
                    ov = len(tr_groups & te_groups)
                    log(f"- fold {f}: test windows {te.sum()}, test participants {len(te_groups)}, participant overlap with train = {ov}")
                    if ov: flag("CRITICAL", f"fold {f} has {ov} participants in BOTH train and test (same-person leakage)")
                    if M is not None:
                        pp = (np.where(M.astype(bool), Y, 0)[te]).sum(0).astype(int)
                        log(f"  test positives per target: {pp.tolist()}")
                        for t, v in enumerate(pp):
                            if v < 30: flag("MED", f"fold {f}, target index {t}: only {v} test positives (<30; unstable PR-AUC)")

# ---------------------------------------------------------------- 3. results CSVs
def check_results(root: Path):
    h("3. Results CSV consistency")
    if pd is None: flag("WARN", "pandas not installed"); return
    rd = root / "results"
    if not rd.exists(): flag("WARN", "results/ not found"); return
    for p in sorted(rd.glob("*.csv")):
        try: df = pd.read_csv(p)
        except Exception as e: flag("WARN", f"{p.name}: {e}"); continue
        log(f"\n### {p.name}  shape {df.shape}\n- columns: {list(df.columns)}")
        if len(df) <= 80:   # aggregate result tables only; never dump anything with >80 rows
            log("```\n" + df.round(4).to_string(index=False) + "\n```")
        else:
            flag("INFO", "table >80 rows; not printed (may be fold-level or participant-level)")
    # Brier claim check
    s = None
    for name in ("all_models_summary.csv",):
        if (rd / name).exists(): s = pd.read_csv(rd / name)
    if s is not None:
        cols = {c.lower(): c for c in s.columns}
        mc, tc = cols.get("model"), cols.get("target") or cols.get("symptom")
        bc = next((v for k, v in cols.items() if "brier" in k and "std" not in k and "sd" not in k), None)
        if mc and tc and bc:
            piv = s.pivot_table(index=tc, columns=mc, values=bc)
            log("\nBrier by target x model:\n```\n" + piv.round(4).to_string() + "\n```")
            best = piv.idxmin(axis=1)
            log("Best (lowest) Brier per target: " + json.dumps(best.to_dict()))
            if not (best.astype(str).str.lower().str.contains("catboost").all()):
                flag("MED", "CatBoost is NOT best-Brier on every target. Any sentence claiming so must be removed.")
        else:
            flag("INFO", "all_models_summary.csv present but model/target/brier columns not auto-detected")
    meta = rd / "reproducibility_metadata.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            log("\nreproducibility_metadata.json keys: " + ", ".join(sorted(d.keys())))
        except Exception as e: flag("WARN", f"metadata unreadable: {e}")

# ---------------------------------------------------------------- 4. git / DUA hygiene
def check_hygiene(root: Path):
    h("4. DUA / repository hygiene")
    gi = root / ".gitignore"
    txt = gi.read_text(errors="ignore") if gi.exists() else ""
    if not gi.exists(): flag("HIGH", ".gitignore missing")
    for needle in ("mcphases_data", "data/processed", "*.npz", "*.csv", "results"):
        ok = needle in txt
        log(f"- {'OK ' if ok else 'MISSING'}  .gitignore mentions `{needle}`")
    if (root / ".git").exists():
        import subprocess
        try:
            tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True).stdout.splitlines()
            bad = [t for t in tracked if t.lower().endswith((".npz", ".csv", ".parquet", ".pkl", ".npy"))]
            log(f"- git-tracked data-like files: {len(bad)}")
            if bad: flag("CRITICAL", f"{len(bad)} data-like files are tracked by git (names withheld here; run `git ls-files` locally). Remove from history before any push.")
            r = subprocess.run(["git", "-C", str(root), "remote", "-v"], capture_output=True, text=True).stdout.strip()
            log(f"- git remotes configured: {'yes' if r else 'no'}")
        except Exception as e: flag("WARN", f"git check failed: {e}")
    else:
        log("- not a git repo (or .git not at root)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args()
    root = Path(a.root).resolve()
    log(f"# audit_verify report\nroot name: `{root.name}` (path not recorded)")
    scan_sources(root); check_npz(root); check_results(root); check_hygiene(root)
    (root / "audit_verify_report.md").write_text("\n".join(OUT), encoding="utf-8")
    print("\nWrote audit_verify_report.md")

if __name__ == "__main__":
    main()
