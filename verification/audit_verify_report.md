# audit_verify report
root name: `DeepLearningProject` (path not recorded)

## 1. Static source scan (line numbers only, no code echoed beyond the matched pattern name)


### inspect_raw_schemas.py  (147 lines)
(no flagged patterns)

### src\data\build_master.py  (299 lines)
- L242 [HIGH] `shift\(\s*-\d` - negative shift = looks at the future; check it is only used to build the TARGET, never features
- L243 [HIGH] `shift\(\s*-\d` - negative shift = looks at the future; check it is only used to build the TARGET, never features
- L4 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L34 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L36 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L38 [MED] `MIN_GAP_TO_BRIDGE|MIN_CYCLE_LENGTH|bridg` - retrospective onset cleaning (needs days after t) -> potential future info in cycle features
- L42 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L43 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L66 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L69 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L70 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L71 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L72 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L74 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L75 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L82 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L84 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L91 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L92 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L93 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L95 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L97 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L108 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L110 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L152 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L153 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L161 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L162 [MED] `fillna\(\s*0|nan_to_num|np\.where\(.*isn` - zero-fill of missing: for TARGETS or persistence this counts missing as 'no symptom'
- L162 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L162 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L163 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L164 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L164 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L165 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L165 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L169 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L170 [MED] `fillna\(\s*0|nan_to_num|np\.where\(.*isn` - zero-fill of missing: for TARGETS or persistence this counts missing as 'no symptom'
- L170 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L172 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L173 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L260 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L261 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only

### src\data\make_windows.py  (218 lines)
- L38 [MED] `cycle_(sin|cos)_pop|_pop\b` - population cycle stats: confirm computed from TRAIN folds only
- L38 [MED] `days_since_onset|has_prior_cycle|onset` - onset-derived features: onset detection bridges gaps/merges with lookahead; confirm only onsets already observable at day t are used
- L6 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L15 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L119 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L189 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id

### src\models\baselines.py  (216 lines)
- L17 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L18 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L88 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L92 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L103 [MED] `fillna\(\s*0|nan_to_num|np\.where\(.*isn` - zero-fill of missing: for TARGETS or persistence this counts missing as 'no symptom'
- L3 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L120 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L121 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L139 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only

### src\models\train_neural.py  (580 lines)
- L27 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L28 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L105 [MED] `fillna\(\s*0|nan_to_num|np\.where\(.*isn` - zero-fill of missing: for TARGETS or persistence this counts missing as 'no symptom'
- L228 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L232 [MED] `StandardScaler|SimpleImputer|MinMaxScale` - preprocessing object: confirm .fit() receives TRAIN rows only
- L457 [LOW] `\.mean\(\)\s*\)?\s*$|nanmean|nanstd` - global statistics: confirm not computed on full data before the split
- L8 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L9 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L29 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L45 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L46 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L47 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L48 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L50 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L103 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L106 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L135 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L213 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L253 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L259 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L260 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L278 [INFO] `random_state|manual_seed|seed\(` - seeding present
- L286 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L287 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L305 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L334 [INFO] `class_weight|pos_weight|scale_pos_weight` - class weights: confirm from TRAIN labels only
- L366 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L368 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L369 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L526 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L568 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L569 [INFO] `GroupKFold|StratifiedGroupKFold|LeaveOne` - grouped split present (good); confirm groups = participant id
- L571 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold
- L572 [INFO] `early_?stop|patience` - early stopping: confirm monitored metric is on inner-validation, not test fold

## 2. windows_dataset.npz structure and leakage checks (aggregates only)

Arrays (name, shape, dtype):
- `X` (3404, 7, 43) float32
- `Y` (3404, 6) float32
- `M` (3404, 6) int32
- `folds` (3404,) int32
- `p_ids` (3404,) int64
- `q_days` (3404,) int32
- `t_days` (3404,) int32
- `fc_splits` (3404,) <U10
- `pers_eligible` (3404,) int32
- `feature_names` (43,) <U25
- `target_names` (6,) <U11

Windows: 3404;  participants (unique groups): 42
- **[HIGH]** No target mask array found: confirm missing target surveys cannot be read as 0 (negatives).

NaN fraction in X: 0.0523
Feature indices with >20% exact zeros (possible zero-fill of missing, e.g. resting HR): [0, 1, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 29, 34, 37, 39, 42]
Overlap: 7-day windows on consecutive days share 6/7 of their inputs; report effective n as participants/cycles, not windows.

Fold array `folds` shape (3404,)
- fold 0: test windows 717, test participants 9, participant overlap with train = 0
- fold 1: test windows 695, test participants 9, participant overlap with train = 0
- fold 2: test windows 664, test participants 8, participant overlap with train = 0
- fold 3: test windows 664, test participants 8, participant overlap with train = 0
- fold 4: test windows 664, test participants 8, participant overlap with train = 0

## 3. Results CSV consistency


### all_models_groupkfold_folds.csv  shape (150, 13)
- columns: ['symptom', 'fold', 'model', 'prevalence', 'n_test', 'n_pos', 'low_pos_warning', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Brier']
- **[INFO]** table >80 rows; not printed (may be fold-level or participant-level)

### all_models_summary.csv  shape (30, 17)
- columns: ['symptom', 'model', 'prevalence_mean', 'n_test_mean', 'folds_with_low_pos', 'PR-AUC_mean', 'PR-AUC_std', 'ROC-AUC_mean', 'ROC-AUC_std', 'F1_mean', 'F1_std', 'Precision_mean', 'Precision_std', 'Recall_mean', 'Recall_std', 'Brier_mean', 'Brier_std']
```
    symptom                    model  prevalence_mean  n_test_mean  folds_with_low_pos  PR-AUC_mean  PR-AUC_std  ROC-AUC_mean  ROC-AUC_std  F1_mean  F1_std  Precision_mean  Precision_std  Recall_mean  Recall_std  Brier_mean  Brier_std
   bloating Persistence_LastObserved           0.3094        607.2                   0       0.5683      0.1820        0.7774       0.0580   0.6848  0.1336          0.6856         0.1351       0.6841      0.1322      0.1690     0.0425
   bloating                   LogReg           0.3094        607.2                   0       0.6490      0.2070        0.8199       0.0658   0.6323  0.1698          0.5930         0.1283       0.6832      0.2162      0.1477     0.0231
   bloating                 CatBoost           0.3094        607.2                   0       0.6916      0.2225        0.8432       0.0740   0.6640  0.1695          0.6655         0.1100       0.6740      0.2216      0.1289     0.0208
   bloating                      MLP           0.3094        607.2                   0       0.6276      0.2328        0.8079       0.0623   0.6087  0.1731          0.5479         0.1525       0.6867      0.2032      0.1592     0.0196
   bloating                      GRU           0.3094        607.2                   0       0.6825      0.2015        0.8437       0.0511   0.6636  0.1529          0.6371         0.1131       0.7017      0.2018      0.1375     0.0225
     cramps Persistence_LastObserved           0.1575        607.2                   0       0.4109      0.1863        0.7541       0.0698   0.5793  0.1493          0.5820         0.1470       0.5766      0.1516      0.1136     0.0391
     cramps                   LogReg           0.1575        607.2                   0       0.5033      0.1699        0.8254       0.0619   0.4895  0.1640          0.3945         0.1539       0.6599      0.1620      0.1349     0.0327
     cramps                 CatBoost           0.1575        607.2                   0       0.5556      0.1906        0.8507       0.0716   0.5506  0.1707          0.5132         0.1684       0.5965      0.1778      0.1020     0.0299
     cramps                      MLP           0.1575        607.2                   0       0.5006      0.2461        0.8184       0.0717   0.4705  0.1763          0.3715         0.1619       0.6607      0.1690      0.1432     0.0322
     cramps                      GRU           0.1575        607.2                   0       0.5494      0.1960        0.8404       0.0655   0.5179  0.1318          0.4231         0.1139       0.6708      0.1581      0.1329     0.0371
    fatigue Persistence_LastObserved           0.5599        607.2                   0       0.7333      0.0483        0.7502       0.0496   0.7794  0.0493          0.7808         0.0475       0.7780      0.0510      0.2456     0.0500
    fatigue                   LogReg           0.5599        607.2                   0       0.8192      0.0521        0.7989       0.0527   0.7592  0.0518          0.7680         0.0591       0.7532      0.0680      0.1827     0.0248
    fatigue                 CatBoost           0.5599        607.2                   0       0.8509      0.0539        0.8315       0.0568   0.7927  0.0484          0.7915         0.0593       0.7953      0.0501      0.1637     0.0280
    fatigue                      MLP           0.5599        607.2                   0       0.7957      0.0611        0.7774       0.0602   0.7434  0.0470          0.7594         0.0654       0.7330      0.0691      0.1929     0.0216
    fatigue                      GRU           0.5599        607.2                   0       0.8242      0.0710        0.8070       0.0703   0.7666  0.0597          0.7770         0.0618       0.7593      0.0798      0.1786     0.0307
  headaches Persistence_LastObserved           0.2533        607.2                   0       0.4771      0.1144        0.7420       0.0518   0.6124  0.0925          0.6139         0.0930       0.6111      0.0921      0.1889     0.0444
  headaches                   LogReg           0.2533        607.2                   0       0.6456      0.0871        0.8187       0.0499   0.5859  0.0821          0.5134         0.0656       0.6885      0.1287      0.1636     0.0389
  headaches                 CatBoost           0.2533        607.2                   0       0.6594      0.0978        0.8327       0.0543   0.6149  0.0780          0.5744         0.0543       0.6720      0.1401      0.1434     0.0334
  headaches                      MLP           0.2533        607.2                   0       0.5792      0.1192        0.7754       0.0578   0.5509  0.0761          0.4852         0.0804       0.6511      0.1196      0.1763     0.0405
  headaches                      GRU           0.2533        607.2                   0       0.6354      0.1012        0.8160       0.0539   0.5914  0.0831          0.5494         0.0553       0.6603      0.1576      0.1563     0.0362
  moodswing Persistence_LastObserved           0.2977        607.2                   0       0.5300      0.1862        0.7488       0.0837   0.6422  0.1628          0.6437         0.1636       0.6407      0.1621      0.1915     0.0596
  moodswing                   LogReg           0.2977        607.2                   0       0.6465      0.1846        0.8232       0.0639   0.6411  0.1440          0.5821         0.1179       0.7166      0.1860      0.1540     0.0319
  moodswing                 CatBoost           0.2977        607.2                   0       0.6592      0.1937        0.8331       0.0765   0.6244  0.1746          0.5902         0.1153       0.6742      0.2381      0.1412     0.0286
  moodswing                      MLP           0.2977        607.2                   0       0.6152      0.1714        0.8029       0.0685   0.5953  0.1391          0.5242         0.1102       0.6959      0.1987      0.1702     0.0246
  moodswing                      GRU           0.2977        607.2                   0       0.6488      0.1833        0.8281       0.0585   0.6262  0.1640          0.5653         0.1366       0.7030      0.2015      0.1523     0.0259
sorebreasts Persistence_LastObserved           0.1161        607.2                   2       0.4061      0.2396        0.7677       0.0918   0.5810  0.1972          0.5857         0.1990       0.5765      0.1954      0.0728     0.0290
sorebreasts                   LogReg           0.1161        607.2                   2       0.4721      0.2422        0.8410       0.0839   0.4602  0.2008          0.3624         0.1830       0.6475      0.1978      0.1052     0.0369
sorebreasts                 CatBoost           0.1161        607.2                   2       0.5165      0.1866        0.8615       0.0502   0.5578  0.1345          0.5064         0.1179       0.6340      0.1813      0.0768     0.0401
sorebreasts                      MLP           0.1161        607.2                   2       0.4249      0.2305        0.8446       0.0728   0.4394  0.2100          0.3391         0.1895       0.6759      0.1507      0.1212     0.0399
sorebreasts                      GRU           0.1161        607.2                   2       0.4569      0.2495        0.8567       0.0862   0.4781  0.1803          0.3672         0.1503       0.6956      0.2298      0.1081     0.0427
```

### baselines_groupkfold_results.csv  shape (120, 11)
- columns: ['symptom', 'fold', 'model', 'prevalence', 'n_test', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Brier']
- **[INFO]** table >80 rows; not printed (may be fold-level or participant-level)

### corrected_persistence_results.csv  shape (30, 13)
- columns: ['symptom', 'fold', 'model', 'prevalence', 'n_test', 'n_pos', 'low_pos_warning', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Brier']
```
    symptom  fold                    model  prevalence  n_test  n_pos  low_pos_warning  PR-AUC  ROC-AUC     F1  Precision  Recall  Brier
     cramps     0 Persistence_LastObserved      0.2263     654    148                0  0.6143   0.8351 0.7458     0.7483  0.7432 0.1147
     cramps     1 Persistence_LastObserved      0.1741     586    102                0  0.3446   0.7004 0.5075     0.5152  0.5000 0.1689
     cramps     2 Persistence_LastObserved      0.0616     601     37                0  0.2271   0.6985 0.4384     0.4444  0.4324 0.0682
     cramps     3 Persistence_LastObserved      0.2441     594    145                0  0.6048   0.8255 0.7354     0.7329  0.7379 0.1296
     cramps     4 Persistence_LastObserved      0.0815     601     49                0  0.2636   0.7111 0.4694     0.4694  0.4694 0.0865
  moodswing     0 Persistence_LastObserved      0.3456     654    226                0  0.5831   0.7623 0.6889     0.6920  0.6858 0.2141
  moodswing     1 Persistence_LastObserved      0.4659     586    273                0  0.6363   0.7257 0.7070     0.7070  0.7070 0.2730
  moodswing     2 Persistence_LastObserved      0.1498     601     90                0  0.2359   0.6341 0.3778     0.3778  0.3778 0.1864
  moodswing     3 Persistence_LastObserved      0.2946     594    175                0  0.7178   0.8675 0.8138     0.8161  0.8114 0.1094
  moodswing     4 Persistence_LastObserved      0.2329     601    140                0  0.4771   0.7543 0.6237     0.6259  0.6214 0.1747
    fatigue     0 Persistence_LastObserved      0.5917     654    387                0  0.7630   0.7608 0.8052     0.8041  0.8062 0.2309
    fatigue     1 Persistence_LastObserved      0.5410     586    317                0  0.6479   0.6673 0.6921     0.6965  0.6877 0.3311
    fatigue     2 Persistence_LastObserved      0.5857     601    352                0  0.7486   0.7483 0.7909     0.7920  0.7898 0.2446
    fatigue     3 Persistence_LastObserved      0.5185     594    308                0  0.7465   0.7945 0.8013     0.8039  0.7987 0.2054
    fatigue     4 Persistence_LastObserved      0.5624     601    338                0  0.7605   0.7803 0.8077     0.8077  0.8077 0.2163
   bloating     0 Persistence_LastObserved      0.3410     654    223                0  0.6976   0.8413 0.7910     0.7928  0.7892 0.1422
   bloating     1 Persistence_LastObserved      0.4932     586    289                0  0.7070   0.7729 0.7687     0.7727  0.7647 0.2270
   bloating     2 Persistence_LastObserved      0.1265     601     76                0  0.3249   0.7213 0.5132     0.5132  0.5132 0.1231
   bloating     3 Persistence_LastObserved      0.3586     594    213                0  0.6922   0.8317 0.7840     0.7840  0.7840 0.1549
   bloating     4 Persistence_LastObserved      0.2280     601    137                0  0.4200   0.7200 0.5673     0.5652  0.5693 0.1980
sorebreasts     0 Persistence_LastObserved      0.1789     654    117                0  0.5749   0.8301 0.7241     0.7304  0.7179 0.0979
sorebreasts     1 Persistence_LastObserved      0.1280     586     75                0  0.3923   0.7573 0.5811     0.5890  0.5733 0.1058
sorebreasts     2 Persistence_LastObserved      0.0300     601     18                1  0.1376   0.6572 0.3429     0.3529  0.3333 0.0383
sorebreasts     3 Persistence_LastObserved      0.1987     594    118                0  0.7103   0.8858 0.8205     0.8276  0.8136 0.0707
sorebreasts     4 Persistence_LastObserved      0.0449     601     27                1  0.2154   0.7083 0.4364     0.4286  0.4444 0.0516
  headaches     0 Persistence_LastObserved      0.2920     654    191                0  0.5156   0.7459 0.6404     0.6421  0.6387 0.2095
  headaches     1 Persistence_LastObserved      0.2884     586    169                0  0.4361   0.6864 0.5536     0.5569  0.5503 0.2560
  headaches     2 Persistence_LastObserved      0.1664     601    100                0  0.3681   0.7241 0.5400     0.5400  0.5400 0.1531
  headaches     3 Persistence_LastObserved      0.3165     594    188                0  0.6574   0.8261 0.7627     0.7647  0.7606 0.1498
  headaches     4 Persistence_LastObserved      0.2030     601    122                0  0.4081   0.7275 0.5656     0.5656  0.5656 0.1764
```

### neural_groupkfold_results.csv  shape (60, 13)
- columns: ['symptom', 'fold', 'model', 'prevalence', 'n_test', 'n_pos', 'low_pos_warning', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Brier']
```
    symptom  fold model  prevalence  n_test  n_pos  low_pos_warning  PR-AUC  ROC-AUC     F1  Precision  Recall  Brier
     cramps     0   MLP      0.2263     654    148                0  0.7760   0.8905 0.6469     0.5381  0.8108 0.1395
  moodswing     0   MLP      0.3456     654    226                0  0.6940   0.8461 0.7178     0.6214  0.8496 0.1714
    fatigue     0   MLP      0.5917     654    387                0  0.7521   0.7643 0.8089     0.7736  0.8475 0.1840
   bloating     0   MLP      0.3410     654    223                0  0.7626   0.8349 0.6692     0.5833  0.7848 0.1745
sorebreasts     0   MLP      0.1789     654    117                0  0.6515   0.9130 0.6980     0.5746  0.8889 0.1201
  headaches     0   MLP      0.2920     654    191                0  0.5805   0.7559 0.5392     0.4379  0.7016 0.2101
     cramps     0   GRU      0.2263     654    148                0  0.7835   0.8975 0.6649     0.5571  0.8243 0.1342
  moodswing     0   GRU      0.3456     654    226                0  0.7503   0.8588 0.7352     0.6455  0.8540 0.1593
    fatigue     0   GRU      0.5917     654    387                0  0.8632   0.8287 0.8166     0.7749  0.8630 0.1635
   bloating     0   GRU      0.3410     654    223                0  0.8355   0.8774 0.7479     0.7143  0.7848 0.1314
sorebreasts     0   GRU      0.1789     654    117                0  0.6420   0.9141 0.6731     0.5385  0.8974 0.1172
  headaches     0   GRU      0.2920     654    191                0  0.7066   0.8131 0.6151     0.5122  0.7696 0.1887
     cramps     1   MLP      0.1741     586    102                0  0.3892   0.7211 0.3986     0.3161  0.5392 0.1948
  moodswing     1   MLP      0.4659     586    273                0  0.6732   0.7476 0.6765     0.6131  0.7546 0.2115
    fatigue     1   MLP      0.5410     586    317                0  0.7172   0.6853 0.6864     0.6462  0.7319 0.2296
   bloating     1   MLP      0.4932     586    289                0  0.7481   0.8083 0.7577     0.6804  0.8547 0.1851
sorebreasts     1   MLP      0.1280     586     75                0  0.4637   0.7778 0.3766     0.2744  0.6000 0.1712
  headaches     1   MLP      0.2884     586    169                0  0.5364   0.7238 0.5322     0.4255  0.7101 0.2249
     cramps     1   GRU      0.1741     586    102                0  0.4289   0.7491 0.4427     0.3709  0.5490 0.1782
  moodswing     1   GRU      0.4659     586    273                0  0.7352   0.7845 0.7208     0.6472  0.8132 0.1936
    fatigue     1   GRU      0.5410     586    317                0  0.7157   0.6951 0.6752     0.6817  0.6688 0.2280
   bloating     1   GRU      0.4932     586    289                0  0.7847   0.8428 0.8067     0.7201  0.9170 0.1666
sorebreasts     1   GRU      0.1280     586     75                0  0.4944   0.8009 0.4554     0.3622  0.6133 0.1433
  headaches     1   GRU      0.2884     586    169                0  0.5830   0.7534 0.5500     0.4762  0.6509 0.2001
     cramps     2   MLP      0.0616     601     37                0  0.3054   0.8065 0.2993     0.2000  0.5946 0.1171
  moodswing     2   MLP      0.1498     601     90                0  0.3685   0.7295 0.3776     0.3491  0.4111 0.1470
    fatigue     2   MLP      0.5857     601    352                0  0.8066   0.7830 0.7115     0.7690  0.6619 0.1924
   bloating     2   MLP      0.1265     601     76                0  0.2591   0.7144 0.3333     0.2959  0.3816 0.1391
sorebreasts     2   MLP      0.0300     601     18                1  0.1384   0.7560 0.1653     0.0971  0.5556 0.1073
  headaches     2   MLP      0.1664     601    100                0  0.4337   0.7244 0.4465     0.4174  0.4800 0.1460
     cramps     2   GRU      0.0616     601     37                0  0.4109   0.8779 0.4516     0.3750  0.5676 0.0799
  moodswing     2   GRU      0.1498     601     90                0  0.3428   0.7541 0.3560     0.3366  0.3778 0.1375
    fatigue     2   GRU      0.5857     601    352                0  0.7923   0.7883 0.7381     0.7750  0.7045 0.1883
   bloating     2   GRU      0.1265     601     76                0  0.3979   0.7701 0.4627     0.5345  0.4079 0.1079
sorebreasts     2   GRU      0.0300     601     18                1  0.1146   0.7325 0.2069     0.1500  0.3333 0.0626
  headaches     2   GRU      0.1664     601    100                0  0.5188   0.7750 0.4828     0.5676  0.4200 0.1225
     cramps     3   MLP      0.2441     594    145                0  0.7563   0.8873 0.6720     0.5478  0.8690 0.1491
  moodswing     3   MLP      0.2946     594    175                0  0.8134   0.8945 0.6667     0.5345  0.8857 0.1616
    fatigue     3   MLP      0.5185     594    308                0  0.8687   0.8483 0.7530     0.8044  0.7078 0.1736
   bloating     3   MLP      0.3586     594    213                0  0.8288   0.8850 0.7310     0.6496  0.8357 0.1503
sorebreasts     3   MLP      0.1987     594    118                0  0.6322   0.8737 0.5955     0.4817  0.7797 0.1428
  headaches     3   MLP      0.3165     594    188                0  0.7631   0.8481 0.6548     0.5632  0.7819 0.1700
     cramps     3   GRU      0.2441     594    145                0  0.7419   0.8846 0.6527     0.5252  0.8621 0.1537
  moodswing     3   GRU      0.2946     594    175                0  0.7975   0.8997 0.7355     0.6577  0.8343 0.1271
    fatigue     3   GRU      0.5185     594    308                0  0.8935   0.8778 0.7938     0.8513  0.7435 0.1511
   bloating     3   GRU      0.3586     594    213                0  0.8505   0.9032 0.7627     0.7227  0.8075 0.1299
sorebreasts     3   GRU      0.1987     594    118                0  0.7276   0.9048 0.6084     0.4720  0.8559 0.1528
  headaches     3   GRU      0.3165     594    188                0  0.7703   0.8820 0.7075     0.6166  0.8298 0.1467
     cramps     4   MLP      0.0815     601     49                0  0.2759   0.7866 0.3357     0.2553  0.4898 0.1157
  moodswing     4   MLP      0.2329     601    140                0  0.5270   0.7965 0.5382     0.5031  0.5786 0.1596
    fatigue     4   MLP      0.5624     601    338                0  0.8337   0.8059 0.7574     0.8040  0.7160 0.1850
   bloating     4   MLP      0.2280     601    137                0  0.5392   0.7966 0.5524     0.5302  0.5766 0.1470
sorebreasts     4   MLP      0.0449     601     27                1  0.2389   0.9027 0.3614     0.2679  0.5556 0.0646
  headaches     4   MLP      0.2030     601    122                0  0.5820   0.8247 0.5820     0.5820  0.5820 0.1306
     cramps     4   GRU      0.0815     601     49                0  0.3820   0.7931 0.3776     0.2872  0.5510 0.1187
  moodswing     4   GRU      0.2329     601    140                0  0.6182   0.8435 0.5836     0.5394  0.6357 0.1437
    fatigue     4   GRU      0.5624     601    338                0  0.8561   0.8450 0.8094     0.8023  0.8166 0.1623
   bloating     4   GRU      0.2280     601    137                0  0.5437   0.8253 0.5382     0.4939  0.5912 0.1515
sorebreasts     4   GRU      0.0449     601     27                1  0.3059   0.9313 0.4468     0.3134  0.7778 0.0646
  headaches     4   GRU      0.2030     601    122                0  0.5982   0.8564 0.6016     0.5746  0.6311 0.1239
```

Brier by target x model:
```
model        CatBoost     GRU  LogReg     MLP  Persistence_LastObserved
symptom                                                                
bloating       0.1289  0.1375  0.1477  0.1592                    0.1690
cramps         0.1020  0.1329  0.1349  0.1432                    0.1136
fatigue        0.1637  0.1786  0.1827  0.1929                    0.2456
headaches      0.1434  0.1563  0.1636  0.1763                    0.1889
moodswing      0.1412  0.1523  0.1540  0.1702                    0.1915
sorebreasts    0.0768  0.1081  0.1052  0.1212                    0.0728
```
Best (lowest) Brier per target: {"bloating": "CatBoost", "cramps": "CatBoost", "fatigue": "CatBoost", "headaches": "CatBoost", "moodswing": "CatBoost", "sorebreasts": "Persistence_LastObserved"}
- **[MED]** CatBoost is NOT best-Brier on every target. Any sentence claiming so must be removed.

reproducibility_metadata.json keys: cv_scheme, dataset_path, device, feat_dim, inner_cv_scheme, models, n_samples, random_seed, seq_len

## 4. DUA / repository hygiene

- OK   .gitignore mentions `mcphases_data`
- OK   .gitignore mentions `data/processed`
- OK   .gitignore mentions `*.npz`
- OK   .gitignore mentions `*.csv`
- OK   .gitignore mentions `results`
- not a git repo (or .git not at root)