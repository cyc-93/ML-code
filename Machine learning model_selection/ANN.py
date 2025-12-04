# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
from sklearn.model_selection import (
    KFold, StratifiedKFold
)
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# Metrics Definitions
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def pearson_r(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return np.corrcoef(yt, yp)[0, 1]

def make_stratify_labels(y, max_bins=10, n_splits=5):
    y = pd.Series(y).astype(float)
    if y.nunique(dropna=True) < 2:
        return None

    ranks = y.rank(method='first')
    q = min(max_bins, int(y.nunique()))
    while q >= 2:
        try:
            bins = pd.qcut(ranks, q=q, labels=False, duplicates='drop')
            counts = np.bincount(bins)
            if counts.min() >= n_splits:
                return bins
        except ValueError:
            pass
        q -= 1
    return None

def fit_imputers(X_df, num_cols, cat_cols):
    num_fill = pd.Series(dtype=float)
    if len(num_cols) > 0:
        num_fill = X_df[num_cols].median()

    cat_fill = {}
    for c in cat_cols:
        mode_vals = X_df[c].mode(dropna=True)
        cat_fill[c] = mode_vals.iloc[0] if mode_vals.shape[0] > 0 else 'missing'
    return num_fill, cat_fill


def apply_imputers(X_df, num_fill, cat_fill, num_cols, cat_cols):
    X_out = X_df.copy()
    if len(num_cols) > 0 and not num_fill.empty:
        X_out[num_cols] = X_out[num_cols].fillna(num_fill)
    for c in cat_cols:
        X_out[c] = X_out[c].fillna(cat_fill.get(c, 'missing')).astype(str)
    return X_out

# Data and Sheets Configuration
xlsx = 'mldatafinal20.xlsx'
try:
    sheet_names = pd.ExcelFile(xlsx).sheet_names
except FileNotFoundError:
    print(f"Error: The file '{xlsx}' was not found. Please ensure it is in the correct directory.")
    exit()

# ANN search space
search_space_ann = {
    'hidden_layer_sizes': Integer(20, 200),
    'activation': Categorical(['relu', 'tanh']),
    'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
    'learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
}

outer_splits = 10
all_params = []

print(f"[INFO] Using scikit-learn MLPRegressor (ANN)")


for s in sheet_names:
    print(f"\nProcessing sheet: {s}...")
    df = pd.read_excel(xlsx, sheet_name=s).replace([np.inf, -np.inf], np.nan)
    X_raw = df.iloc[:, :-1]
    y_raw = df.iloc[:, -1]
    y = pd.to_numeric(y_raw, errors='coerce')
    mask = y.notna()
    X_raw = X_raw[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    if len(y) < 2 * outer_splits:
        print(f"  [SKIP] Sheet '{s}' has too few samples ({len(y)}) to process.")
        continue

    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.columns.difference(num_cols).tolist()

    labels = make_stratify_labels(y, max_bins=10, n_splits=outer_splits)
    if labels is None:
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
        split_iter = outer_cv.split(X_raw)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
        split_iter = outer_cv.split(X_raw, labels)

    oof_pred = np.zeros(len(y))
    oof_fold = np.zeros(len(y), dtype=int)
    fold_rows = []

    for k, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        print(f"  - Fold {k}/{outer_splits}...")
        X_tr_raw, X_te_raw = X_raw.iloc[tr_idx].copy(), X_raw.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        num_fill, cat_fill = fit_imputers(X_tr_raw, num_cols, cat_cols)
        X_tr_imp = apply_imputers(X_tr_raw, num_fill, cat_fill, num_cols, cat_cols)
        X_te_imp = apply_imputers(X_te_raw, num_fill, cat_fill, num_cols, cat_cols)

        X_tr_ohe = pd.get_dummies(X_tr_imp, columns=cat_cols, dummy_na=False)
        X_te_ohe = pd.get_dummies(X_te_imp, columns=cat_cols, dummy_na=False)

        X_tr_aligned, X_te_aligned = X_tr_ohe.align(X_te_ohe, join='left', axis=1, fill_value=0)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_aligned)
        X_te = scaler.transform(X_te_aligned)

        base = MLPRegressor(
            random_state=42,
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=10
        )

        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

        opt = BayesSearchCV(
            estimator=base,
            search_spaces=search_space_ann,
            n_iter=50,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            refit=True,
            return_train_score=False
        )

        opt.fit(X_tr, y_tr)
        final_model = opt.best_estimator_
        pred_te = final_model.predict(X_te)
        oof_pred[te_idx] = pred_te
        oof_fold[te_idx] = k

        fold_rows.append({
            'sheet': s, 'fold': k,
            'rmse': rmse(y_te, pred_te),
            'r2': r2_score(y_te, pred_te),
            'r': pearson_r(y_te, pred_te),
        })

        params_row = {'sheet': s, 'fold': k}
        best_params_log = {}
        for param, value in opt.best_params_.items():
            if isinstance(value, np.integer):
                best_params_log[param] = int(value)
            else:
                best_params_log[param] = value
        params_row.update(best_params_log)
        all_params.append(params_row)

    fold_df = pd.DataFrame(fold_rows)
    summary = fold_df[['rmse', 'r2', 'r']].agg(['mean', 'std']).rename_axis('agg').reset_index()
    fold_out = pd.concat([fold_df, summary], ignore_index=True)

    metric_filename = f'final_tuned_metrics_ann_{s}.csv'
    preds_filename = f'final_tuned_predictions_ann_{s}.csv'

    fold_out.to_csv(metric_filename, index=False)

    preds_df = pd.DataFrame({'obs': y.values, 'oof_pred': oof_pred, 'fold': oof_fold})
    preds_df.to_csv(preds_filename, index=False)

    print(f"  [SUCCESS] Results for sheet '{s}' saved to {metric_filename} and {preds_filename}")

pd.DataFrame(all_params).to_csv('final_best_ann_params_final20.csv', index=False)

print("\n✅ Script finished successfully!")