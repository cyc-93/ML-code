# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from copy import deepcopy
import warnings
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# Metrics Definitions
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def pearson_r(y_true, y_pred):
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if np.std(yt) == 0 or np.std(yp) == 0:
        return np.nan
    return np.corrcoef(yt, yp)[0, 1]

# Helper Functions: Stratification & Imputation
def make_stratify_labels(y, max_bins=10, n_splits=10):
    y = pd.Series(y).astype(float)
    if y.nunique(dropna=True) < 2:
        return None  # Almost constant

    ranks = y.rank(method='first')  # Break ties
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
xlsx = 'mldatafinal21.xlsx'
try:
    sheet_names = pd.ExcelFile(xlsx).sheet_names  # Slice if you only want to run the first N sheets
except FileNotFoundError:
    print(f"Error: The file '{xlsx}' was not found. Please ensure it is in the correct directory.")
    exit()

# Search Space for Random Forest
search_space_rf = {
    'n_estimators': Integer(100, 5000),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Real(0.1, 1.0, prior='uniform'),
}

outer_splits = 10
all_params = []

print(f"[INFO] Using scikit-learn RandomForestRegressor")

with pd.ExcelWriter('final_tuned_metrics_rf_final21.xlsx', engine='openpyxl') as metrics_writer, \
        pd.ExcelWriter('final_tuned_predictions_rf_final21.xlsx', engine='openpyxl') as preds_writer:
    pd.DataFrame({'status': ['init']}).to_excel(metrics_writer, sheet_name='__init__', index=False)
    pd.DataFrame({'status': ['init']}).to_excel(preds_writer, sheet_name='__init__', index=False)
    wrote_any = False

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

        #Outer CV: Stratified or KFold
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

        # Outer fold loop
        for k, (tr_idx, te_idx) in enumerate(split_iter, start=1):
            print(f"  - Fold {k}/{outer_splits}...")
            X_tr_raw, X_te_raw = X_raw.iloc[tr_idx].copy(), X_raw.iloc[te_idx].copy()
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            # Fit imputer within the fold to avoid leakage
            num_fill, cat_fill = fit_imputers(X_tr_raw, num_cols, cat_cols)
            X_tr_imp = apply_imputers(X_tr_raw, num_fill, cat_fill, num_cols, cat_cols)
            X_te_imp = apply_imputers(X_te_raw, num_fill, cat_fill, num_cols, cat_cols)

            X_tr = pd.get_dummies(X_tr_imp, columns=cat_cols, dummy_na=False)
            X_te = pd.get_dummies(X_te_imp, columns=cat_cols, dummy_na=False)

            X_tr, X_te = X_tr.align(X_te, join='left', axis=1, fill_value=0)

            base = RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )

            inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

            opt = BayesSearchCV(
                estimator=base,
                search_spaces=search_space_rf,
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

            # Predict on the outer test fold
            pred_te = final_model.predict(X_te)
            oof_pred[te_idx] = pred_te
            oof_fold[te_idx] = k

            best_n_estimators = opt.best_params_['n_estimators']

            fold_rows.append({
                'sheet': s,
                'fold': k,
                'rmse': rmse(y_te, pred_te),
                'r2': r2_score(y_te, pred_te),
                'r': pearson_r(y_te, pred_te),
                'best_n_estimators': best_n_estimators
            })

            params_row = {'sheet': s, 'fold': k}
            params_row.update(opt.best_params_)
            all_params.append(params_row)

        fold_df = pd.DataFrame(fold_rows)
        summary = fold_df[['rmse', 'r2', 'r']].agg(['mean', 'std']).rename_axis('agg').reset_index()
        fold_out = pd.concat([fold_df, summary], ignore_index=True)
        fold_out.to_excel(metrics_writer, sheet_name=s, index=False)

        preds_df = pd.DataFrame({'obs': y.values, 'oof_pred': oof_pred, 'fold': oof_fold})
        preds_df.to_excel(preds_writer, sheet_name=s, index=False)

        wrote_any = True

    if wrote_any and '__init__' in metrics_writer.book.sheetnames:
        ws = metrics_writer.book['__init__']
        metrics_writer.book.remove(ws)
    if wrote_any and '__init__' in preds_writer.book.sheetnames:
        ws = preds_writer.book['__init__']
        preds_writer.book.remove(ws)

pd.DataFrame(all_params).to_csv('final_best_rf_params_final21.csv', index=False)

print("\n✅ Script finished successfully!")