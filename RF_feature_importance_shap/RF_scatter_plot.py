import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("--> Loading Excel data and best parameters...")

PARAMS_FILE = 'final_best_rf_params_final19.csv'
DATA_FILE = 'mldatafinal19.xlsx'
PREDICTIONS_FILE = 'final_tuned_predictions_rf_final19.xlsx'
SHEET_NAME = 'biomass'
TARGET_FOLD = 9

try:
    params_df = pd.read_csv(PARAMS_FILE)
    best_params_series = params_df[(params_df['sheet'] == SHEET_NAME) & (params_df['fold'] == TARGET_FOLD)].iloc[0]
    model_params = best_params_series.drop(['sheet', 'fold']).to_dict()
    model_params['n_estimators'] = int(model_params['n_estimators'])
    model_params['max_depth'] = int(model_params['max_depth'])
    model_params['min_samples_split'] = int(model_params['min_samples_split'])
    model_params['min_samples_leaf'] = int(model_params['min_samples_leaf'])
    print(f"Loaded best parameters for sheet '{SHEET_NAME}', fold {TARGET_FOLD}: \n{model_params}")
except (FileNotFoundError, IndexError) as e:
    print(f"Error: Could not find parameters in '{PARAMS_FILE}'. Error: {e}")
    sys.exit(1)

try:
    df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    df_preds = pd.read_excel(PREDICTIONS_FILE, sheet_name=SHEET_NAME)
except FileNotFoundError as e:
    print(f"Error: Missing data file: {e.filename}.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}.")
    sys.exit(1)

print(f"--> Reconstructing training and test sets for fold {TARGET_FOLD}...")
X_raw = df_raw.iloc[:, :-1]; y_raw = df_raw.iloc[:, -1]
y = pd.to_numeric(y_raw, errors='coerce'); mask = y.notna()
X_raw = X_raw[mask].reset_index(drop=True); y = y[mask].reset_index(drop=True)
df_preds = df_preds[mask].reset_index(drop=True)
test_indices = df_preds[df_preds['fold'] == TARGET_FOLD].index
train_indices = df_preds[df_preds['fold'] != TARGET_FOLD].index
X_train_raw, X_test_raw = X_raw.iloc[train_indices], X_raw.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

def fit_imputers(X_df):
    num_fill = X_df.select_dtypes(include=np.number).median()
    cat_cols = X_df.select_dtypes(exclude=np.number).columns
    cat_fill = {c: X_df[c].mode().iloc[0] if not X_df[c].mode().empty else 'missing' for c in cat_cols}
    return num_fill, cat_fill

def apply_imputers(X_df, num_fill, cat_fill):
    X_out = X_df.copy()
    X_out[num_fill.index] = X_out[num_fill.index].fillna(num_fill)
    for c, val in cat_fill.items(): X_out[c] = X_out[c].fillna(val).astype(str)
    return X_out

num_fill, cat_fill = fit_imputers(X_train_raw)
X_train_imp = apply_imputers(X_train_raw, num_fill, cat_fill)
X_test_imp = apply_imputers(X_test_raw, num_fill, cat_fill)
X_train = pd.get_dummies(X_train_imp, dummy_na=False)
X_test = pd.get_dummies(X_test_imp, dummy_na=False)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print(f"Data preparation complete. Train size: {X_train.shape}, Test size: {X_test.shape}")

print("--> Training Random Forest model with best parameters...")
rf_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training completed.")
print("--> Generating model performance scatter plot...")

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(y_train, y_pred_train, c='#1f77b4', alpha=0.5, edgecolors='w', s=50, label='Train Data')
ax.scatter(y_test, y_pred_test, c='#d62728', alpha=0.7, edgecolors='w', s=50, label='Test Data')

all_values = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
min_val, max_val = all_values.min() - 0.5, all_values.max() + 0.5
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')

m, b = np.polyfit(y_test, y_pred_test, 1)
ax.plot(y_test, m * y_test + b, color='darkred', linewidth=2.5, label='Test Fit Line')

train_text = (f'Train Metrics\n'
              f'$R^2$: {r2_train:.3f}\n'
              f'RMSE: {rmse_train:.3f}')
ax.text(0.05, 0.95, train_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', color='#1f77b4',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='none'))

test_text = (f'Test Metrics\n'
             f'$R^2$: {r2_test:.3f}\n'
             f'RMSE: {rmse_test:.3f}')
ax.text(0.95, 0.05, test_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', horizontalalignment='right', color='#d62728',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7, ec='none'))

ax.set_xlabel('True Value', fontsize=16)
ax.set_ylabel('Predicted Value', fontsize=16)
ax.set_title('Random Forest Performance (Fold 9)', fontsize=18, weight='bold')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.6)

output_dir = 'outputs_performance_plot'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'rf_performance_scatter_fold9.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"--> Model performance scatter plot successfully saved to: {output_path}")
print("\n✅ All tasks completed!")