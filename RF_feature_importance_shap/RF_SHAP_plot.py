import pandas as pd
import numpy as np
import os
import sys
import re
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap

print("--> Loading Excel data and best parameters...")

# --- File Paths ---
PARAMS_FILE = 'final_best_rf_params_final19.csv'
DATA_FILE = 'mldatafinal19.xlsx'
PREDICTIONS_FILE = 'final_tuned_predictions_rf_final19.xlsx'
SHEET_NAME_IDENTIFIER = 'biomass'
EXCEL_SHEET_NAME = 'biomass'
TARGET_FOLD = 9

try:
    params_df = pd.read_csv(PARAMS_FILE)
    best_params_series = params_df[(params_df['sheet'] == SHEET_NAME_IDENTIFIER) & (params_df['fold'] == TARGET_FOLD)].iloc[0]
    model_params = best_params_series.drop(['sheet', 'fold']).to_dict()
    model_params['n_estimators'] = int(model_params['n_estimators'])
    model_params['max_depth'] = int(model_params['max_depth'])
    model_params['min_samples_split'] = int(model_params['min_samples_split'])
    model_params['min_samples_leaf'] = int(model_params['min_samples_leaf'])
    print(f"Loaded best parameters for sheet '{SHEET_NAME_IDENTIFIER}', fold {TARGET_FOLD}: \n{model_params}")
except (FileNotFoundError, IndexError) as e:
    print(f"Error: Could not find parameters in '{PARAMS_FILE}'. Ensure the file exists and the 'sheet' column contains '{SHEET_NAME_IDENTIFIER}'. Error: {e}")
    sys.exit(1)

try:
    df_raw = pd.read_excel(DATA_FILE, sheet_name=EXCEL_SHEET_NAME)
    df_preds = pd.read_excel(PREDICTIONS_FILE, sheet_name=EXCEL_SHEET_NAME)
except FileNotFoundError as e:
    print(f"Error: Missing data file: {e.filename}.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}.")
    print(f"Please confirm that '{DATA_FILE}' and '{PREDICTIONS_FILE}' both contain a worksheet named '{EXCEL_SHEET_NAME}'.")
    sys.exit(1)

print(f"--> Reconstructing training and test sets for fold {TARGET_FOLD}...")
X_raw = df_raw.iloc[:, :-1]
y_raw = df_raw.iloc[:, -1]
y = pd.to_numeric(y_raw, errors='coerce')
mask = y.notna()
X_raw = X_raw[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df_preds = df_preds[mask].reset_index(drop=True)
num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()
test_indices = df_preds[df_preds['fold'] == TARGET_FOLD].index
train_indices = df_preds[df_preds['fold'] != TARGET_FOLD].index
X_train_raw, X_test_raw = X_raw.iloc[train_indices], X_raw.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]


def fit_imputers(X_df, num_cols, cat_cols):
    num_fill = X_df[num_cols].median() if num_cols else pd.Series(dtype=float)
    cat_fill = {c: X_df[c].mode().iloc[0] if not X_df[c].mode().empty else 'missing' for c in cat_cols}
    return num_fill, cat_fill


def apply_imputers(X_df, num_fill, cat_fill, num_cols, cat_cols):
    X_out = X_df.copy()
    if num_cols and not num_fill.empty: X_out[num_cols] = X_out[num_cols].fillna(num_fill)
    for c in cat_cols: X_out[c] = X_out[c].fillna(cat_fill.get(c, 'missing')).astype(str)
    return X_out


num_fill, cat_fill = fit_imputers(X_train_raw, num_cols, cat_cols)
X_train_imp = apply_imputers(X_train_raw, num_fill, cat_fill, num_cols, cat_cols)
X_test_imp = apply_imputers(X_test_raw, num_fill, cat_fill, num_cols, cat_cols)
X_train = pd.get_dummies(X_train_imp, columns=cat_cols, dummy_na=False)
X_test = pd.get_dummies(X_test_imp, columns=cat_cols, dummy_na=False)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print(f"Data preparation complete. Train size: {X_train.shape}, Test size: {X_test.shape}")

print("--> Training Random Forest model with best parameters...")
rf_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model evaluation results - R2 Score: {r2:.4f}, RMSE: {rmse_val:.4f}")

print("--> Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

print("--> Preparing plots...")

SV = shap_values
new_shap_colors = ["#008afb", "#ff0051"]
SHAP_COLOR_MAP = LinearSegmentedColormap.from_list("custom_shap", new_shap_colors)
aesthetic_params = {
    'ax_label_size': 16, 'tick_label_size': 14, 'cbar_label_size': 14,
    'summary_cbar_width': 0.01,
    'summary_cbar_height_shrink': 0.8,
    'summary_cbar_pad': 0.02,
    'spine_linewidth': 1.5
}
plt.rcParams['font.family'] = 'Arial'
out_dir = os.path.join(os.getcwd(), 'outputs_fold9_shap')
os.makedirs(out_dir, exist_ok=True)

# --- Plot SHAP Summary Plot ---
print("--> Plotting and saving SHAP Summary Plot...")
mean_abs_shaps = np.abs(SV).mean(axis=0)
feature_importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': mean_abs_shaps}).sort_values(
    'importance', ascending=True)

if not feature_importance_df.empty:
    fig_height = max(8, len(feature_importance_df) * 0.5)
    fig_summary = plt.figure(figsize=(12, fig_height))
    ax_main = fig_summary.add_subplot(111)

    for i, feature_name in enumerate(feature_importance_df['feature']):
        original_idx = X_test.columns.get_loc(feature_name)
        shap_vals_for_feature = SV[:, original_idx]
        vals = X_test.iloc[:, original_idx]
        y_jitter = np.random.normal(0, 0.08, len(shap_vals_for_feature))

        if is_numeric_dtype(vals):
            ax_main.scatter(shap_vals_for_feature, i + y_jitter, c=vals.to_numpy(), cmap=SHAP_COLOR_MAP, s=15,
                            alpha=0.8, zorder=10)
        else:
            vals_cat = vals.astype('category')
            ax_main.scatter(shap_vals_for_feature, i + y_jitter, c=vals_cat.cat.codes.to_numpy(),
                            cmap=plt.get_cmap("tab10"), s=15, alpha=0.9, zorder=10)

    ax_main.set_xlabel("SHAP value (impact on model output)", fontsize=aesthetic_params['ax_label_size'])
    ax_main.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
    ax_main.set_yticks(range(len(feature_importance_df)))
    ax_main.set_yticklabels(feature_importance_df['feature'], fontsize=aesthetic_params['tick_label_size'])
    ax_main.grid(True, axis='x', linestyle='--', alpha=0.35)

    ax_main.spines['top'].set_visible(True)
    ax_main.spines['right'].set_visible(True)
    for spine in ax_main.spines.values():
        spine.set_linewidth(aesthetic_params['spine_linewidth'])

    ax_top = ax_main.twiny()
    ax_top.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color="#cce8fe", alpha=0.6,
                height=0.7, zorder=0)
    ax_top.set_xlabel("Mean Absolute SHAP Value (Global Importance)", fontsize=aesthetic_params['ax_label_size'])
    ax_top.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])

    for spine in ax_top.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(aesthetic_params['spine_linewidth'])
    ax_top.get_yaxis().set_visible(False)

    y_limit = [-0.5, len(feature_importance_df) - 0.5]
    ax_main.set_ylim(y_limit)
    ax_top.set_ylim(y_limit)

    fig_summary.canvas.draw()
    ax_main_pos = ax_main.get_position()
    cax = fig_summary.add_axes([ax_main_pos.x1 + aesthetic_params['summary_cbar_pad'], ax_main_pos.y0 + (
                ax_main_pos.height * (1 - aesthetic_params['summary_cbar_height_shrink']) / 2),
                                aesthetic_params['summary_cbar_width'],
                                ax_main_pos.height * aesthetic_params['summary_cbar_height_shrink']])
    num_matrix = X_test.select_dtypes(include=[np.number]).to_numpy()
    if num_matrix.size > 0:
        vmin, vmax = np.nanmin(num_matrix), np.nanmax(num_matrix)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=SHAP_COLOR_MAP, norm=norm)
        cbar = fig_summary.colorbar(sm, cax=cax)
        cbar.set_label('Feature Value', rotation=270, labelpad=15, fontsize=aesthetic_params['cbar_label_size'])
        cbar.outline.set_visible(False)
        cbar.ax.text(0.5, 1.02, 'High', ha='center', va='bottom', transform=cbar.ax.transAxes, fontsize=12)
        cbar.ax.text(0.5, -0.02, 'Low', ha='center', va='top', transform=cbar.ax.transAxes, fontsize=12)
        cbar.set_ticks([])

    summary_output_path = os.path.join(out_dir, 'rf_shap_summary_plot_fold9_19.pdf')
    plt.savefig(summary_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig_summary)
    print(f"--> SHAP Summary Plot successfully saved to: {summary_output_path}")

print(f"--> Plotting Dependence Plots for all {len(X_test.columns)} features...")
for feature in X_test.columns:
    try:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_test, interaction_index=None, show=False, cmap=SHAP_COLOR_MAP)

        ax_dep = plt.gca()
        for spine in ax_dep.spines.values():
            spine.set_linewidth(aesthetic_params['spine_linewidth'])

        safe_feature_name = re.sub(r'[\\/*?:"<>|]', "", feature)
        output_path = os.path.join(out_dir, f'shap_dependence_{safe_feature_name}_fold9.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not create dependence plot for feature '{feature}'. Error: {e}")
        plt.close()

print(f"--> All dependence plots successfully saved to '{out_dir}' folder.")
print("\n✅ All tasks completed!")