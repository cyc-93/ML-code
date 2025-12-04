import pandas as pd
import numpy as np
import os
import re
import sys
import matplotlib

# Set backend to Agg to avoid display issues in non-interactive environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pandas.api.types import is_numeric_dtype

# --- Machine Learning and Statistics Libraries ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler 
import shap
import statsmodels.formula.api as smf

PARAMS_FILE = 'final_best_rf_params_final19.csv'
DATA_FILE = 'mldatafinal19.xlsx'
PREDICTIONS_FILE = 'final_tuned_predictions_rf_final19.xlsx'
SHEET_NAME = 'biomass' 
TARGET_FOLD = 9

print("--> Loading Excel data and best parameters...")

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
    df_raw = pd.read_excel(DATA_FILE, sheet_name='biomass') 
    df_preds = pd.read_excel(PREDICTIONS_FILE, sheet_name='biomass') 
except FileNotFoundError as e:
    print(f"Error: Missing data file: {e.filename}.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}.")
    sys.exit(1)

print(f"--> Reconstructing training and test sets for fold {TARGET_FOLD}...")
X_raw = df_raw.iloc[:, :-1];
y_raw = df_raw.iloc[:, -1]
y = pd.to_numeric(y_raw, errors='coerce');
mask = y.notna()
X_raw = X_raw[mask].reset_index(drop=True);
y = y[mask].reset_index(drop=True)
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
r2 = r2_score(y_test, rf_model.predict(X_test))
print(f"Model evaluation results - R2 Score: {r2:.4f}")

print("--> Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
print("--> Preparing plots...")

SV = shap_values
new_shap_colors = ["#008afb", "#ff0051"]
SHAP_COLOR_MAP = LinearSegmentedColormap.from_list("custom_shap", new_shap_colors)
aesthetic_params = {
    'ax_label_size': 20, 'tick_label_size': 18, 'legend_font_size': 14,
    'spine_linewidth': 1.8, 'trendline_color': '#3b5998', 'trendline_width': 2.5,
    'cbar_label_size': 16
}
plt.rcParams['font.family'] = 'Arial'
out_dir = os.path.join(os.getcwd(), 'outputs_fold9_shap_final_v8')
os.makedirs(out_dir, exist_ok=True)


def find_knee_point_distance(x, y):
    """
    Identifies the knee point (elbow point) of the curve using the maximum distance 
    from the line connecting the start and end points of the curve.
    """
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x.reshape(-1, 1))
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    line_start = np.array([x_scaled[0, 0], y_scaled[0, 0]])
    line_end = np.array([x_scaled[-1, 0], y_scaled[-1, 0]])

    points = np.c_[x_scaled, y_scaled]
    line_vec = line_end - line_start
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

    vec_from_start = points - line_start
    scalar_product = np.sum(vec_from_start * np.tile(line_vec_norm, (len(x), 1)), axis=1)
    vec_from_start_parallel = np.outer(scalar_product, line_vec_norm)

    vec_to_line = vec_from_start - vec_from_start_parallel
    distances = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

    knee_idx = np.argmax(distances)
    return knee_idx


print(f"--> Plotting advanced dependence plots for all {len(X_test.columns)} features...")
thresholds_summary = {}

for feature in X_test.columns:
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        for spine in ax.spines.values(): spine.set_linewidth(aesthetic_params['spine_linewidth'])
        ax.grid(True, which='major', linestyle='--', dashes=(5, 5), alpha=0.6)
        ax.axhline(0, color='grey', linestyle='--', dashes=(8, 5))

        feature_idx = X_test.columns.get_loc(feature)
        notna_mask = X_test[feature].notna()
        x_series = X_test.loc[notna_mask, feature]
        y_data = SV[notna_mask, feature_idx]
        y_test_filtered = y_test.loc[notna_mask]

        if x_series.empty:
            print(f"Warning: Feature '{feature}' has no valid data, skipping.")
            plt.close(fig)
            continue

        scatter_plot = ax.scatter(x_series, y_data, c=y_test_filtered, cmap=SHAP_COLOR_MAP, s=30, alpha=0.7, zorder=4)

        legend_handles = []
        feature_thresholds = {'peak_positive': None, 'knee_point': None}

        # Attempt to fit a smoothing spline if data is numeric and sufficient
        if is_numeric_dtype(x_series) and len(x_series.unique()) > 10:
            try:
                plot_df = pd.DataFrame({'x': x_series, 'y': y_data})
                # Natural cubic spline with 5 degrees of freedom
                fit = smf.ols('y ~ cr(x, df=5)', data=plot_df).fit()
                x_pred = np.linspace(plot_df['x'].min(), plot_df['x'].max(), 500)
                pred_data = fit.get_prediction(exog=dict(x=x_pred)).summary_frame()
                y_pred_mean = pred_data['mean'].values

                ax.fill_between(x_pred, pred_data['mean_ci_lower'], pred_data['mean_ci_upper'],
                                color='lightgrey', alpha=0.4, zorder=0)
                # Highlight positive effect regions
                ax.fill_between(x_pred, y_pred_mean, 0, where=y_pred_mean >= 0, color='#90EE90', alpha=0.3,
                                interpolate=True, zorder=1)
                # Highlight negative effect regions
                ax.fill_between(x_pred, y_pred_mean, 0, where=y_pred_mean < 0, color='#FFB6C1', alpha=0.3,
                                interpolate=True, zorder=1)
                ax.plot(x_pred, y_pred_mean, color=aesthetic_params['trendline_color'],
                        linewidth=aesthetic_params['trendline_width'], zorder=3)

                positive_intervals = np.where(y_pred_mean > 0)[0]

                if len(positive_intervals) > 1:
                    # Find Peak Positive
                    peak_pos_idx_in_pos = np.argmax(y_pred_mean[positive_intervals])
                    peak_pos_idx = positive_intervals[peak_pos_idx_in_pos]
                    peak_pos_x = x_pred[peak_pos_idx]
                    feature_thresholds['peak_positive'] = peak_pos_x
                    ax.axvline(peak_pos_x, color='dimgrey', linestyle='-.', alpha=0.9)

                    # Find Knee Point within positive region
                    x_positive = x_pred[positive_intervals]
                    y_positive = y_pred_mean[positive_intervals]

                    if len(np.unique(y_positive)) > 1:
                        knee_idx_local = find_knee_point_distance(x_positive, y_positive)
                        knee_idx_global = positive_intervals[knee_idx_local]
                        knee_x = x_pred[knee_idx_global]
                        feature_thresholds['knee_point'] = knee_x
                        ax.axvline(knee_x, color='#f0736e', linestyle='--', alpha=0.9, linewidth=2.5)

                legend_handles.append(mpatches.Patch(color='#90EE90', alpha=0.6, label='Positive Effect Region'))
                legend_handles.append(mpatches.Patch(color='#FFB6C1', alpha=0.6, label='Negative Effect Region'))
                if feature_thresholds['peak_positive']:
                    legend_handles.append(Line2D([0], [0], color='dimgrey', lw=2, linestyle='-.',
                                                 label=f"Peak Positive ({feature_thresholds['peak_positive']:.2f})"))
                if feature_thresholds['knee_point']:
                    legend_handles.append(Line2D([0], [0], color='#f0736e', lw=2.5, linestyle='--',
                                                 label=f"Knee Point ({feature_thresholds['knee_point']:.2f})"))

            except Exception as e:
                print(f"Warning: Could not fit curve for feature '{feature}'. Error: {e}")

        if legend_handles:
            ax.legend(handles=legend_handles, loc='best', fontsize=aesthetic_params['legend_font_size'],
                      fancybox=True, framealpha=0.8)

        ax.set_xlabel(f"{feature}", fontsize=aesthetic_params['ax_label_size'])
        ax.set_ylabel("SHAP Value", fontsize=aesthetic_params['ax_label_size'])
        ax.tick_params(axis='both', which='major', labelsize=aesthetic_params['tick_label_size'])

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(scatter_plot, cax=cbar_ax)
        cbar.set_label(f"Target Value ({y_test.name})", size=aesthetic_params['cbar_label_size'], rotation=270,
                       labelpad=20)
        cbar.ax.tick_params(labelsize=aesthetic_params['tick_label_size'])

        safe_feature_name = re.sub(r'[\\/*?:"<>|]', "", feature)
        output_path = os.path.join(out_dir, f'shap_dependence_{safe_feature_name}_fold9.pdf')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        thresholds_summary[feature] = feature_thresholds

    except Exception as e:
        print(f"Could not create dependence plot for feature '{feature}'. Error: {e}")
        plt.close()

print(f"--> All advanced dependence plots successfully saved to '{out_dir}' folder.")

print("\n" + "=" * 70)
print("             Summary of Key Feature Effect Points")
print("=" * 70)
for feature, thresholds in thresholds_summary.items():
    print(f"\nFeature: {feature}")

    if thresholds.get('knee_point') is not None:
        print(f"  - ◊ Knee Point: {thresholds['knee_point']:.2f}")
        print("    (Macro turning point where effect growth slows significantly within positive region)")
    else:
        print("  - ◊ Knee Point: Not found")

    if thresholds.get('peak_positive') is not None:
        print(f"  - ★ Peak Positive Point: {thresholds['peak_positive']:.2f}")
        print("    (Point where positive effect reaches maximum)")
    else:
        print("  - ★ Peak Positive Point: Not found")

print("=" * 70)

print("\n✅ All tasks completed!")