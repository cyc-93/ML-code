import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestRegressor

# ======================== Step 1: Load Required Data and Parameters ========================
print("--> Loading Excel data and best parameters...")

# --- File Paths ---
PARAMS_FILE = 'final_best_rf_params_final21.csv'
DATA_FILE = 'mldatafinal21.xlsx'
PREDICTIONS_FILE = 'final_tuned_predictions_rf_final21.xlsx'

SHEET_NAME = 'biomass'
TARGET_FOLD = 9

# --- Read Best Parameters (CSV File) ---
try:
    params_df = pd.read_csv(PARAMS_FILE)
    best_params_series_df = params_df.loc[(params_df['sheet'] == SHEET_NAME) & (params_df['fold'] == TARGET_FOLD)]

    if best_params_series_df.empty:
        raise IndexError(f"Combination of sheet='{SHEET_NAME}' and fold={TARGET_FOLD} not found in {PARAMS_FILE}.")

    best_params_series = best_params_series_df.iloc[0]
    model_params = best_params_series.drop(['sheet', 'fold']).to_dict()
    model_params['n_estimators'] = int(model_params['n_estimators'])
    model_params['max_depth'] = int(model_params['max_depth'])
    model_params['min_samples_split'] = int(model_params['min_samples_split'])
    model_params['min_samples_leaf'] = int(model_params['min_samples_leaf'])
    print(f"Best parameters loaded for sheet '{SHEET_NAME}', fold {TARGET_FOLD}: \n{model_params}")
except (FileNotFoundError, IndexError) as e:
    print(f"Error: Could not find parameters in '{PARAMS_FILE}'. Error: {e}")
    sys.exit(1)

try:
    df_raw = pd.read_excel(DATA_FILE, sheet_name=0)
    df_preds = pd.read_excel(PREDICTIONS_FILE, sheet_name=0)
except FileNotFoundError as e:
    print(f"Error: Missing data file: {e.filename}.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}.")
    sys.exit(1)

# ======================== Step 2: Rebuild Training Set for Model Training and Preprocessing Alignment ========================
print(f"--> Rebuilding training set for fold {TARGET_FOLD}...")
X_raw = df_raw.iloc[:, :-1]
y_raw = df_raw.iloc[:, -1]
y = pd.to_numeric(y_raw, errors='coerce')
mask = y.notna()
X_raw = X_raw[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)
df_preds = df_preds[mask].reset_index(drop=True)
train_indices = df_preds[df_preds['fold'] != TARGET_FOLD].index
X_train_raw = X_raw.iloc[train_indices]
y_train = y.iloc[train_indices]


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
X_train = pd.get_dummies(X_train_imp, dummy_na=False)
print(f"Training data preparation complete. Training set size: {X_train.shape}")

# ======================== Step 3: Train RF Model using Best Parameters ========================
print("--> Training Random Forest model with best parameters...")
rf_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Model training complete. Ready for global prediction.")

# ======================== Step 4: Load and Preprocess Global Data ========================
GLOBAL_DATA_FILE = 'RICE_yield_prediction_features_stress2.csv'
print(f"--> Loading global prediction data: '{GLOBAL_DATA_FILE}'...")

try:
    df_global_raw = pd.read_csv(GLOBAL_DATA_FILE)
except FileNotFoundError:
    print(f"Error: Global data file '{GLOBAL_DATA_FILE}' not found. Please ensure the file exists in the script directory.")
    sys.exit(1)

X_global_raw = df_global_raw.iloc[:, :-3]
print("--> Preprocessing global data using the same rules as training data...")

X_global_imp = apply_imputers(X_global_raw, num_fill, cat_fill)

X_global_encoded = pd.get_dummies(X_global_imp, dummy_na=False)

print("--> Aligning feature columns to match the model...")
X_global_aligned, _ = X_global_encoded.align(X_train, join='right', axis=1, fill_value=0)

# ======================== Step 5: Perform Global Scale Prediction ========================
print("--> Predicting on global data...")
global_predictions = rf_model.predict(X_global_aligned)
print("Prediction complete.")

# ======================== Step 6: Append Predictions to Raw Data and Save ========================
print("--> Saving predictions to a new CSV file...")

df_global_raw['prey'] = global_predictions

OUTPUT_CSV_FILE = 'global_predictions_output_rice_stress2.csv'

df_global_raw.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"--> Success! New file with predictions saved as: '{OUTPUT_CSV_FILE}'")
print("\n✅ All tasks completed!")