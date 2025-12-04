# -*- coding: utf-8 -*-
"""
Batch Crop Yield Prediction Script (Supports any number of crops x climate scenarios)

This script integrates three processing steps:
1. Extract feature point data from raster files.
2. Predict yields using a pre-trained Random Forest model.
3. Convert prediction results (CSV) back to Raster (GeoTIFF) files.

✅ Automatically calculates total runs based on configuration: len(CROP_CONFIGS) × len(SSP_CONFIGS)
-- Simply add a 4th crop to CROP_CONFIGS to automatically scale to 16 runs.
"""

import os
import sys
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from rasterio.transform import Affine


# =============================================================================
# Step 1: Raster Extraction Function
# =============================================================================

def raster_to_points_and_extract(
        yield_raster_path,
        climate_raster_paths,
        fixed_feature_values,
        final_column_order,
        output_csv_path
):
    """
    Converts a yield raster to points, extracts climate data, adds fixed feature columns,
    and saves the result as a CSV file in a specified order.
    """
    print(f"--- Step 1: Start Feature Extraction -> {os.path.basename(output_csv_path)}")

    # 1.1: Read yield raster and find all valid data points
    print(f"  [1.1] Reading valid yield points from '{os.path.basename(yield_raster_path)}'...")
    try:
        with rasterio.open(yield_raster_path) as src_yield:
            yield_array = src_yield.read(1)
            yield_nodata = src_yield.nodata
            transform = src_yield.transform
            rows, cols = np.where((yield_array != yield_nodata) & (yield_array > 0))

            if rows.size == 0:
                print(f"  [Warning] No valid data points found in '{yield_raster_path}'. Skipping file.")
                return False

            yield_values = yield_array[rows, cols]
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            print(f"  [1.1] Found {len(xs)} valid data points.")
    except rasterio.errors.RasterioIOError as e:
        print(f"  [Error] Unable to read yield raster: {e}")
        return False

    # 1.2: Base Data
    df = pd.DataFrame({
        'Maizold': yield_values,
        'longitude': xs,
        'latitude': ys
    })
    coords = list(zip(xs, ys))

    # 1.3: Loop to extract data from each climate raster
    for col_name, climate_path in climate_raster_paths.items():
        print(f"  [1.3] Extracting '{col_name}' from '{os.path.basename(climate_path)}'...")
        try:
            with rasterio.open(climate_path) as src_climate:
                climate_values = [val[0] for val in src_climate.sample(coords)]
                df[col_name] = climate_values
        except rasterio.errors.RasterioIOError as e:
            print(f"  [Error] Unable to read climate raster: {e}")
            return False

    # 1.4: Add fixed features
    print("  [1.4] Adding fixed feature columns...")
    for col_name, value in fixed_feature_values.items():
        df[col_name] = value

    # 1.5: Column validation and reordering
    all_required_columns = final_column_order + ['Maizold', 'longitude', 'latitude']
    for col in all_required_columns:
        if col not in df.columns:
            print(f"  [Error] Column '{col}' does not exist after processing. Check configuration.")
            return False

    final_ordered_df = df[all_required_columns]

    # 1.6: Save
    print(f"  [1.6] Saving data to: {output_csv_path}")
    final_ordered_df.to_csv(output_csv_path, index=False)
    print(f"--- Step 1: Feature Extraction Complete ---")
    return True


# =============================================================================
# Step 2: Model Training and Prediction Functions
# =============================================================================

def fit_imputers(X_df):
    """Learn median and mode from training data"""
    num_fill = X_df.select_dtypes(include=np.number).median()
    cat_cols = X_df.select_dtypes(exclude=np.number).columns
    cat_fill = {c: X_df[c].mode().iloc[0] if not X_df[c].mode().empty else 'missing' for c in cat_cols}
    return num_fill, cat_fill


def apply_imputers(X_df, num_fill, cat_fill):
    """Apply learned median and mode to new data"""
    X_out = X_df.copy()
    X_out[num_fill.index] = X_out[num_fill.index].fillna(num_fill)
    for c, val in cat_fill.items():
        X_out[c] = X_out[c].fillna(val).astype(str)
    return X_out


def train_model_once(params_file, data_file, predictions_file, sheet_name, target_fold):
    """
    Load data, read best parameters, and train Random Forest model (Runs Only Once).
    """
    print("\n--- Loading data and training main model (Runs Only Once) ---")

    # 2.2.1: Read Best Parameters
    print(f"  [2.1] Loading best parameters from '{params_file}'...")
    try:
        params_df = pd.read_csv(params_file)
        best_params_series_df = params_df.loc[
            (params_df['sheet'] == sheet_name) & (params_df['fold'] == target_fold)
        ]
        if best_params_series_df.empty:
            raise IndexError(f"Combination of sheet='{sheet_name}' and fold={target_fold} not found in {params_file}.")
        best_params_series = best_params_series_df.iloc[0]
        model_params = best_params_series.drop(['sheet', 'fold']).to_dict()
        # Force convert to int
        for k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            if k in model_params:
                model_params[k] = int(model_params[k])
        print(f"  [2.1] Parameters loaded successfully: {model_params}")
    except Exception as e:
        print(f"  [Error] Unable to load parameters: {e}")
        return None

    # 2.2.2: Read Training Data and Fold Info
    print(f"  [2.2] Loading training data from '{data_file}' and '{predictions_file}'...")
    try:
        df_raw = pd.read_excel(data_file, sheet_name=0)
        df_preds = pd.read_excel(predictions_file, sheet_name=0)
    except Exception as e:
        print(f"  [Error] Unable to read Excel training files: {e}")
        return None

    # 2.2.3: Rebuild Training Set
    print(f"  [2.3] Rebuilding training set for fold {target_fold}...")
    X_raw = df_raw.iloc[:, :-1]
    y_raw = df_raw.iloc[:, -1]
    y = pd.to_numeric(y_raw, errors='coerce')
    mask = y.notna()
    X_raw = X_raw[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    df_preds = df_preds[mask].reset_index(drop=True)
    train_indices = df_preds[df_preds['fold'] != target_fold].index
    X_train_raw = X_raw.iloc[train_indices]
    y_train = y.iloc[train_indices]

    # 2.2.4: Fit Preprocessors
    print("  [2.4] Fitting preprocessors (imputers)...")
    num_fill, cat_fill = fit_imputers(X_train_raw)

    # 2.2.5: Encoding
    X_train_imp = apply_imputers(X_train_raw, num_fill, cat_fill)
    X_train = pd.get_dummies(X_train_imp, dummy_na=False)
    print(f"  [2.4] Training data preparation complete. Size: {X_train.shape}")

    # 2.2.6: Train RF
    print("  [2.5] Training Random Forest model...")
    rf_model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("--- Model training complete. Ready for batch prediction ---")

    return rf_model, X_train.columns, num_fill, cat_fill


def predict_with_model(
        rf_model, X_train_columns, num_fill, cat_fill,
        input_features_csv, output_predictions_csv
):
    """
    Use trained model and preprocessors to predict on new global data.
    """
    print(f"--- Step 2: Start Prediction -> {os.path.basename(output_predictions_csv)}")

    # 2.3.1: Load Features
    print(f"  [2.1] Loading feature data: '{input_features_csv}'...")
    try:
        df_global_raw = pd.read_csv(input_features_csv)
    except FileNotFoundError:
        print(f"  [Error] Feature file not found: '{input_features_csv}'")
        return False

    # 2.3.2: Feature Columns (All except last 3)
    X_global_raw = df_global_raw.iloc[:, :-3]
    print("  [2.2] Preprocessing global data using training data rules...")

    # 2.3.3: Imputation
    X_global_imp = apply_imputers(X_global_raw, num_fill, cat_fill)

    # 2.3.4: One-Hot Encoding
    X_global_encoded = pd.get_dummies(X_global_imp, dummy_na=False)

    # 2.3.5: Column Alignment
    print("  [2.3] Aligning feature columns to match model...")
    X_global_aligned = X_global_encoded.reindex(columns=X_train_columns, fill_value=0)

    # 2.3.6: Prediction
    print("  [2.4] Predicting on global data...")
    global_predictions = rf_model.predict(X_global_aligned)

    # 2.3.7: Append and Save
    df_global_raw['prey'] = global_predictions
    df_global_raw.to_csv(output_predictions_csv, index=False)

    print(f"  [2.5] Success! Predictions saved to: '{output_predictions_csv}'")
    print(f"--- Step 2: Prediction Complete ---")
    return True


# =============================================================================
# Step 3: CSV -> Raster
# =============================================================================

def csv_to_raster(
        template_raster_path,
        input_csv_path,
        value_column_name,
        lon_column_name,
        lat_column_name,
        output_raster_path
):
    """
    Converts a CSV file containing geographic coordinates and prediction values 
    into a GeoTIFF raster file.
    """
    print(f"--- Step 3: Start Raster Conversion -> {os.path.basename(output_raster_path)}")

    # 3.1: Template Metadata
    print(f"  [3.1] Reading spatial parameters from template '{os.path.basename(template_raster_path)}'...")
    try:
        with rasterio.open(template_raster_path) as src:
            profile = src.profile
            profile['dtype'] = 'float32'
            transform = src.transform
            nodata_val = src.nodata if src.nodata is not None else -9999.0
            profile['nodata'] = nodata_val
    except rasterio.errors.RasterioIOError as e:
        print(f"  [Error] Unable to read template raster: {e}")
        return False

    # 3.2: Read CSV
    print(f"  [3.2] Reading prediction data from '{os.path.basename(input_csv_path)}'...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"  [Error] CSV file not found: {input_csv_path}")
        return False

    # 3.3: Required Columns
    required_cols = [lon_column_name, lat_column_name, value_column_name]
    if not all(col in df.columns for col in required_cols):
        print(f"  [Error] CSV file must contain the following columns: {required_cols}")
        print(f"  Actual columns in CSV: {list(df.columns)}")
        return False

    lons = df[lon_column_name].values
    lats = df[lat_column_name].values
    values = df[value_column_name].values

    # 3.4: Empty Raster
    print("  [3.4] Creating empty raster matching template...")
    output_array = np.full((profile['height'], profile['width']), nodata_val, dtype=np.float32)

    # 3.5: Write Pixels
    print("  [3.5] Writing point data to raster positions...")
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    valid_indices = (
            (np.array(rows) >= 0) & (np.array(rows) < profile['height']) &
            (np.array(cols) >= 0) & (np.array(cols) < profile['width'])
    )
    output_array[rows[valid_indices], cols[valid_indices]] = values[valid_indices]

    # 3.6: Save File
    print(f"  [3.6] Writing final raster to file: {output_raster_path}")
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(output_array, 1)

    print(f"--- Step 3: Raster Conversion Complete ---")
    return True


# =============================================================================
# --- Main Program Entry: Batch Processing ---
# =============================================================================

if __name__ == '__main__':
    print("====== Starting Batch Processing Workflow ======")

    # --- 1) Model Training Config ---
    MODEL_CONFIG = {
        "PARAMS_FILE": 'final_best_rf_params_final20.csv',
        "DATA_FILE": 'mldatafinal20.xlsx',
        "PREDICTIONS_FILE": 'final_tuned_predictions_rf_final20.xlsx',
        "SHEET_NAME": 'biomass',
        "TARGET_FOLD": 9
    }

    # --- 2) Step 1 Static Config ---
    STATIC_PRE_RASTER = "2020pre_678mean_prepared_for_yield_area.tif"

    # Note: If different crops need different fixed values, override inside the loop
    FIXED_VALUES_TEMPLATE = {
        "stress": 3,
        "NMs type1": 2,
        "Com.1": 0,
        "Com.2": 363,
        "Size ": 37,
        "Application Dose": 142,
        "Pre-cultivation ": 0,
        "Duration ": 30,
        "Application method": 2,
        "Crop type": -1  # Placeholder, will be updated in loop
    }

    FEATURE_COLUMN_ORDER = [
        "stress", "tmp", "pre", "NMs type1", "Com.1", "Com.2",
        "Size ", "Application Dose", "Pre-cultivation ", "Duration ",
        "Application method", "Crop type"
    ]

    RASTER_CONVERSION_COLS = {
        "lon_col": "longitude",
        "lat_col": "latitude",
        "value_col": "prey"
    }

    # --- 3) Batch Loop Config ---
    # ✅ Simply add the 4th crop here to automatically run 16 times
    CROP_CONFIGS = [
        {"name": "WHEA", "type_id": 2, "yield_raster": "spam2020_V2r0_global_Y_WHEA_A.tif"},
        {"name": "MAIZ", "type_id": 3, "yield_raster": "spam2020_V2r0_global_Y_MAIZ_A.tif"},
        {"name": "SOYB", "type_id": 4, "yield_raster": "spam2020_V2r0_global_Y_SOYB_A.tif"},
        {"name": "RICE", "type_id": 1, "yield_raster": "spam2020_V2r0_global_Y_RICE_A.tif"},
    ]

    SSP_CONFIGS = [
        {"name": "ssp126", "raster_file": "UKESM1-0-LL_ssp126_2081-2100_678mean.tif"},
        {"name": "ssp245", "raster_file": "UKESM1-0-LL_ssp245_2081-2100_678mean.tif"},
        {"name": "ssp370", "raster_file": "UKESM1-0-LL_ssp370_2081-2100_678mean.tif"},
        {"name": "ssp585", "raster_file": "UKESM1-0-LL_ssp585_2081-2100_678mean.tif"}
    ]

    # --- 4) Pre-run Checks ---
    print("\n--- Checking all required input files... ---")
    required_files = [STATIC_PRE_RASTER]
    required_files.append(MODEL_CONFIG["PARAMS_FILE"])
    required_files.append(MODEL_CONFIG["DATA_FILE"])
    required_files.append(MODEL_CONFIG["PREDICTIONS_FILE"])
    required_files.extend([c['yield_raster'] for c in CROP_CONFIGS])
    required_files.extend([s['raster_file'] for s in SSP_CONFIGS])

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n[!!! FATAL ERROR !!!] The following required input files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please ensure all files are in the script's directory and try again.")
        sys.exit(1)

    print("--- All input files found. ---")

    # --- 5) Train Model (Runs Once) ---
    training_result = train_model_once(
        MODEL_CONFIG["PARAMS_FILE"],
        MODEL_CONFIG["DATA_FILE"],
        MODEL_CONFIG["PREDICTIONS_FILE"],
        MODEL_CONFIG["SHEET_NAME"],
        MODEL_CONFIG["TARGET_FOLD"]
    )

    if training_result is None:
        print("[!!! FATAL ERROR !!!] Model training failed. Check error logs above.")
        sys.exit(1)

    rf_model, X_train_cols, num_fill, cat_fill = training_result

    # --- 6) Start Batch Loop (Dynamic Total) ---
    total_runs = len(CROP_CONFIGS) * len(SSP_CONFIGS)
    print("\n\n========================================")
    print(f"=== Starting Loop for {total_runs} Combinations... ===")
    print("========================================")

    success_count = 0
    fail_count = 0
    run_idx = 0

    for crop in CROP_CONFIGS:
        for ssp in SSP_CONFIGS:
            run_idx += 1
            run_name = f"{crop['name']}_{ssp['name']}"
            print(f"\n======= Processing: {run_name} ({run_idx}/{total_runs}) =======")

            # 6.1: Define filenames for this run
            step1_output_csv = f"{run_name}_features.csv"
            step2_output_csv = f"{run_name}_predictions.csv"
            step3_output_tif = f"{run_name}_predicted_yield.tif"

            # 6.2: Step 1 Parameters
            current_climate_paths = {
                "pre": STATIC_PRE_RASTER,
                "tmp": ssp['raster_file']
            }
            current_fixed_values = FIXED_VALUES_TEMPLATE.copy()
            current_fixed_values["Crop type"] = crop['type_id']

            try:
                # 6.3: Step 1 (Extraction)
                success_step1 = raster_to_points_and_extract(
                    yield_raster_path=crop['yield_raster'],
                    climate_raster_paths=current_climate_paths,
                    fixed_feature_values=current_fixed_values,
                    final_column_order=FEATURE_COLUMN_ORDER,
                    output_csv_path=step1_output_csv
                )
                if not success_step1:
                    raise Exception("Step 1 Feature Extraction failed or skipped.")

                # 6.4: Step 2 (Prediction)
                success_step2 = predict_with_model(
                    rf_model, X_train_cols, num_fill, cat_fill,
                    input_features_csv=step1_output_csv,
                    output_predictions_csv=step2_output_csv
                )
                if not success_step2:
                    raise Exception("Step 2 Prediction failed.")

                # 6.5: Step 3 (Raster Conversion)
                success_step3 = csv_to_raster(
                    template_raster_path=crop['yield_raster'],
                    input_csv_path=step2_output_csv,
                    value_column_name=RASTER_CONVERSION_COLS["value_col"],
                    lon_column_name=RASTER_CONVERSION_COLS["lon_col"],
                    lat_column_name=RASTER_CONVERSION_COLS["lat_col"],
                    output_raster_path=step3_output_tif
                )
                if not success_step3:
                    raise Exception("Step 3 Raster Conversion failed.")

                print(f"======= ✅ Successfully Completed: {run_name} =======")
                success_count += 1

            except Exception as e:
                print(f"[!!! Error !!!] Issue processing {run_name}: {e}")
                print(f"======= ❌ Failed: {run_name} =======")
                fail_count += 1

    # --- 7) Final Summary ---
    print("\n\n========================================")
    print("====== ✅ All Batch Tasks Completed ======")
    print(f"  Success: {success_count} / {total_runs}")
    print(f"  Failed:  {fail_count} / {total_runs}")
    print("========================================")