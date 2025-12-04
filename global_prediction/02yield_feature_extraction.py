import os
import rasterio
import numpy as np
import pandas as pd


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

    Parameters:
    - yield_raster_path (str): Path to the high-resolution yield raster.
    - climate_raster_paths (dict): Dictionary where keys are column names and values are paths to climate rasters.
    - fixed_feature_values (dict): Dictionary where keys are column names and values are fixed values to add.
    - final_column_order (list): List of strings defining the final order of columns in the output.
    - output_csv_path (str): Path for the output CSV file.
    """
    print("--- Starting Processing ---")

    # Step 1: Read yield raster and find locations of all valid data points
    print(f"Step 1: Reading yield data from '{os.path.basename(yield_raster_path)}' and generating coordinates...")
    with rasterio.open(yield_raster_path) as src_yield:
        yield_array = src_yield.read(1)
        yield_nodata = src_yield.nodata
        transform = src_yield.transform
        rows, cols = np.where((yield_array != yield_nodata) & (yield_array > 0))

        if rows.size == 0:
            print("Warning: No valid data points found in the yield raster.")
            return

        yield_values = yield_array[rows, cols]
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        print(f"Found {len(xs)} valid yield data points.")

    # Step 2: Store base data in a Pandas DataFrame
    print("Step 2: Creating DataFrame and storing yield and coordinates...")
    df = pd.DataFrame({
        'Maizold': yield_values,
        'longitude': xs,
        'latitude': ys
    })
    coords = list(zip(xs, ys))

    # Step 3: Loop through and extract data from each climate raster
    for col_name, climate_path in climate_raster_paths.items():
        print(f"Step 3: Extracting '{col_name}' data from '{os.path.basename(climate_path)}'...")
        with rasterio.open(climate_path) as src_climate:
            climate_values = [val[0] for val in src_climate.sample(coords)]
            df[col_name] = climate_values

    # Step 4 (New): Add all fixed feature columns
    print("Step 4: Adding fixed feature columns...")
    for col_name, value in fixed_feature_values.items():
        df[col_name] = value
    print("All fixed features added.")

    # Step 5 (New): Reorder columns according to specification
    print("Step 5: Reordering data columns as specified...")
    # Check if all required columns exist in the DataFrame
    all_required_columns = final_column_order + ['Maizold', 'longitude', 'latitude']
    for col in all_required_columns:
        if col not in df.columns:
            raise ValueError(f"Error: Column '{col}' does not exist after processing. Please check configuration.")

    # Append yield and coordinate info to the end of the sorted feature list
    final_ordered_df = df[all_required_columns]

    # Step 6: Save the final DataFrame to a CSV file
    print(f"\nStep 6: Saving all data to CSV file: {output_csv_path}")
    final_ordered_df.to_csv(output_csv_path, index=False)

    print("\n--- Processing Complete! ---")
    print(f"Total of {len(final_ordered_df)} records successfully saved.")
    if len(final_ordered_df) > 1048576:
        print("Note: Data size exceeds Excel row limit. Please use Python (Pandas), R, QGIS, or other professional software to open this CSV.")


# --- Main Program Entry ---
if __name__ == '__main__':
    # --- File Configuration ---
    yield_raster = "spam2020_V2r0_global_Y_RICE_A.tif"

    climate_files = {
        "pre": "2020pre_678mean_prepared_for_yield_area.tif",
        "tmp": "2020tmp_678mean_prepared_for_yield_area.tif"
    }

    # --- New Configuration: Define fixed feature values ---
    fixed_values = {
        "stress": 1,
        "NMs type1": 2,
        "Com.1": 0,
        "Com.2": 363,
        "Size ": 37,
        "Application Dose": 142,
        "Pre-cultivation ": 0,
        "Duration ": 30,
        "Application method": 2,
        "Crop type": 4
    }

    # --- New Configuration: Define final order of feature columns ---
    feature_column_order = [
        "stress",
        "tmp",
        "pre",
        "NMs type1",
        "Com.1",
        "Com.2",
        "Size ",
        "Application Dose",
        "Pre-cultivation ",
        "Duration ",
        "Application method",
        "Crop type"
    ]

    output_csv = "RICE_yield_prediction_features_stress1.csv"

    # --- Pre-run Checks ---
    all_files = [yield_raster] + list(climate_files.values())
    for f in all_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Error: Input file '{f}' not found. Please ensure all files are in the script's directory.")

    # --- Call Core Function (Passing new configurations) ---
    raster_to_points_and_extract(
        yield_raster_path=yield_raster,
        climate_raster_paths=climate_files,
        fixed_feature_values=fixed_values,
        final_column_order=feature_column_order,
        output_csv_path=output_csv
    )