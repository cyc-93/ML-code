import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.transform import Affine


def csv_to_raster(
        template_raster_path,
        input_csv_path,
        value_column_name,
        lon_column_name,  # New parameter: Longitude column name
        lat_column_name,  # New parameter: Latitude column name
        output_raster_path
):
    """
    Converts a CSV file containing geographic coordinates and prediction values 
    into a GeoTIFF raster file.
    """
    print("--- Starting Processing: CSV to Raster ---")

    # Step 1: Read metadata from the template raster
    print(f"Step 1: Reading spatial parameters from template '{os.path.basename(template_raster_path)}'...")
    with rasterio.open(template_raster_path) as src:
        profile = src.profile
        profile['dtype'] = 'float32'
        transform = src.transform
        nodata_val = src.nodata

    # Step 2: Read CSV data
    print(f"Step 2: Reading prediction data from '{os.path.basename(input_csv_path)}'...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {input_csv_path}")
        return

    # Step 3: Check if user-specified required columns exist
    # [Modification] Now checking user-specified column names
    required_cols = [lon_column_name, lat_column_name, value_column_name]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV file must contain the following columns you specified: {required_cols}")
        print(f"Columns actually present in CSV: {list(df.columns)}")
        return

    # [Modification] Extract data using user-specified column names
    lons = df[lon_column_name].values
    lats = df[lat_column_name].values
    values = df[value_column_name].values
    print(f"Read {len(df)} data points.")

    # Step 4: Create an empty raster array
    print("Step 4: Creating an empty raster matching the template...")
    output_array = np.full((profile['height'], profile['width']), nodata_val, dtype=np.float32)

    # Step 5: Fill raster array with values from CSV points
    print("Step 5: Writing point data to raster positions...")
    rows, cols = rasterio.transform.rowcol(transform, lons, lats)
    valid_indices = (
            (np.array(rows) >= 0) & (np.array(rows) < profile['height']) &
            (np.array(cols) >= 0) & (np.array(cols) < profile['width'])
    )
    output_array[rows[valid_indices], cols[valid_indices]] = values[valid_indices]

    # Step 6: Write array to a new GeoTIFF file
    print(f"Step 6: Writing final raster to file: {output_raster_path}")
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(output_array, 1)

    print("\n--- Processing Complete! ---")


# --- Main Program Entry ---
if __name__ == '__main__':
    # --- 1. File Configuration ---
    template_raster = "spam2020_V2r0_global_Y_RICE_A.tif"
    input_csv = "global_predictions_output_rice_stress2.csv"
    output_raster = "predicted_yield_effect_rice_stress2.tif"

    # --- 2. [MODIFY HERE] Specify Column Names in CSV ---

    # Open your CSV, check the header for the longitude column (e.g., column 13), and enter it here
    lon_col = "longitude"  # <--- Enter the actual longitude column name from your CSV here

    # Open your CSV, check the header for the latitude column (e.g., column 14), and enter it here
    lat_col = "latitude"  # <--- Enter the actual latitude column name from your CSV here

    # Confirm the name of your predicted value column
    value_col = "prey"

    # --- Pre-run Checks ---
    if lon_col == "???" or lat_col == "???":
        raise ValueError("Error: Please set the correct longitude/latitude column names (lon_col, lat_col) in the script.")
    if not os.path.exists(template_raster):
        raise FileNotFoundError(f"Error: Template raster '{template_raster}' not found.")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Error: Input CSV file '{input_csv}' not found.")

    # --- Call Core Function (Passing new column name parameters) ---
    csv_to_raster(
        template_raster_path=template_raster,
        input_csv_path=input_csv,
        value_column_name=value_col,
        lon_column_name=lon_col,
        lat_column_name=lat_col,
        output_raster_path=output_raster
    )