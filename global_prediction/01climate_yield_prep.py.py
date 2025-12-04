import os
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from osgeo import gdal


def prepare_climate_for_yield_area(
        reference_raster_path,
        climate_raster_path,
        output_raster_path
):
    """
    Precisely prepares a climate raster for point sampling.
    This function aligns the climate raster with the yield raster and interpolates 
    missing values in the climate raster only in areas where valid data exists 
    in the yield raster.

    Parameters:
    - reference_raster_path (str): High-resolution yield raster used to define extent, CRS, and valid data areas.
    - climate_raster_path (str): Low-resolution climate raster to be processed.
    - output_raster_path (str): Path for the final processed climate raster.
    """
    print(f"--- Starting processing for file: {os.path.basename(climate_raster_path)} ---")

    # Step 1: Align climate raster to yield raster's CRS and extent (keeping low resolution)
    print("Step 1: Aligning climate raster to yield raster's CRS and extent...")
    temp_aligned_climate_path = "temp_aligned_climate.tif"

    ref_ds = gdal.Open(reference_raster_path)
    target_crs = ref_ds.GetProjection()
    gt = ref_ds.GetGeoTransform()
    target_extent = [gt[0], gt[3] + gt[5] * ref_ds.RasterYSize, gt[0] + gt[1] * ref_ds.RasterXSize, gt[3]]
    ref_ds = None

    gdal.Warp(
        temp_aligned_climate_path,
        climate_raster_path,
        format='GTiff',
        dstSRS=target_crs,
        outputBounds=target_extent,
        resampleAlg='bilinear',
        creationOptions=['COMPRESS=LZW']
    )

    # Step 2: Create a yield mask with the same dimensions as the aligned climate raster
    print("Step 2: Creating yield data mask aligned with climate raster...")
    temp_yield_mask_path = "temp_yield_mask.tif"

    with rasterio.open(temp_aligned_climate_path) as aligned_climate:
        # Use aligned climate raster attributes as a template to downsample the yield raster
        gdal.Warp(
            temp_yield_mask_path,
            reference_raster_path,
            format='GTiff',
            width=aligned_climate.width,
            height=aligned_climate.height,
            outputBounds=aligned_climate.bounds,
            dstSRS=aligned_climate.crs.to_wkt(),
            resampleAlg='average'  # Use average resampling; as long as there is yield data in the area, the downsampled value will be > 0
        )

    # Step 3: Perform conditional filling
    print("Step 3: Performing conditional interpolation filling...")
    with rasterio.open(temp_aligned_climate_path) as src_climate:
        # Read aligned raw climate data
        climate_array = src_climate.read(1)
        climate_nodata = src_climate.nodata
        profile = src_climate.profile  # Save profile for output

        # Read downsampled yield mask
        with rasterio.open(temp_yield_mask_path) as src_mask:
            mask_array = src_mask.read(1)
            mask_nodata = src_mask.nodata

        # Create a "fallback" fully filled version
        # First create a boolean mask, True represents valid data
        valid_data_mask = climate_array != climate_nodata
        # Use rasterio.fillnodata for efficient interpolation
        filled_climate_array = fillnodata(climate_array, mask=valid_data_mask)

        # Define condition:
        # 1. Original climate data is NoData
        # 2. Yield mask must have data (handle mask's own NoData case)
        if mask_nodata is not None:
            condition_to_fill = (climate_array == climate_nodata) & (mask_array != mask_nodata) & (mask_array > 0)
        else:
            condition_to_fill = (climate_array == climate_nodata) & (mask_array > 0)

        # Use numpy.where to synthesize the final raster based on the condition
        # If condition is true, use "fallback" filled value; otherwise, use original value
        final_array = np.where(condition_to_fill, filled_climate_array, climate_array)

    # Step 4: Save final result
    print("Step 4: Saving final result...")
    with rasterio.open(output_raster_path, 'w', **profile) as dst:
        dst.write(final_array, 1)

    # Step 5: Clean up temporary files
    print("Step 5: Cleaning up temporary files...")
    os.remove(temp_aligned_climate_path)
    os.remove(temp_yield_mask_path)

    print(f"--- Processing complete! Precisely prepared climate raster saved to: {output_raster_path} ---\n")


# --- Main Program Entry ---
if __name__ == '__main__':
    reference_raster = "spam2020_V2r0_global_Y_WHEA_A.tif"
    climate_rasters_to_process = [
        "2020pre_678mean.tif",
        "2020tmp_678mean.tif"
    ]

    if not os.path.exists(reference_raster):
        raise FileNotFoundError(f"Error: Reference raster '{reference_raster}' not found.")
    for raster_file in climate_rasters_to_process:
        if not os.path.exists(raster_file):
            raise FileNotFoundError(f"Error: Climate raster '{raster_file}' not found.")

    print("All input files found, starting processing workflow...\n")

    for climate_raster in climate_rasters_to_process:
        base_name, extension = os.path.splitext(climate_raster)
        output_raster = f"{base_name}_prepared_for_yield_area{extension}"

        prepare_climate_for_yield_area(
            reference_raster_path=reference_raster,
            climate_raster_path=climate_raster,
            output_raster_path=output_raster
        )

    print("All files processed!")