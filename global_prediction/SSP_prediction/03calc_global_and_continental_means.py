# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

# ================= 0) Basic Configuration =================
PIXEL_SCALE = 1.0  # Since input files are already multiplied by 0.542, keep this as 1.0 to avoid double scaling

# Continent Vector File
shp_path   = r"E:/2025论文/cyn/world_region.shp"
name_field = "REGION_ADJ"   # Change this to your actual continent name field
continent_order = [
    "Africa", "Asia", "Australia", "Europe", "Latin America", "North America"
]

# 16 TIF Paths (4 Species x 4 Scenarios) -- Modify according to your actual paths
tif_map = {
    "MAIZ": {
        "SSP126": r"E:/2025论文/cyn/ssp预测/MAIZ_ssp126_multiplied_0.542.tif",
        "SSP245": r"E:/2025论文/cyn/ssp预测/MAIZ_ssp245_multiplied_0.542.tif",
        "SSP370": r"E:/2025论文/cyn/ssp预测/MAIZ_ssp370_multiplied_0.542.tif",
        "SSP585": r"E:/2025论文/cyn/ssp预测/MAIZ_ssp585_multiplied_0.542.tif",
    },
    "WHEA": {
        "SSP126": r"E:/2025论文/cyn/ssp预测/WHEA_ssp126_multiplied_0.542.tif",
        "SSP245": r"E:/2025论文/cyn/ssp预测/WHEA_ssp245_multiplied_0.542.tif",
        "SSP370": r"E:/2025论文/cyn/ssp预测/WHEA_ssp370_multiplied_0.542.tif",
        "SSP585": r"E:/2025论文/cyn/ssp预测/WHEA_ssp585_multiplied_0.542.tif",
    },
    "SOYB": {
        "SSP126": r"E:/2025论文/cyn/ssp预测/SOYB_ssp126_multiplied_0.542.tif",
        "SSP245": r"E:/2025论文/cyn/ssp预测/SOYB_ssp245_multiplied_0.542.tif",
        "SSP370": r"E:/2025论文/cyn/ssp预测/SOYB_ssp370_multiplied_0.542.tif",
        "SSP585": r"E:/2025论文/cyn/ssp预测/SOYB_ssp585_multiplied_0.542.tif",
    },
    "RICE": {
        "SSP126": r"E:/2025论文/cyn/ssp预测 - 4/RICE_ssp126_multiplied_0.542.tif",
        "SSP245": r"E:/2025论文/cyn/ssp预测 - 4/RICE_ssp245_multiplied_0.542.tif",
        "SSP370": r"E:/2025论文/cyn/ssp预测 - 4/RICE_ssp370_multiplied_0.542.tif",
        "SSP585": r"E:/2025论文/cyn/ssp预测 - 4/RICE_ssp585_multiplied_0.542.tif",
    },
}

# Output Files
out_overall_csv   = "global_means_16rasters.csv"
out_continent_csv = "continent_means_16rasters.csv"

# ============== 1) Helper Functions ==============
def round5(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return np.nan
    return round(float(x), 5)

def ensure_name_field(gdf: gpd.GeoDataFrame, name_field: str) -> str:
    """Checks if the name field exists, otherwise guesses."""
    if name_field in gdf.columns:
        return name_field
    cand = [c for c in gdf.columns if c.lower() in ("name","continent","region","admin","area")]
    if cand:
        return cand[0]
    non_geom = [c for c in gdf.columns if c != gdf.geometry.name]
    if not non_geom:
        raise ValueError("SHP contains no suitable attribute fields to use as names.")
    return non_geom[0]

def global_mean_of_tif(tif_path: str) -> float:
    """Calculates the global mean of valid pixels in a TIF."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[np.isinf(arr)] = np.nan
        if PIXEL_SCALE != 1.0:
            arr *= float(PIXEL_SCALE)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

def zonal_means_by_continent(tif_path: str, gdf_diss: gpd.GeoDataFrame, name_field: str) -> pd.DataFrame:
    """Calculates mean values for each continent."""
    rows = []
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        for _, r in gdf_diss.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            data, _ = mask(src, [mapping(geom)], crop=True, nodata=nodata)
            a = data[0].astype("float64")
            if nodata is not None:
                a[a == nodata] = np.nan
            a[np.isinf(a)] = np.nan
            if PIXEL_SCALE != 1.0:
                a *= float(PIXEL_SCALE)
            mean = np.nan if np.isnan(a).all() else np.nanmean(a)
            rows.append({"continent": str(r[name_field]), "mean": mean})
    return pd.DataFrame(rows)

# ============== 2) Read Shapefile & Align CRS ==============
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"Shapefile not found: {shp_path}")
gdf = gpd.read_file(shp_path)
name_field = ensure_name_field(gdf, name_field)

# Find one existing TIF to determine target CRS
first_tif = None
for sp, scen in tif_map.items():
    for sc, p in scen.items():
        if os.path.exists(p):
            first_tif = p
            break
    if first_tif:
        break
if first_tif is None:
    raise FileNotFoundError("No valid TIF paths found in tif_map. Please check file paths.")

with rasterio.open(first_tif) as src0:
    rcrs = src0.crs
if gdf.crs != rcrs:
    gdf = gdf.to_crs(rcrs)

gdf[name_field] = gdf[name_field].astype(str)
gdf["geometry"] = gdf.geometry.buffer(0)  # Fix invalid geometries
gdf_diss = gdf.dissolve(by=name_field, as_index=False)

# ============== 3) Calculate Means ==============
overall_rows = []
continent_rows = []

species_order = list(tif_map.keys())
scenario_order = ["SSP126", "SSP245", "SSP370", "SSP585"]

print("Starting statistics calculation...")

for sp in species_order:
    scen_paths = tif_map[sp]
    scenarios = [s for s in scenario_order if s in scen_paths]
    for sc in scenarios:
        tif_path = scen_paths[sc]
        if not os.path.exists(tif_path):
            print(f"[Warning] File missing: {tif_path}")
            continue
        
        print(f"Processing: {sp} - {sc}")

        # Global Mean
        gmean = global_mean_of_tif(tif_path)
        overall_rows.append({
            "species": sp,
            "scenario": sc,
            "global_mean": round5(gmean)
        })

        # Continent Mean
        df_cont = zonal_means_by_continent(tif_path, gdf_diss, name_field)
        for cont in continent_order:
            # Safely extract mean for specific continent
            v = df_cont.loc[df_cont["continent"] == cont, "mean"]
            mean_val = round5(v.values[0]) if not v.empty else np.nan
            continent_rows.append({
                "species": sp,
                "scenario": sc,
                "continent": cont,
                "mean": mean_val
            })

# ============== 4) Save to CSV ==============
df_overall   = pd.DataFrame(overall_rows)[["species", "scenario", "global_mean"]]
df_continent = pd.DataFrame(continent_rows)[["species", "scenario", "continent", "mean"]]

df_overall.to_csv(out_overall_csv, index=False, encoding="utf-8-sig")
df_continent.to_csv(out_continent_csv, index=False, encoding="utf-8-sig")

print(f"\n[Done] Files saved:\n  1. {out_overall_csv}\n  2. {out_continent_csv}")