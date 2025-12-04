# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

# =============== 0) Global and General Configuration ===============
# Enable editable text in AI (Adobe Illustrator)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42

# Input TIFs are already *_multiplied_0.542.tif, so NO further scaling needed here.
PIXEL_SCALE = 1.0

# Display as percentage (Multiply 0-1 values by 100)
SCALE_HINT = 100

# =============== 1) Required: Data Paths ===============
# Continent Vector File
shp_path   = r"E:/2025论文/cyn/world_region.shp"
name_field = "REGION_ADJ"  # Adjust this to match your continent name field

# Species and Scenario TIFs (4 Species x 4 Scenarios)
# Note: RICE paths use the specific folder "ssp预测 - 4" as provided.
tif_map = {
    "MAIZE": {
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

# —— Output Files ——
out_png      = "continent_means_linepanel.png"
out_pdf      = "continent_means_linepanel.pdf"
out_csv_long = "continent_means_long_by_scenario.csv"
out_csv_wide = "continent_means_wide_last_scenario.csv"  # Only for reference (last scenario)

# =============== 2) Styles (Symbols + Colors) ===============
continent_order = [
    "Africa", "Asia", "Australia", "Europe", "Latin America", "North America"
]
continent_style = {
    "Africa":         dict(color="#2a7fe5", marker="^"),
    "Asia":           dict(color="#6c757d", marker="s"),
    "Australia":      dict(color="#9b59b6", marker="D"),
    "Europe":         dict(color="#e74c3c", marker="o"),
    "Latin America":  dict(color="#2ecc71", marker="v"),
    "North America":  dict(color="#f1c40f", marker=">"),
}

# =============== 3) Helper Functions ===============
def ensure_name_field(gdf: gpd.GeoDataFrame, name_field: str) -> str:
    """Checks if the specified name field exists, otherwise tries to guess."""
    if name_field in gdf.columns:
        return name_field
    cand = [c for c in gdf.columns if c.lower() in ("name","continent","region","admin","area")]
    if cand:
        print(f"[Info] Field '{name_field}' not found. Using: {cand[0]}")
        return cand[0]
    non_geom = [c for c in gdf.columns if c != gdf.geometry.name]
    if not non_geom:
        raise ValueError("SHP contains no suitable attribute fields to use as names.")
    print(f"[Info] Field '{name_field}' not found. Using: {non_geom[0]}")
    return non_geom[0]

def zonal_mean_for_one(tif_path: str, gdf_diss: gpd.GeoDataFrame, name_field: str) -> pd.DataFrame:
    """Calculates zonal mean for a single TIF by continent (Pixels multiplied by PIXEL_SCALE)."""
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"File not found: {tif_path}")
    rows = []
    with rasterio.open(tif_path) as src:
        raster_nodata = src.nodata
        for _, r in gdf_diss.iterrows():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            data, _ = mask(src, [mapping(geom)], crop=True, nodata=raster_nodata)
            a = data[0].astype("float64")
            if raster_nodata is not None:
                a[a == raster_nodata] = np.nan
            a[np.isinf(a)] = np.nan
            a *= PIXEL_SCALE
            mean = np.nan if np.isnan(a).all() else np.nanmean(a)
            rows.append({"continent": str(r[name_field]), "mean_raw": mean})
    return pd.DataFrame(rows)

def apply_scale(df_raw: pd.DataFrame, scale_hint):
    """Scales data to percentage if needed; SCALE_HINT=100 -> *100"""
    if scale_hint is None:
        vals = df_raw["mean_raw"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        # Heuristic: if most values are small (<= 1.2), assume ratio and convert to %
        use_percent = (finite.size>0 and np.mean((finite>=0)&(finite<=1.2))>=0.6)
        scale = 100.0 if use_percent else 1.0
    else:
        scale = float(scale_hint)
    df = df_raw.copy()
    df["mean"] = df["mean_raw"] * scale
    return df, (scale==100.0)

# =============== 4) Main Statistical Workflow ===============
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"Shapefile not found: {shp_path}")
gdf = gpd.read_file(shp_path)
name_field = ensure_name_field(gdf, name_field)

# Find at least one existing TIF to align CRS
first_tif = None
for sp, scen_map in tif_map.items():
    for sc, p in scen_map.items():
        if os.path.exists(p):
            first_tif = p
            break
    if first_tif:
        break
if first_tif is None:
    raise FileNotFoundError("No existing TIF files found in tif_map. Please check file paths.")

with rasterio.open(first_tif) as src0:
    raster_crs = src0.crs

if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)
gdf[name_field] = gdf[name_field].astype(str)
gdf["geometry"] = gdf.geometry.buffer(0)   # Fix invalid geometries
gdf_diss = gdf.dissolve(by=name_field, as_index=False)

# Statistics: Generate Long Format DataFrame
print("Calculating zonal statistics...")
records = []
for species, scen_paths in tif_map.items():
    order_pref = ["SSP126","SSP245","SSP370","SSP585"]
    scenarios = [s for s in order_pref if s in scen_paths]
    for sc in scenarios:
        tif_path = scen_paths[sc]
        df_raw = zonal_mean_for_one(tif_path, gdf_diss, name_field)
        df_scaled, as_percent = apply_scale(df_raw, SCALE_HINT)
        for _, r in df_scaled.iterrows():
            records.append({
                "species": species,
                "scenario": sc,
                "continent": r["continent"],
                "mean": r["mean"],
                "as_percent": as_percent
            })

df_long = pd.DataFrame.from_records(records)
df_long.to_csv(out_csv_long, index=False, encoding="utf-8-sig")
print(f"[Done] Exported Long CSV: {out_csv_long}")

# Wide Format (For reference: taking the last scenario per species column)
last_scen_per_species = df_long.groupby("species")["scenario"].apply(lambda s: s.iloc[-1]).to_dict()
df_wide = (
    df_long[df_long.apply(lambda r: last_scen_per_species[r["species"]]==r["scenario"], axis=1)]
    .pivot_table(index="continent", columns="species", values="mean")
    .reset_index()
)
df_wide.to_csv(out_csv_wide, index=False, encoding="utf-8-sig")

# =============== 5) Plotting Line Panels ===============
print("Generating line plots...")
species_list = list(tif_map.keys())
n_col = len(species_list)
fig_w = max(10, 4.0 * n_col)
fig, axes = plt.subplots(1, n_col, figsize=(fig_w, 4.2), dpi=220, sharey=False)
if n_col == 1:
    axes = [axes]

for ax, sp in zip(axes, species_list):
    sub = df_long[df_long["species"] == sp].copy()
    x_labels = ["SSP126","SSP245","SSP370","SSP585"]
    sub["scenario"] = pd.Categorical(sub["scenario"], categories=x_labels, ordered=True)
    
    # Determine if percentage formatting is used for this species
    use_percent = bool(sub["as_percent"].mode().iloc[0]) if not sub.empty else False

    for cont in continent_order:
        dat = sub[sub["continent"] == cont].sort_values("scenario")
        if dat.empty:
            continue
        style = continent_style.get(cont, dict(color="C0", marker="o"))
        ax.plot(
            dat["scenario"].cat.codes, dat["mean"].to_numpy(dtype=float),
            linewidth=2.0, marker=style["marker"], markersize=5.5,
            color=style["color"], label=cont, zorder=3
        )

    ax.set_title(sp, fontsize=13, pad=4, loc="center")
    ax.set_xticks(range(len(x_labels)), x_labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(False)
    for side in ["top","right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    if use_percent:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.3g}%"))
        ax.set_ylabel("Mean (%)", fontsize=11)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_ylabel("Mean", fontsize=11)

# Unified Legend
handles, labels = axes[0].get_legend_handles_labels()
keep = {lb: hd for hd, lb in zip(handles, labels)}
ordered_handles = [keep[lb] for lb in continent_order if lb in keep]
fig.legend(
    ordered_handles, [lb for lb in continent_order if lb in keep],
    loc="lower center", ncol=min(6, len(ordered_handles)),
    frameon=False, borderaxespad=0.3, fontsize=10
)

fig.tight_layout(rect=(0, 0.12, 1, 1))
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"[Done] Plot saved: {out_png} | PDF: {out_pdf}")