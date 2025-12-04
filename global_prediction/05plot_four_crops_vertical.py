# -*- coding: utf-8 -*-
"""
Four species vertical layout (Robinson projection).
Shared colorbar located at the bottom of all maps (length 50% of map width).
Data Processing: After reading each raster, pixel values are multiplied by 0.542 
before being used in global min/max calculation and plotting.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.colors import LinearSegmentedColormap

# ----------------- User Parameters -----------------
land_shp_path    = r"E:/2025_Paper/cyn/ssp_prediction/ditu.shp"       
country_shp_path = r"E:/2025_Paper/cyn/ssp_prediction/worldmap.shp"   

species = [
    ("WHEA", r"E:/2025_Paper/cyn/yield_pred/results/whea/predicted_yield_effect_whea_stressall.tif"),
    ("MAIZ", r"E:/2025_Paper/cyn/yield_pred/results/maiz/predicted_yield_effect_maiz_stressall.tif"),
    ("SOYB", r"E:/2025_Paper/cyn/yield_pred/results/soyb/predicted_yield_effect_soyb_stressall.tif"),
    ("RICE", r"E:/2025_Paper/cyn/yield_pred/results/rice/predicted_yield_effect_rice_stressall.tif"),
]

figure_title = "Predictions for Four Crops"
cbar_label   = "Predicted value (unit)"    
out_pdf = r"./four_crops_vertical.pdf"
out_png = r"./four_crops_vertical.png"

# Raster Scaling Factor
scale_factor = 0.542

# Colorbar and Data Clipping
fixed_vrange    = None      
clip_percentile = None      

# Styles
land_facecolor_bottom = "#e6e6e6"
country_facecolor     = "#e0e0e0"
country_edgecolor     = "white"
country_linewidth     = 0.3
outline_lw = 0.8

cmap = LinearSegmentedColormap.from_list(
    "BO_soft_nowhite",
    [
        (0.00, "#0B3C6D"),  # Deep Blue
        (0.15, "#1F66A7"),  # Mid Blue
        (0.32, "#4F92C8"),  # Light Blue
        (0.50, "#A6C6DA"),  # Very Light Blue (Not white)
        (0.68, "#F3C29C"),  # Very Light Orange (Not white)
        (0.85, "#E67C3A"),  # Mid Orange
        (1.00, "#8E1B10"),  # Deep Red
    ],
    N=256,
)

figsize = (7.2, 12.6)
dpi = 300

cbar_length_ratio = 0.5
cbar_height_ratio = 0.06   # Height = 6% of the main plot height
cbar_pad_ratio    = 0.12   # Padding = 8-12% of the main plot height

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
mpl.rcParams["font.family"] = "sans-serif"


def reproject_to_wgs84_array(ds):
    """
    Reprojects band 1 to WGS84 (EPSG:4326).
    Returns: array (float32, with NaNs), transform, bounds
    """
    if ds.crs is None:
        raise ValueError("TIF is missing CRS, cannot reproject.")
    
    epsg = getattr(ds.crs, "to_epsg", lambda: None)()
    
    if epsg == 4326:
        arr = ds.read(1).astype("float32")
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        b = ds.bounds
        return arr, ds.transform, (b.left, b.bottom, b.right, b.top)

    dst_crs = "EPSG:4326"
    transform, width, height = calculate_default_transform(
        ds.crs, dst_crs, ds.width, ds.height, *ds.bounds
    )
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=rasterio.band(ds, 1),
        destination=dst,
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        dst_nodata=np.nan,
    )
    miny, minx, maxy, maxx = rasterio.transform.array_bounds(height, width, transform)
    bounds = (minx, miny, maxx, maxy)
    return dst, transform, bounds


print("Reading base map shapefiles...")
land_gdf = gpd.read_file(land_shp_path) if land_shp_path and Path(land_shp_path).exists() else None
country_gdf = gpd.read_file(country_shp_path)

# ---------- Read 4 Species TIFs, Scale by 0.542, and Cache ----------
print("Processing rasters...")
items = []
for name, tif_path in species:
    if not Path(tif_path).exists():
        print(f"Warning: File not found {tif_path}")
        continue
        
    with rasterio.open(tif_path) as ds:
        arr, transform, bounds = reproject_to_wgs84_array(ds)
        arr = arr.astype("float32") * scale_factor
        
        if clip_percentile is not None:
            lo, hi = np.nanpercentile(arr, clip_percentile)
            arr = np.clip(arr, lo, hi)
            
        items.append(dict(name=name, arr=arr, bounds=bounds))

if fixed_vrange is not None:
    vmin, vmax = fixed_vrange
else:
    stacked = np.array([it["arr"] for it in items], dtype="float32")
    vmin = float(np.nanmin(stacked))
    vmax = float(np.nanmax(stacked))
    print(f"Global Data Range: Min={vmin:.4f}, Max={vmax:.4f}")

if (vmin < 0) and (vmax > 0):
    norm_shared = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
else:
    norm_shared = Normalize(vmin=vmin, vmax=vmax)

# ---------- Plotting ----------
print("Generating plot...")
proj_map = ccrs.Robinson()
pc = ccrs.PlateCarree()

fig, axes = plt.subplots(
    nrows=len(items), ncols=1, figsize=figsize, dpi=dpi,
    subplot_kw={"projection": proj_map}, constrained_layout=False
)
axes = np.atleast_1d(axes)

fig.subplots_adjust(top=0.92, bottom=0.12, left=0.06, right=0.98, hspace=0.15)

fig.text(0.5, 0.975, figure_title, ha="center", va="top",
         fontsize=14, fontweight="bold")

for i, ax in enumerate(axes):
    obj = items[i]
    arr = obj["arr"]
    left, bottom, right, top = obj["bounds"]

    if land_gdf is not None:
        ax.add_geometries(
            land_gdf.geometry, crs=pc,
            facecolor=land_facecolor_bottom, edgecolor="none",
            linewidth=0.0, zorder=0.5
        )
    ax.add_geometries(
        country_gdf.geometry, crs=pc,
        facecolor=country_facecolor, edgecolor=country_edgecolor,
        linewidth=country_linewidth, zorder=1.0
    )

    ax.imshow(
        arr, origin="upper", extent=(left, right, bottom, top),
        transform=pc, cmap=cmap, norm=norm_shared, zorder=3.0
    )

    ax.set_global()
    spine = ax.spines.get('geo', None)
    if spine is not None:
        spine.set_visible(True)
        spine.set_linewidth(outline_lw)
        spine.set_edgecolor("black")
    elif hasattr(ax, "outline_patch"):
        ax.outline_patch.set_visible(True)
        ax.outline_patch.set_linewidth(outline_lw)
        ax.outline_patch.set_edgecolor("black")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(-0.02, 0.5, obj["name"], transform=ax.transAxes,
            va="center", ha="right", fontsize=10)

ref_bbox = axes[-1].get_position(fig)
cb_w = ref_bbox.width * cbar_length_ratio
cb_h = ref_bbox.height * cbar_height_ratio
pad  = ref_bbox.height * cbar_pad_ratio

cb_x0 = ref_bbox.x0 + (ref_bbox.width - cb_w) / 2.0
cb_y0 = ref_bbox.y0 - pad - cb_h

cax = fig.add_axes([cb_x0, cb_y0, cb_w, cb_h])

mappable = mpl.cm.ScalarMappable(norm=norm_shared, cmap=cmap)
mappable.set_array([])  
cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
cb.set_label(cbar_label, fontsize=8)
cb.ax.tick_params(labelsize=8, length=3, pad=2)
cb.outline.set_linewidth(0.6)
cax.set_zorder(10)
cax.set_clip_on(False)

# ---------- Saving ----------
print("Saving results...")
Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved successfully:\n  {out_pdf}\n  {out_png}")