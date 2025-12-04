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
from matplotlib.ticker import FuncFormatter, MultipleLocator, MaxNLocator
from matplotlib.ticker import FixedLocator

# ==================== 0) Global Configuration ====================
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType
matplotlib.rcParams["ps.fonttype"]  = 42


tif_infos = [
    (r"E:/2025论文/cyn/产量预测/预测结果/whea-页面/predicted_yield_effect_whea_stressall.tif", "WHEA", 100),
    (r"E:/2025论文/cyn/产量预测/预测结果/玉米-页面/predicted_yield_effect_maiz_stressall.tif", "MAIZ", 100),
    (r"E:/2025论文/cyn/产量预测/预测结果/soyb-页面/predicted_yield_effect_soyb_stressall.tif", "SOYB", 100),
    (r"E:/2025论文/cyn/产量预测/预测结果/rice-页面/predicted_yield_effect_rice_stressall.tif", "RICE", 100),
]

shp_path = r"E:/2025论文/cyn/world_region.shp"
name_field = "REGION_ADJ"  

# Outputs
out_png      = "continent_means_quadpanel.png"  
out_pdf      = "continent_means_quadpanel.pdf"
out_csv_long = "continent_means_long.csv"
out_csv_wide = "continent_means_wide.csv"

# Plot Styling
axis_pad_ratio  = 0.05       
share_x_range   = False     
separator_ls    = "--"      
separator_alpha = 0.5
separator_lw    = 0.6
bar_color       = "#cadce3"
PIXEL_SCALE     = 0.542     


# ===================== Utility Functions =====================
def set_smart_percent_axis(ax, xmin, xmax):
    """
    Aligns percentage X-axis ticks with display precision to avoid duplicate labels.
    Rules:
      - Prioritize step sizes that result in 4–7 ticks.
      - If step >= 1, use integer percentages (e.g., 50%).
      - If step < 1, use 1 decimal place (e.g., 0.5% step -> 47.5%).
      - Removes duplicates (prevents rounding errors creating identical labels).
    """
    span = xmax - xmin
    steps = [10, 5, 2, 1, 0.5, 0.2, 0.1]

    step = None
    for s in steps:
        n = int(np.floor(span / s) + 1)
        if 4 <= n <= 7:
            step = s
            break
    if step is None:
        target = max(span / 5.0, 0.1)
        step = min(steps, key=lambda x: abs(x - target))

    start = np.floor(xmin / step) * step
    ticks = np.arange(start, xmax + step * 0.5, step)
    ticks = ticks[(ticks >= xmin) & (ticks <= xmax)]

    if step >= 1:
        dec = 0
        fmt = FuncFormatter(lambda v, p: f"{int(round(v))}%")
        labels = np.array([f"{int(round(t))}" for t in ticks])
    else:
        dec = 1
        fmt = FuncFormatter(lambda v, p, dec=dec: f"{v:.{dec}f}%")
        labels = np.array([f"{t:.{dec}f}" for t in ticks])

    _, uniq_idx = np.unique(labels, return_index=True)
    ticks = ticks[np.sort(uniq_idx)]

    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

def ensure_name_field(gdf: gpd.GeoDataFrame, name_field: str) -> str:
    """Checks if the name field exists, otherwise guesses a suitable one."""
    if name_field in gdf.columns: return name_field
    cand = [c for c in gdf.columns if c.lower() in ("name","continent","region","admin","area")]
    if cand:
        print(f"[Info] Specified field not found. Using: {cand[0]}")
        return cand[0]
    non_geom = [c for c in gdf.columns if c != gdf.geometry.name]
    if not non_geom: 
        raise ValueError("SHP contains no attribute fields to use as names. Please check 'name_field'.")
    print(f"[Info] Specified field not found. Using: {non_geom[0]}")
    return non_geom[0]

def zonal_mean_for_tif(tif_path: str, shp_gdf: gpd.GeoDataFrame, name_field: str) -> pd.DataFrame:
    """
    Calculates zonal mean by region (dissolves first).
    Note: Pixels are multiplied by PIXEL_SCALE (0.542) before calculation.
    """
    if not os.path.exists(tif_path): raise FileNotFoundError(tif_path)
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs; raster_nodata = src.nodata
    
    gdf = shp_gdf.copy()
    if gdf.crs != raster_crs: gdf = gdf.to_crs(raster_crs)
    gdf[name_field] = gdf[name_field].astype(str)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf_diss = gdf.dissolve(by=name_field, as_index=False)

    rows = []
    with rasterio.open(tif_path) as src:
        for _, r in gdf_diss.iterrows():
            geom = r.geometry; name = str(r[name_field])
            if geom is None or geom.is_empty: continue
            
            data, _ = mask(src, [mapping(geom)], crop=True, nodata=raster_nodata)
            a = data[0].astype("float64")
            if raster_nodata is not None: a[a == raster_nodata] = np.nan
            a[np.isinf(a)] = np.nan
            
            a *= PIXEL_SCALE                      
            
            mean = np.nan if np.isnan(a).all() else np.nanmean(a)
            rows.append({"continent": name, "mean_raw": mean})
    return pd.DataFrame(rows)

def infer_scale_and_format(df_raw: pd.DataFrame):
    """Infers whether data is 0-1 or 0-100 based on values."""
    vals = df_raw["mean_raw"].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        scale = 1.0; as_percent = False
    else:
        share_le1 = np.mean((finite >= 0) & (finite <= 1.2))
        if share_le1 >= 0.6:
            scale = 100.0; as_percent = True
        elif np.nanmax(finite) <= 120:
            scale = 1.0;  as_percent = True
        else:
            scale = 1.0;  as_percent = False
            
    df = df_raw.copy(); df["mean"] = df["mean_raw"] * scale
    def fmt(x): return "NA" if np.isnan(x) else (f"{x:.2f}%" if as_percent else f"{x:.2f}")
    return df[["continent","mean"]], as_percent, fmt

def apply_scale_hint(df_raw: pd.DataFrame, scale_hint):
    """Applies user-specified scaling hint."""
    if scale_hint is None: return infer_scale_and_format(df_raw)
    df = df_raw.copy(); df["mean"] = df["mean_raw"] * float(scale_hint)
    as_percent = (float(scale_hint) == 100.0)
    def fmt(x): return "NA" if np.isnan(x) else (f"{x:.2f}%" if as_percent else f"{x:.2f}")
    return df[["continent","mean"]], as_percent, fmt

def draw_panel(ax, df_panel: pd.DataFrame, title: str, as_percent: bool, fmt):
    """Draws a single horizontal bar chart panel."""
    dfp = df_panel.sort_values("mean", ascending=False).reset_index(drop=True)
    labels = dfp["continent"].tolist()
    vals   = dfp["mean"].to_numpy(dtype=float)
    y_pos = np.arange(len(labels))
    is_nan = np.isnan(vals)
    x_plot = np.where(is_nan, 0.0, vals)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        xmin, xmax = 0.0, 1.0
    else:
        xmin, xmax = float(finite.min()), float(finite.max())
        if xmin == xmax: xmin -= 0.5; xmax += 0.5
        pad = (xmax - xmin) * axis_pad_ratio
        xmin -= pad; xmax += pad
    ax.set_xlim(xmin, xmax)

    bars = ax.barh(y_pos, x_plot, color=bar_color, zorder=2, align="center")
    ax.set_yticks(y_pos, labels=labels)

    for y in y_pos:
        ax.hlines(y=y, xmin=xmin, xmax=xmax,
                  linestyles=separator_ls, linewidth=separator_lw,
                  alpha=separator_alpha, zorder=1)

    span = xmax - xmin
    inset = 0.012 * span
    right_guard = xmax - 0.015 * span
    for bar, v, nanflag in zip(bars, vals, is_nan):
        label = "NA" if nanflag else fmt(v)
        w = bar.get_width()
        x_text = w - inset; ha = "right"
        if (w - xmin) < 0.06 * span:
            x_text = min(w + inset, right_guard)
            ha = "left" if x_text < right_guard else "right"
        ax.text(x_text, bar.get_y() + bar.get_height()/2,
                label, va="center", ha=ha, fontsize=11, zorder=3)

    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    if as_percent:
        set_smart_percent_axis(ax, xmin, xmax)  
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    for s in ["top","right","bottom","left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=13, pad=6)

# ===================== Main Execution Flow =====================
if not os.path.exists(shp_path):
    raise FileNotFoundError(f"Shapefile not found: {shp_path}")
gdf_all = gpd.read_file(shp_path)
name_field = ensure_name_field(gdf_all, name_field)

panel_tables, long_rows = [], []
for entry in tif_infos:
    if len(entry) == 2:
        tif_path, title = entry; scale_hint = None
    elif len(entry) == 3:
        tif_path, title, scale_hint = entry
    else:
        raise ValueError("tif_infos elements must be 2 or 3 values: (path, title[, scale])")
    
    df_raw = zonal_mean_for_tif(tif_path, gdf_all, name_field)
    df_scaled, as_percent, fmt = apply_scale_hint(df_raw, scale_hint)
    df_scaled["panel"] = title; df_scaled["as_percent"] = as_percent
    
    panel_tables.append((title, df_scaled.copy(), as_percent, fmt))
    long_rows.append(df_scaled.assign(tif=os.path.basename(tif_path)))

# Export CSVs
df_long = pd.concat(long_rows, ignore_index=True)
df_long.to_csv(out_csv_long, index=False, encoding="utf-8-sig")
df_wide = df_long.pivot_table(index="continent", columns="panel", values="mean").reset_index()
df_wide.to_csv(out_csv_wide, index=False, encoding="utf-8-sig")
print(f"[Done] CSVs saved: {out_csv_long} | {out_csv_wide}")

# ===================== Plotting (4 Panels Vertical) =====================
n_panels = len(panel_tables)
assert n_panels == 4, f"Current panel count is {n_panels}, required 4 for vertical layout."

fig, axes = plt.subplots(n_panels, 1, figsize=(5.5, 16), dpi=200)
if n_panels == 1:
    axes = [axes]

for ax, (title, dfp, as_percent, fmt) in zip(axes, panel_tables):
    draw_panel(ax, dfp[["continent","mean"]], title, as_percent, fmt)

fig.text(0.5, 0.02, "Regional mean", ha="center", fontsize=16)

plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()
print(f"[Done] Plot saved: {out_png} | PDF: {out_pdf}")