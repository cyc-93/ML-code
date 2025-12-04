from pathlib import Path
import numpy as np
import rasterio

# ====== Options (Currently fixed for UKESM1-0-LL ssp126 2081-2100) ======
GLOB_TMAX = "wc2.1_5m_tmax_UKESM1-0-LL_ssp585_2081-2100.tif"
OUTDIR = "out"               # Output directory
DO_SPLIT = True              # Whether to split tmax/tmin into 12 individual monthly files
WRITE_TMEAN_STACK = True     # Whether to additionally write a 12-band tmean stack
# ======================================================================

def month_tag(i: int) -> str:
    return f"{i:02d}"  # 01..12

def ensure_same_grid(src_a, src_b):
    keys = ["driver", "width", "height", "transform", "crs"]
    for k in keys:
        if getattr(src_a, k) != getattr(src_b, k):
            raise ValueError(f"Input rasters do not match in {k}, cannot perform pixel-wise addition.")

def write_single_band(array: np.ndarray, profile: dict, out_path: Path):
    p = profile.copy()
    p.update(count=1, dtype="float32", compress="LZW")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(array.astype("float32", copy=False), 1)

def split_multiband(in_path: Path, outdir: Path, prefix: str, varname: str):
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        nodata = src.nodata
        for b in range(1, src.count + 1):
            data = src.read(b, masked=True)
            filled = data.filled(nodata if nodata is not None else np.nan)
            out_path = outdir / f"{prefix}_{varname}_{month_tag(b)}.tif"
            write_single_band(filled, profile, out_path)

def compute_tmean(tmax_path: Path, tmin_path: Path, outdir: Path, prefix: str, write_stack: bool):
    with rasterio.open(tmax_path) as tx_src, rasterio.open(tmin_path) as tn_src:
        ensure_same_grid(tx_src, tn_src)
        profile = tx_src.profile.copy()
        if tx_src.nodata is None:
            profile.update(nodata=-9999.0)
        profile.update(dtype="float32")

        stack_dst = None
        if write_stack:
            stack_prof = profile.copy()
            stack_prof.update(count=tx_src.count, compress="LZW")
            stack_dst = rasterio.open(outdir / f"{prefix}_tmean_stack.tif", "w", **stack_prof)

        for b in range(1, tx_src.count + 1):
            tx = tx_src.read(b, masked=True).astype("float32")
            tn = tn_src.read(b, masked=True).astype("float32")
            tmean = (tx + tn) / 2.0
            out_path = outdir / f"{prefix}_tmean_{month_tag(b)}.tif"
            write_single_band(tmean.filled(profile["nodata"]), profile, out_path)
            if stack_dst is not None:
                stack_dst.write(tmean.filled(profile["nodata"]), b)

        if stack_dst is not None:
            stack_dst.close()

def infer_prefix(tmax_name: str) -> str:
    name = Path(tmax_name).stem
    return name.replace("wc2.1_5m_tmax_", "", 1)

def main():
    cur = Path(".")
    outdir = cur / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    tmax_list = sorted(cur.glob(GLOB_TMAX))
    if not tmax_list:
        raise SystemExit(f"tmax not found: {GLOB_TMAX}")

    for tmax_path in tmax_list:
        tmin_path = tmax_path.with_name(tmax_path.name.replace("tmax", "tmin"))
        if not tmin_path.exists():
            raise SystemExit(f"Matching tmin not found: {tmin_path.name} (Please ensure it exists in the same directory)")

        prefix = infer_prefix(tmax_path.name)
        print(f"[Processing] {prefix}")

        if DO_SPLIT:
            split_multiband(tmax_path, outdir, prefix, "tmax")
            split_multiband(tmin_path, outdir, prefix, "tmin")

        compute_tmean(tmax_path, tmin_path, outdir, prefix, WRITE_TMEAN_STACK)

    print("Done ✅ Output directory:", outdir.as_posix())

if __name__ == "__main__":
    main()