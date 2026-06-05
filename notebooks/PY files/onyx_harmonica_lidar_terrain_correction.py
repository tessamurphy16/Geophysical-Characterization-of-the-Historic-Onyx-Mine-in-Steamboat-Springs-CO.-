"""
Onyx Mine gravity terrain correction using Harmonica + LiDAR DEM prisms.

This follows the Fatiando/Harmonica topographic-correction workflow:
1) load gravity observations,
2) load/crop/project DEM,
3) build a prism layer from DEM topography,
4) forward model g_z from topographic masses at gravity stations,
5) replace the simple Bouguer slab correction with DEM-derived topography effect.

Input files expected in the same folder as this script or update the paths below.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
import xarray as xr

import harmonica as hm

# ------------------------- USER SETTINGS -------------------------
DATA_DIR = Path("/mnt/data")  # change to Path.cwd() if running locally beside the files

GRAVITY_CSV = DATA_DIR / "Onyx_Gravity_Final_Bouguer_Corrected.csv"
GPS_CSV = DATA_DIR / "gps_points_with_elevation.csv"  # optional but useful for checking GPS elevations
DEM_TILES = [
    DATA_DIR / "LD26231421.tif",
    DATA_DIR / "LD26231424.tif",
    DATA_DIR / "LD26261421.tif",
    DATA_DIR / "LD26261424.tif",
]

OUT_CSV = DATA_DIR / "Onyx_Gravity_Harmonica_LiDAR_Terrain_Corrected.csv"
OUT_DEM = DATA_DIR / "Onyx_LiDAR_DEM_UTM13N_meters.tif"

# LiDAR tiles are EPSG:6430, horizontal units = US survey feet; elevation values also appear to be feet.
DST_CRS = "EPSG:26913"       # UTM zone 13N, meters, appropriate for lon ~ -106.8, lat ~ 40.5
DEM_Z_FACTOR = 0.30480060960121924  # US survey ft -> m
MODEL_SPACING_M = 5.0        # increase to 10 or 20 m if Harmonica is slow; decrease for finer correction
PAD_M = 500.0                # crop DEM this far beyond gravity line before forward modeling
DENSITY_KG_M3 = 2670.0       # standard Bouguer crustal density; test 2200-2670 for sandstone sensitivity
UPWARD_OFFSET_M = 0.25       # prevents stations being exactly on/inside prism tops due to DEM/GPS mismatch
# -----------------------------------------------------------------


def load_and_prepare_gravity() -> pd.DataFrame:
    """Load gravity data and merge GPS elevations if needed."""
    g = pd.read_csv(GRAVITY_CSV)
    g.columns = [c.strip() for c in g.columns]

    # Standardize gravity coordinate column names.
    if "latitude" not in g.columns and "Latitude" in g.columns:
        g = g.rename(columns={"Latitude": "latitude"})
    if "longitude" not in g.columns and "Longitude" in g.columns:
        g = g.rename(columns={"Longitude": "longitude"})

    # Optional: merge GPS elevations/coordinates for missing stations or QC.
    if GPS_CSV.exists():
        gps = pd.read_csv(GPS_CSV)
        gps = gps.rename(columns={"Latitude": "gps_latitude", "Longitude": "gps_longitude"})
        gps = gps.rename(columns={"elevation_m": "gps_elevation_m"})
        g = g.merge(gps[["station", "gps_longitude", "gps_latitude", "gps_elevation_m"]], on="station", how="left")

        # Fill missing coordinates/elevations from GPS file, but keep the gravity CSV as primary.
        g["longitude"] = g["longitude"].fillna(g["gps_longitude"])
        g["latitude"] = g["latitude"].fillna(g["gps_latitude"])
        g["elevation_m"] = g["elevation_m"].fillna(g["gps_elevation_m"])

    required = ["station", "longitude", "latitude", "elevation_m", "gravity_tied_mgal", "free_air_correction_mgal", "bouguer_correction_mgal"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Missing required gravity columns: {missing}\nAvailable columns: {list(g.columns)}")

    g = g.dropna(subset=["longitude", "latitude", "elevation_m"]).copy()
    g = g.sort_values("station").reset_index(drop=True)
    return g


def merge_crop_reproject_dem(gravity: pd.DataFrame) -> tuple[xr.DataArray, pd.DataFrame]:
    """Merge LiDAR tiles, crop around stations, reproject to metric CRS, output xarray DEM in meters."""
    existing_tiles = [p for p in DEM_TILES if p.exists()]
    if not existing_tiles:
        raise FileNotFoundError("No DEM tiles were found. Check DEM_TILES paths.")

    # Project station lon/lat to destination CRS.
    station_transformer = Transformer.from_crs("EPSG:4326", DST_CRS, always_xy=True)
    easting_m, northing_m = station_transformer.transform(gravity["longitude"].values, gravity["latitude"].values)
    gravity = gravity.copy()
    gravity["easting_m"] = easting_m
    gravity["northing_m"] = northing_m

    # Merge tiles in original CRS/units.
    srcs = [rasterio.open(p) for p in existing_tiles]
    merged, merged_transform = merge(srcs)
    src_profile = srcs[0].profile.copy()
    src_crs = srcs[0].crs
    for src in srcs:
        src.close()

    # Convert DEM elevations from feet to meters and set nodata to NaN.
    dem_src = merged[0].astype("float32")
    nodata = src_profile.get("nodata", -9999.0)
    dem_src = np.where(dem_src == nodata, np.nan, dem_src * DEM_Z_FACTOR).astype("float32")

    # Write merged source DEM to memory so rasterio can reproject it.
    src_profile.update(
        driver="GTiff",
        height=dem_src.shape[0],
        width=dem_src.shape[1],
        count=1,
        dtype="float32",
        crs=src_crs,
        transform=merged_transform,
        nodata=np.nan,
    )

    with MemoryFile() as memfile:
        with memfile.open(**src_profile) as src:
            src.write(dem_src, 1)

            # Crop bounds in destination CRS around gravity stations, then invert to source bounds.
            dst_left = gravity["easting_m"].min() - PAD_M
            dst_right = gravity["easting_m"].max() + PAD_M
            dst_bottom = gravity["northing_m"].min() - PAD_M
            dst_top = gravity["northing_m"].max() + PAD_M

            # Build destination transform at requested model spacing.
            dst_width = int(np.ceil((dst_right - dst_left) / MODEL_SPACING_M))
            dst_height = int(np.ceil((dst_top - dst_bottom) / MODEL_SPACING_M))
            dst_transform = rasterio.transform.from_origin(dst_left, dst_top, MODEL_SPACING_M, MODEL_SPACING_M)
            dst_dem = np.full((dst_height, dst_width), np.nan, dtype="float32")

            reproject(
                source=rasterio.band(src, 1),
                destination=dst_dem,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=DST_CRS,
                src_nodata=np.nan,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

    # Drop empty rows/columns around edges caused by reprojection/crop.
    valid_rows = np.where(np.any(np.isfinite(dst_dem), axis=1))[0]
    valid_cols = np.where(np.any(np.isfinite(dst_dem), axis=0))[0]
    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise RuntimeError("Reprojected/cropped DEM contains no valid cells. Check CRS, station positions, and DEM tiles.")

    dst_dem = dst_dem[valid_rows.min():valid_rows.max()+1, valid_cols.min():valid_cols.max()+1]
    new_left = dst_left + valid_cols.min() * MODEL_SPACING_M
    new_top = dst_top - valid_rows.min() * MODEL_SPACING_M
    new_transform = rasterio.transform.from_origin(new_left, new_top, MODEL_SPACING_M, MODEL_SPACING_M)

    # Fill any small internal NaN holes with the DEM median. For a cleaner workflow, replace this with interpolation if needed.
    if np.isnan(dst_dem).any():
        dst_dem = np.where(np.isfinite(dst_dem), dst_dem, np.nanmedian(dst_dem)).astype("float32")

    # Write metric DEM for QC/reuse.
    with rasterio.open(
        OUT_DEM, "w",
        driver="GTiff",
        height=dst_dem.shape[0],
        width=dst_dem.shape[1],
        count=1,
        dtype="float32",
        crs=DST_CRS,
        transform=new_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(dst_dem, 1)

    # Cell-center coordinates. Raster row 0 is north/top, so reverse to ascending northing for Harmonica.
    nrows, ncols = dst_dem.shape
    easting = new_left + (np.arange(ncols) + 0.5) * MODEL_SPACING_M
    northing_desc = new_top - (np.arange(nrows) + 0.5) * MODEL_SPACING_M
    northing = northing_desc[::-1]
    dem_ascending = dst_dem[::-1, :]

    topo = xr.DataArray(
        dem_ascending,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="topography",
        attrs={"units": "m", "crs": DST_CRS, "description": "LiDAR DEM reprojected to meters"},
    )
    return topo, gravity


def compute_harmonica_terrain_correction(gravity: pd.DataFrame, topo: xr.DataArray) -> pd.DataFrame:
    """Build Harmonica prism layer and compute DEM-derived terrain/topographic effect."""
    density = xr.full_like(topo, fill_value=DENSITY_KG_M3, dtype=float)

    prisms = hm.prism_layer(
        coordinates=(topo.easting, topo.northing),
        surface=topo,
        reference=0.0,
        properties={"density": density},
    )

    # Observation height: use station elevation, but force it slightly above local DEM to avoid point-inside-prism errors.
    station_topo = topo.interp(
        easting=("points", gravity["easting_m"].values),
        northing=("points", gravity["northing_m"].values),
        method="linear",
    ).values
    obs_height = np.maximum(gravity["elevation_m"].values, station_topo + UPWARD_OFFSET_M)

    coordinates = (gravity["easting_m"].values, gravity["northing_m"].values, obs_height)
    terrain_effect_mgal = prisms.prism_layer.gravity(coordinates, field="g_z")

    out = gravity.copy()
    out["dem_elevation_m_at_station"] = station_topo
    out["obs_height_used_m"] = obs_height
    out["harmonica_topography_effect_mgal"] = terrain_effect_mgal

    # Your existing final gravity is: gravity_tied + free_air - simple_Bouguer.
    # Replace the slab correction with DEM-forward-modeled topography:
    out["gravity_harmonica_lidar_corrected_mgal"] = (
        out["gravity_tied_mgal"]
        + out["free_air_correction_mgal"]
        - out["harmonica_topography_effect_mgal"]
    )

    # Equivalent way to adjust your already-final simple Bouguer result.
    if "gravity_final_mgal" in out.columns:
        out["harmonica_minus_simple_bouguer_mgal"] = (
            out["gravity_harmonica_lidar_corrected_mgal"] - out["gravity_final_mgal"]
        )

    out["simple_bouguer_minus_harmonica_effect_mgal"] = (
        out["bouguer_correction_mgal"] - out["harmonica_topography_effect_mgal"]
    )
    return out


def plot_qc(out: pd.DataFrame) -> None:
    """Basic QC plots."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out["station"], out["bouguer_correction_mgal"], "o-", label="simple Bouguer slab")
    ax.plot(out["station"], out["harmonica_topography_effect_mgal"], "o-", label="Harmonica LiDAR topo effect")
    ax.set_xlabel("Station")
    ax.set_ylabel("Correction/effect (mGal)")
    ax.set_title("Simple Bouguer correction vs LiDAR DEM forward-modeled topography")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 4))
    if "gravity_final_mgal" in out.columns:
        ax.plot(out["station"], out["gravity_final_mgal"], "o-", label="existing simple Bouguer corrected")
    ax.plot(out["station"], out["gravity_harmonica_lidar_corrected_mgal"], "o-", label="Harmonica LiDAR corrected")
    ax.set_xlabel("Station")
    ax.set_ylabel("Corrected relative gravity (mGal)")
    ax.set_title("Gravity after replacing slab Bouguer with LiDAR terrain model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def main():
    gravity = load_and_prepare_gravity()
    topo, gravity_projected = merge_crop_reproject_dem(gravity)
    out = compute_harmonica_terrain_correction(gravity_projected, topo)
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved corrected gravity CSV: {OUT_CSV}")
    print(f"Saved reprojected DEM: {OUT_DEM}")
    print("\nKey output columns:")
    print("  harmonica_topography_effect_mgal")
    print("  gravity_harmonica_lidar_corrected_mgal")
    print("  harmonica_minus_simple_bouguer_mgal")
    print("  simple_bouguer_minus_harmonica_effect_mgal")
    print("\nSummary:")
    cols = ["bouguer_correction_mgal", "harmonica_topography_effect_mgal", "simple_bouguer_minus_harmonica_effect_mgal"]
    print(out[cols].describe())

    plot_qc(out)


if __name__ == "__main__":
    main()
