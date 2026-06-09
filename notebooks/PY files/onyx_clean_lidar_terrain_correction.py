from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask

import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer

import harmonica as hm


def load_profile_csv(profile_csv, dst_crs="EPSG:26913"):
    gravity = pd.read_csv(profile_csv)
    gravity.columns = gravity.columns.str.strip()

    required_input = ["station", "lat", "lon", "elev_m", "grav_rel", "FAC"]

    missing = [col for col in required_input if col not in gravity.columns]
    if missing:
        raise ValueError(f"Profile CSV is missing required columns: {missing}")

    print("\nNaN count before dropping:")
    print(gravity[required_input].isna().sum())

    bad_rows = gravity[gravity[required_input].isna().any(axis=1)]

    if len(bad_rows) > 0:
        print("\nDropping rows with missing required values:")
        print(bad_rows[required_input])

    gravity = gravity.dropna(subset=required_input).copy()
    gravity = gravity.sort_values("station").reset_index(drop=True)

    print(f"\nRows retained after dropping NaNs: {len(gravity)}")

    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)

    gravity["easting"], gravity["northing"] = transformer.transform(
        gravity["lon"].to_numpy(),
        gravity["lat"].to_numpy(),
    )

    # Standard names used by the correction workflow
    gravity["station_m"] = gravity["station"]
    gravity["elevation_m"] = gravity["elev_m"]
    gravity["gravity_tied_mgal"] = gravity["grav_rel"]
    gravity["free_air_correction_mgal"] = gravity["FAC"]

    required = [
        "station",
        "station_m",
        "gravity_tied_mgal",
        "free_air_correction_mgal",
        "easting",
        "northing",
        "elevation_m",
    ]

    if gravity[required].isna().any().any():
        print("\nRows still containing NaN values:")
        print(gravity[gravity[required].isna().any(axis=1)][required])
        raise ValueError("NaN values remain after interpolation.")

    return gravity


def merge_reproject_crop_dem(
    dem_tiles,
    gravity,
    out_dem,
    dst_crs="EPSG:26913",
    dem_z_factor=0.30480060960121924,
    pad_m=500.0,
):
    dem_tiles = [Path(p) for p in dem_tiles]

    if len(dem_tiles) == 0:
        raise FileNotFoundError("No DEM tiles were provided.")

    srcs = [rasterio.open(p) for p in dem_tiles]
    mosaic, mosaic_transform = merge(srcs)

    src_crs = srcs[0].crs
    src_meta = srcs[0].meta.copy()

    for src in srcs:
        src.close()

    left, bottom, right, top = rasterio.transform.array_bounds(
        mosaic.shape[1],
        mosaic.shape[2],
        mosaic_transform,
    )

    transform, width, height = calculate_default_transform(
        src_crs,
        dst_crs,
        mosaic.shape[2],
        mosaic.shape[1],
        left,
        bottom,
        right,
        top,
    )

    reprojected = np.empty((height, width), dtype=np.float32)

    reproject(
        source=mosaic[0],
        destination=reprojected,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    meta = src_meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": transform,
        }
    )

    temp_path = Path(out_dem).with_name("TEMP_reprojected_dem.tif")

    with rasterio.open(temp_path, "w", **meta) as dst:
        dst.write(reprojected, 1)

    minx = gravity["easting"].min() - pad_m
    maxx = gravity["easting"].max() + pad_m
    miny = gravity["northing"].min() - pad_m
    maxy = gravity["northing"].max() + pad_m

    crop_box = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs=dst_crs,
    )

    with rasterio.open(temp_path) as src:
        cropped, cropped_transform = mask(src, crop_box.geometry, crop=True)

        cropped_meta = src.meta.copy()
        cropped_meta.update(
            {
                "height": cropped.shape[1],
                "width": cropped.shape[2],
                "transform": cropped_transform,
            }
        )

    temp_path.unlink(missing_ok=True)

    dem = cropped[0].astype(float)

    nodata = cropped_meta.get("nodata")
    if nodata is not None:
        dem[dem == nodata] = np.nan

    dem = dem * dem_z_factor

    if np.isnan(dem).all():
        raise ValueError("Cropped DEM is entirely NaN. Check CRS/DEM coverage.")

    cropped_meta.update(dtype="float32", nodata=np.nan)

    with rasterio.open(out_dem, "w", **cropped_meta) as dst:
        dst.write(dem.astype("float32"), 1)

    print("\nStation easting range:")
    print(gravity["easting"].min(), gravity["easting"].max())

    print("\nStation northing range:")
    print(gravity["northing"].min(), gravity["northing"].max())

    print("\nCropped DEM elevation range, meters:")
    print(np.nanmin(dem), np.nanmax(dem))

    return dem, cropped_transform


def dem_to_xarray(dem, transform):
    nrows, ncols = dem.shape

    west = transform.c
    north = transform.f
    dx = transform.a
    dy = abs(transform.e)

    easting = west + dx * (np.arange(ncols) + 0.5)
    northing = north - dy * (np.arange(nrows) + 0.5)

    return xr.DataArray(
        dem,
        coords={"northing": northing, "easting": easting},
        dims=("northing", "easting"),
        name="topography",
    )


def resample_topography(topo, model_spacing_m=5.0):
    easting_new = np.arange(
        float(topo.easting.min()),
        float(topo.easting.max()),
        model_spacing_m,
    )

    northing_new = np.arange(
        float(topo.northing.min()),
        float(topo.northing.max()),
        model_spacing_m,
    )

    return topo.interp(easting=easting_new, northing=northing_new)


def compute_harmonica_correction(
    gravity,
    topo,
    density_kg_m3=2670.0,
    upward_offset_m=0.25,
    reference_mode="mean_station_elevation",
):
    if reference_mode == "mean_station_elevation":
        reference_elevation_m = float(gravity["elevation_m"].mean())
    elif reference_mode == "min_station_elevation":
        reference_elevation_m = float(gravity["elevation_m"].min())
    elif isinstance(reference_mode, (int, float)):
        reference_elevation_m = float(reference_mode)
    else:
        raise ValueError("Invalid reference_mode.")

    density = xr.where(
        topo >= reference_elevation_m,
        density_kg_m3,
        -density_kg_m3,
    )

    prisms = hm.prism_layer(
        coordinates=(topo.easting, topo.northing),
        surface=topo,
        reference=reference_elevation_m,
        properties={"density": density},
    )

    coordinates = (
        gravity["easting"].to_numpy(),
        gravity["northing"].to_numpy(),
        gravity["elevation_m"].to_numpy() + upward_offset_m,
    )

    harmonica_gz_mgal = prisms.prism_layer.gravity(
        coordinates,
        field="g_z",
    )

    out = gravity.copy()

    out["reference_elevation_m"] = reference_elevation_m
    out["obs_height_used_m"] = gravity["elevation_m"] + upward_offset_m
    out["harmonica_gz_mgal"] = harmonica_gz_mgal

    # Free-air only
    out["gravity_free_air_only_mgal"] = (
        out["gravity_tied_mgal"]
        + out["free_air_correction_mgal"]
    )

    # Free-air corrected relative gravity
    out["gravity_free_air_only_mgal"] = (
        out["gravity_tied_mgal"]
        + out["free_air_correction_mgal"]
    )
    
    # Remove constant Harmonica offset because this is a relative gravity survey
    out["harmonica_gz_relative_mgal"] = (
        out["harmonica_gz_mgal"]
        - out["harmonica_gz_mgal"].mean()
    )
    
    # Keep both sign options for QC
    out["corrected_add_rel_mgal"] = (
        out["gravity_free_air_only_mgal"]
        + out["harmonica_gz_relative_mgal"]
    )
    
    out["corrected_subtract_rel_mgal"] = (
        out["gravity_free_air_only_mgal"]
        - out["harmonica_gz_relative_mgal"]
    )
    
    # Final correction: subtract relative DEM-modeled terrain effect
    out["gravity_harmonica_lidar_corrected_mgal"] = out["corrected_subtract_rel_mgal"]

    return out


def run_lidar_terrain_correction(
    profile_csv,
    dem_tiles,
    out_csv,
    out_dem,
    dst_crs="EPSG:26913",
    dem_z_factor=0.30480060960121924,
    model_spacing_m=5.0,
    pad_m=500.0,
    density_kg_m3=2670.0,
    upward_offset_m=0.25,
    reference_mode="mean_station_elevation",
):
    gravity = load_profile_csv(
        profile_csv=profile_csv,
        dst_crs=dst_crs,
    )

    dem, transform = merge_reproject_crop_dem(
        dem_tiles=dem_tiles,
        gravity=gravity,
        out_dem=out_dem,
        dst_crs=dst_crs,
        dem_z_factor=dem_z_factor,
        pad_m=pad_m,
    )

    topo = dem_to_xarray(dem, transform)
    topo = resample_topography(topo, model_spacing_m=model_spacing_m)

    out = compute_harmonica_correction(
        gravity=gravity,
        topo=topo,
        density_kg_m3=density_kg_m3,
        upward_offset_m=upward_offset_m,
        reference_mode=reference_mode,
    )

    out.to_csv(out_csv, index=False)

    print("\nSaved corrected gravity CSV:")
    print(out_csv)

    print("\nSaved merged DEM:")
    print(out_dem)

    print("\nFinal correction summary:")
    print(
        out[
            [
                "station",
                "gravity_free_air_only_mgal",
                "harmonica_gz_mgal",
                "gravity_harmonica_lidar_corrected_mgal",
            ]
        ].head()
    )

    return out, topo