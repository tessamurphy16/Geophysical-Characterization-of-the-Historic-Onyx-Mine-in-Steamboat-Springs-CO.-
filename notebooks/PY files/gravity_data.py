from pathlib import Path
import pandas as pd
import numpy as np


CG5_COLUMNS = [
    "line", "station", "alt", "gravity_mgal", "sd_mgal",
    "tiltx", "tilty", "temp", "tide", "duration_s", "rejected",
    "time", "decimal_time", "terrain", "date"
]


def read_cg5_txt(filepath, instrument_name=None):
    filepath = Path(filepath)
    rows = []
    current_survey = None
    current_sn = None

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("/"):
                if "Survey name:" in line:
                    current_survey = line.split("Survey name:")[-1].strip()
                if "Instrument S/N:" in line:
                    current_sn = line.split("Instrument S/N:")[-1].strip()
                continue

            parts = line.split()

            if len(parts) < 15:
                continue

            try:
                numeric_values = [float(x) for x in parts[:11]]
                row = dict(
                    zip(
                        CG5_COLUMNS,
                        numeric_values + [
                            parts[11],
                            float(parts[12]),
                            float(parts[13]),
                            parts[14],
                        ],
                    )
                )

                row["survey"] = current_survey
                row["serial_number"] = current_sn
                row["instrument"] = instrument_name if instrument_name else current_sn
                rows.append(row)

            except ValueError:
                continue

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def load_onyx_gravity(cg5_1_path, cg5_2_path, survey_date="2026/05/27"):
    cg5_1 = read_cg5_txt(cg5_1_path, instrument_name="CG5-1")
    cg5_2 = read_cg5_txt(cg5_2_path, instrument_name="CG5-2")

    raw = pd.concat([cg5_1, cg5_2], ignore_index=True)

    raw = raw[raw["date"] == survey_date].copy()
    raw = raw[raw["duration_s"] > 0].copy()

    raw = raw[
        [
            "instrument",
            "survey",
            "serial_number",
            "line",
            "station",
            "gravity_mgal",
            "sd_mgal",
            "time",
            "date",
            "datetime",
            "decimal_time",
            "duration_s",
            "rejected",
            "alt",
            "tiltx",
            "tilty",
            "temp",
            "tide",
            "terrain",
        ]
    ]

    return raw.sort_values(["instrument", "datetime"]).reset_index(drop=True)


def fix_cg5_1_station_16(raw_df):
    """
    CG5-1 station 16 was accidentally labeled as station 15.
    
    Based on the field notes and timestamps:
    - first 3 station-15 readings are true station 15
    - all later station-15 readings before moving to station 17 are true station 16
    """
    df = raw_df.copy()

    mask = (df["instrument"] == "CG5-1") & (df["station"] == 15)
    idx = df[mask].sort_values("datetime").index

    if len(idx) > 3:
        df.loc[idx[3:], "station"] = 16.0

    return df


def separate_base_and_profile(raw_df):
    df = raw_df.copy()
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    output = []

    for instrument, group in df.groupby("instrument"):
        group = group.sort_values("datetime").copy()
        group["reading_type"] = "profile"

        group.iloc[:3, group.columns.get_loc("reading_type")] = "initial_base"
        group.iloc[-3:, group.columns.get_loc("reading_type")] = "final_base"

        output.append(group)

    labeled = pd.concat(output, ignore_index=True)

    profile_raw = labeled[labeled["reading_type"] == "profile"].copy()
    base_raw = labeled[labeled["reading_type"].isin(["initial_base", "final_base"])].copy()

    return profile_raw, base_raw, labeled


def average_profile_repeats(profile_raw):
    profile_avg = (
        profile_raw
        .groupby(["instrument", "station"], as_index=False)
        .agg(
            gravity_mean_mgal=("gravity_mgal", "mean"),
            gravity_std_mgal=("gravity_mgal", "std"),
            sd_mean_mgal=("sd_mgal", "mean"),
            sd_min_mgal=("sd_mgal", "min"),
            sd_max_mgal=("sd_mgal", "max"),
            n_readings=("gravity_mgal", "count"),
            first_time=("datetime", "min"),
            last_time=("datetime", "max"),
            mean_decimal_time=("decimal_time", "mean"),
            line_mean=("line", "mean"),
            alt_mean=("alt", "mean"),
            tiltx_mean=("tiltx", "mean"),
            tilty_mean=("tilty", "mean"),
            temp_mean=("temp", "mean"),
        )
        .sort_values(["instrument", "station"])
        .reset_index(drop=True)
    )

    return profile_avg


def summarize_base_readings(base_raw):
    base_summary = (
        base_raw
        .groupby(["instrument", "reading_type"], as_index=False)
        .agg(
            base_gravity_mean_mgal=("gravity_mgal", "mean"),
            base_gravity_std_mgal=("gravity_mgal", "std"),
            base_sd_mean_mgal=("sd_mgal", "mean"),
            n_base_readings=("gravity_mgal", "count"),
            mean_decimal_time=("decimal_time", "mean"),
            first_time=("datetime", "min"),
            last_time=("datetime", "max"),
        )
    )

    return base_summary


def apply_drift_correction(profile_avg, base_summary):
    corrected = profile_avg.copy()

    corrected["initial_base_mgal"] = np.nan
    corrected["final_base_mgal"] = np.nan
    corrected["total_drift_mgal"] = np.nan
    corrected["drift_rate_mgal_per_day"] = np.nan
    corrected["drift_correction_mgal"] = np.nan
    corrected["gravity_drift_corrected_mgal"] = np.nan

    for instrument in corrected["instrument"].unique():
        inst_base = base_summary[base_summary["instrument"] == instrument]

        initial = inst_base[inst_base["reading_type"] == "initial_base"].iloc[0]
        final = inst_base[inst_base["reading_type"] == "final_base"].iloc[0]

        g_initial = initial["base_gravity_mean_mgal"]
        g_final = final["base_gravity_mean_mgal"]

        t_initial = initial["mean_decimal_time"]
        t_final = final["mean_decimal_time"]

        total_drift = g_final - g_initial
        drift_rate = total_drift / (t_final - t_initial)

        mask = corrected["instrument"] == instrument
        t_station = corrected.loc[mask, "mean_decimal_time"]

        drift_at_station = drift_rate * (t_station - t_initial)

        corrected.loc[mask, "initial_base_mgal"] = g_initial
        corrected.loc[mask, "final_base_mgal"] = g_final
        corrected.loc[mask, "total_drift_mgal"] = total_drift
        corrected.loc[mask, "drift_rate_mgal_per_day"] = drift_rate
        corrected.loc[mask, "drift_correction_mgal"] = drift_at_station
        corrected.loc[mask, "gravity_drift_corrected_mgal"] = (
            corrected.loc[mask, "gravity_mean_mgal"] - drift_at_station
        )

    return corrected


def process_gravity_through_drift(
    cg5_1_path,
    cg5_2_path,
    survey_date="2026/05/27",
    fix_station_16=True,
):
    raw = load_onyx_gravity(cg5_1_path, cg5_2_path, survey_date=survey_date)

    if fix_station_16:
        raw = fix_cg5_1_station_16(raw)

    profile_raw, base_raw, labeled_raw = separate_base_and_profile(raw)

    profile_avg = average_profile_repeats(profile_raw)
    base_summary = summarize_base_readings(base_raw)
    drift_corrected = apply_drift_correction(profile_avg, base_summary)

    return raw, labeled_raw, profile_raw, base_raw, profile_avg, base_summary, drift_corrected

def tie_instruments_to_base(drift_corrected):
    """
    Places CG5-1 and CG5-2 onto a common relative gravity datum.

    After each instrument has been drift corrected separately, subtract
    that instrument's initial base value.

    Result:
        relative_gravity_mgal = drift-corrected gravity - initial base gravity

    This makes both instruments comparable if they used the same physical base station.
    """
    df = drift_corrected.copy()

    df["relative_gravity_mgal"] = (
        df["gravity_drift_corrected_mgal"] - df["initial_base_mgal"]
    )

    return df


def make_combined_profile(drift_corrected):
    """
    Combines both instruments into one station-ordered gravity profile.
    """
    tied = tie_instruments_to_base(drift_corrected)

    combined = (
        tied[
            [
                "instrument",
                "station",
                "gravity_mean_mgal",
                "gravity_drift_corrected_mgal",
                "relative_gravity_mgal",
                "gravity_std_mgal",
                "sd_mean_mgal",
                "n_readings",
                "first_time",
                "last_time",
                "drift_correction_mgal",
                "initial_base_mgal",
                "final_base_mgal",
                "total_drift_mgal",
            ]
        ]
        .sort_values("station")
        .reset_index(drop=True)
    )

    return combined


def process_gravity_full(
    cg5_1_path,
    cg5_2_path,
    survey_date="2026/05/27",
    fix_station_16=True,
):
    """
    Full processing through:
    1. Extract CG-5 data
    2. Fix station 16 issue
    3. Separate base/profile readings
    4. Average repeats
    5. Drift correct each instrument
    6. Tie both instruments to common base-relative datum
    """
    raw, labeled_raw, profile_raw, base_raw, profile_avg, base_summary, drift_corrected = process_gravity_through_drift(
        cg5_1_path=cg5_1_path,
        cg5_2_path=cg5_2_path,
        survey_date=survey_date,
        fix_station_16=fix_station_16,
    )

    combined_profile = make_combined_profile(drift_corrected)

    return raw, labeled_raw, profile_raw, base_raw, profile_avg, base_summary, drift_corrected, combined_profile

def apply_instrument_tie_correction(
    combined_profile,
    reference_instrument="CG5-2",
    tie_station=15.0,
):
    """
    Applies a constant instrument offset correction using an overlap station.

    This shifts all non-reference instruments so that their relative gravity
    matches the reference instrument at the tie station.
    """
    df = combined_profile.copy()

    df["instrument_tie_offset_mgal"] = 0.0
    df["gravity_tied_mgal"] = df["relative_gravity_mgal"]

    ref_rows = df[
        (df["instrument"] == reference_instrument) &
        (df["station"] == tie_station)
    ]

    if ref_rows.empty:
        raise ValueError(f"No reference reading found for {reference_instrument} at station {tie_station}")

    ref_value = ref_rows["relative_gravity_mgal"].mean()

    for instrument in df["instrument"].unique():
        if instrument == reference_instrument:
            continue

        tie_rows = df[
            (df["instrument"] == instrument) &
            (df["station"] == tie_station)
        ]

        if tie_rows.empty:
            print(f"Warning: no tie station found for {instrument}. No offset applied.")
            continue

        inst_value = tie_rows["relative_gravity_mgal"].mean()
        offset = ref_value - inst_value

        mask = df["instrument"] == instrument
        df.loc[mask, "instrument_tie_offset_mgal"] = offset
        df.loc[mask, "gravity_tied_mgal"] = df.loc[mask, "relative_gravity_mgal"] + offset

    return df

def read_gps_elevations(gps_csv_path):
    """
    Reads GPS CSV with columns:
    station, Longitude, Latitude, elevation_m
    """
    gps = pd.read_csv(gps_csv_path)

    gps_elev = gps[["station", "Longitude", "Latitude", "elevation_m"]].copy()

    gps_elev = gps_elev.rename(
        columns={
            "Longitude": "longitude",
            "Latitude": "latitude",
        }
    )

    gps_elev["station"] = pd.to_numeric(gps_elev["station"], errors="coerce")
    gps_elev["elevation_m"] = pd.to_numeric(gps_elev["elevation_m"], errors="coerce")

    gps_elev = gps_elev.dropna(subset=["station", "elevation_m"])
    gps_elev = gps_elev.sort_values("station").reset_index(drop=True)

    return gps_elev


def apply_free_air_and_bouguer(
    gravity_gps_df,
    gravity_col="gravity_tied_mgal",
    elevation_col="elevation_m",
    density_kg_m3=2670,
    reference_elevation=None,
):
    """
    Applies free-air and Bouguer slab corrections.

    FAC = +0.3086 * h
    BC  =  0.04193 * rho * h

    where:
    h = elevation relative to reference elevation, in meters
    rho = density in g/cm^3

    Bouguer-corrected gravity:
    g_corr = g_obs + FAC - BC
    """
    df = gravity_gps_df.copy()

    if reference_elevation is None:
        reference_elevation = df[elevation_col].min()

    rho_g_cm3 = density_kg_m3 / 1000.0

    df["reference_elevation_m"] = reference_elevation
    df["height_above_reference_m"] = df[elevation_col] - reference_elevation

    df["free_air_correction_mgal"] = (
        0.3086 * df["height_above_reference_m"]
    )

    df["bouguer_correction_mgal"] = (
        0.04193 * rho_g_cm3 * df["height_above_reference_m"]
    )

    df["free_air_gravity_mgal"] = (
        df[gravity_col] + df["free_air_correction_mgal"]
    )

    df["bouguer_gravity_mgal"] = (
        df[gravity_col]
        + df["free_air_correction_mgal"]
        - df["bouguer_correction_mgal"]
    )

    return df

def merge_gravity_with_gps(tied_profile, gps_csv_path):
    """
    Merges instrument-tied gravity data with GPS elevation data.

    Required GPS columns:
    station, Longitude, Latitude, elevation_m
    """
    gps_elev = read_gps_elevations(gps_csv_path)

    merged = tied_profile.merge(
        gps_elev,
        on="station",
        how="left"
    )

    missing = merged[merged["elevation_m"].isna()]["station"].tolist()

    if len(missing) > 0:
        print("Warning: missing GPS elevation for these stations:")
        print(missing)

    return merged, gps_elev