# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: tiki13
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aggregate GPS data

# %%
# %%
import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# If your current working directory is the notebooks directory, use this:
notebook_dir = os.getcwd()  # current working directory
src_path = os.path.abspath(os.path.join(notebook_dir, "..", "src"))
sys.path.append(src_path)


# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(notebook_dir, ".."))
sys.path.append(parent_dir)
import pickle
from server_config import (
    datapath,
    preprocessed_path_freezed,
    redcap_path,
    preprocessed_path,
)

from functions.preprocessing import gps_features

# %% [markdown]
# ## Load Data

# %%
backup_path = (
    "/sc-projects/sc-proj-cc15-preact/SP6/raw/backup_passive_recent.feather"
)
df_backup = pd.read_feather(backup_path)
print(f"Loaded backup data shape: {df_backup.shape}")

# %%
debug = True  # Set to True for fast debug runs, False for full dataset runs
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON!!!!!!!!!!!!!")
    # Take a small subset of participants for fast debugging
    selected_participants = ["4MLe", "5nNG"]
    df_backup = df_backup[df_backup["id"].isin(selected_participants)].copy()
    print(f"Selected participants for debugging: {selected_participants}")
    print(f"Data shape after filtering for debug: {df_backup.shape}")
else:
    dbscan_crash_participants = ["UiPj"]
    df_backup = df_backup[~df_backup["id"].isin(dbscan_crash_participants)].copy()
    print("Processing all participants except dbscan crash participants")
    print(f"Discarded participants: {dbscan_crash_participants}")


# %% [markdown]
# ## GPS

# %%
# #TODO remove this once we have the new file with local_timestamp_start
# if "local_timestamp_start" not in df_backup.columns:
#     df_backup["local_timestamp_start"] = df_backup["timestamp_start"].dt.tz_convert("Europe/Berlin")
#     print("!"*70)
#     print("local_timestamp_start column created from timestamp_start with timezone conversion to Europe/Berlin")
#     print("!"*70)

# %%
df_backup.id.unique()

# %%
df_gps_raw = df_backup[df_backup["modality"].isin(["Latitude", "Longitude"])].copy()

lat = (
    df_gps_raw.loc[
        df_gps_raw["modality"].eq("Latitude"),
        ["id", "timestamp_start", "local_timestamp_start", "float_value"],
    ]
    .rename(columns={"float_value": "Latitude"})
    .copy()
)
lon = (
    df_gps_raw.loc[
        df_gps_raw["modality"].eq("Longitude"),
        ["id", "timestamp_start", "local_timestamp_start", "float_value"],
    ]
    .rename(columns={"float_value": "Longitude"})
    .copy()
)

df_loc_all = lat.merge(
    lon, on=["id", "timestamp_start", "local_timestamp_start"], how="inner"
)
df_loc_all = df_loc_all.dropna(
    subset=["Latitude", "Longitude", "local_timestamp_start"]
)

df_loc_all["local_day"] = df_loc_all["local_timestamp_start"].dt.floor("d")
df_loc_all["local_hour"] = df_loc_all["local_timestamp_start"].dt.hour


# %%
# # Restrict to each participant's first two weeks of GPS data
# first_obs_per_customer = df_loc_all.groupby("id", observed=True)["local_timestamp_start"].transform("min")
# df_loc = df_loc_all[
#     df_loc_all["local_timestamp_start"] < (first_obs_per_customer + pd.Timedelta(days=30))
# ].copy()

# # Restrict to each participant's first year for now
first_obs_per_customer = df_loc_all.groupby("id", observed=True)[
    "local_timestamp_start"
].transform("min")
df_loc = df_loc_all[
    df_loc_all["local_timestamp_start"]
    < (first_obs_per_customer + pd.Timedelta(days=365))
].copy()


# df_loc = df_loc_all.copy()
# df_loc
# %% [markdown]
# #### calculate the time to next sample

# %%
df_loc = df_loc.sort_values(by=["id", "timestamp_start"])
df_loc["time_to_next_sample"] = (
    df_loc.groupby("id", observed=True)["timestamp_start"].shift(-1)
    - df_loc["timestamp_start"]
).dt.total_seconds()
print(df_loc.shape)
df_loc.tail()

# %%
# clip
df_loc["time_to_next_clipped"] = df_loc["time_to_next_sample"].clip(upper=12 * 3600)



# %%
gps_raw_points = (
    df_loc.groupby(["id", "local_day"], observed=True)
    .size()
    .reset_index(name="GPS_raw_points")
)

gps_hour_counts = (
    df_loc.groupby(["id", "local_day", "local_hour"], observed=True)
    .size()
    .reset_index(name="points_in_hour")
)

gps_hourly_summary = (
    gps_hour_counts.groupby(["id", "local_day"], observed=True)
    .agg(
        GPS_hours_with_data=("local_hour", "nunique"),
        GPS_points_per_hour_mean=("points_in_hour", "mean"),
        GPS_points_per_hour_median=("points_in_hour", "median"),
        GPS_points_per_hour_std=("points_in_hour", "std"),
    )
    .reset_index()
)


def _gap_stats(group: pd.DataFrame) -> pd.Series:
    # times = group["local_timestamp_start"].sort_values()
    times = group["timestamp_start"].sort_values()
    if len(times) < 2:
        return pd.Series(
            {
                "GPS_max_gap_seconds": np.nan,
                "GPS_mean_gap_seconds": np.nan,
            }
        )
    gaps = times.diff().dt.total_seconds().dropna()
    return pd.Series(
        {
            "GPS_max_gap_seconds": gaps.max(),
            "GPS_mean_gap_seconds": gaps.mean(),
        }
    )


gps_gap_stats = (
    df_loc.groupby(["id", "local_day"], observed=True).apply(_gap_stats).reset_index()
)

df_gps_coverage = gps_raw_points.merge(
    gps_hourly_summary, on=["id", "local_day"], how="outer"
).merge(gps_gap_stats, on=["id", "local_day"], how="outer")

df_gps_coverage["GPS_coverage_pct"] = (
    df_gps_coverage["GPS_hours_with_data"] / 24.0 * 100.0
)

# %%
df_gps_coverage

# %%
logging.info("GPS coverage metrics calculated and merged into df_gps_coverage")
print(df_gps_coverage.shape)
df_gps_coverage.head()
# logging.info(f"GPS coverage metrics calculated for {df_gps_coverage['customer'].nunique()} customers and {len(df_gps_coverage)} customer-days")
# %%
df_loc.head()

# %%
df_loc.groupby("id", observed=True).size().sort_values(ascending=False)

# %%
df_loc

# %% [markdown]
# ### gps_features cluster & home extract

# %%
home_extractor = gps_features.HomeClusterExtractor(
    df_loc,
    speed_limit=1.4,
    max_distance=150,
    epsilon=100,  # Müller uses 30
    min_samples=10,  # Müller uses 3 but we have more data
    min_nights_obs=4,
    min_f_home=0.5,
    clustering_method="dbscan",
    normalize_min_samples=False,
    min_data_points=50,
    # n_jobs=64,
    n_jobs=16,
    # new_cluster_labeling=True,  # ensures cluster labels are consistent across days for the same participant
    use_weights=True,
    home_v2=True,
)

geodata_clusters = home_extractor.run()
logging.info("Home cluster extraction completed")

df_gps_metrics = gps_features.calculate_metrics(
    geodata_clusters, group_by=["id", "local_day"]
)
logging.info("GPS metrics calculated and stored in df_gps_metrics")

df_gps_daily = df_gps_metrics

print(f"GPS daily rows: {len(df_gps_daily)}")
print(f"GPS daily columns: {len(df_gps_daily.columns)}")
print(f"Unique customers: {df_gps_daily['id'].nunique()}")
print(
    f"Date range: {df_gps_daily['local_day'].min()} to {df_gps_daily['local_day'].max()}"
)
df_gps_daily.head()

# %%
assert np.all(
    (home_extractor.df["transition"] == 1) | (home_extractor.df["stationary"])
)
assert not np.any(
    (home_extractor.df["transition"] == 1) & (home_extractor.df["stationary"])
)

# %%
# save the aggregated daily data
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON: Saving locally inside the repository!!!!!!!!!!!!!")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    output_path = output_dir / "temp_gps_daily_metrics_debug.feather"
else:
    passive_daily_dir = Path("/sc-projects/sc-proj-cc15-preact/SP6/preprocessed/passive/daily/")
    output_path = passive_daily_dir / "daily_gps_metrics.feather"

df_gps_daily.to_feather(output_path)
print(f"GPS daily metrics saved to: {output_path}")

# %% [markdown]
# ## Summary Visualizations

# %%
# Plot non-null counts of each key metric in df_gps_daily
key_metrics = [
    "total_distance_km", "n_unique_places", "at_home_minute", "entropy"
]
available_metrics = [m for m in key_metrics if m in df_gps_daily.columns]
if available_metrics:
    plt.figure(figsize=(10, 5))
    df_gps_daily[available_metrics].notna().sum().plot(kind="bar", color="#2ecc71", edgecolor="black")
    plt.title("Number of Day-Records containing each GPS Metric")
    plt.ylabel("Non-Null Count")
    plt.xlabel("GPS Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

