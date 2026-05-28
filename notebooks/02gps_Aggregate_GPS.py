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
import os
import sys
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
# import cProfile


# %load_ext autoreload
# %autoreload 2
import logging

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
from functions.preprocessing.ema_mappings import clean_heart_rate_data
from functions.preprocessing.aggregation import compute_sleep_sessions

from functions.preprocessing import gps_features
from functions.preprocessing.ema_mappings import run_ema_mappings
from functions.preprocessing.missing_data import summarize_missing_data


# %% [markdown]
# ## Load Data

# %%
# backup_path = preprocessed_path_freezed + "/backup_data_passive_actual.feather"
backup_path = (
    "/sc-projects/sc-proj-cc15-preact/SP6/raw/backup_passive_recent.feather"  # new file
)
df_backup = pd.read_feather(backup_path)
print(df_backup.shape)
df_backup.head()

# %%
df_backup.id.unique()

# %%
debug = True
# debug = False
if debug:  # take subset of participants
    # filter to random subset of participants
    unique_participants = df_backup["id"].unique()
    print(f"Total unique participants: {len(unique_participants)}")

    # np.random.seed(42)  # For reproducibility
    # selected_participants = np.random.choice(unique_participants, size=100, replace=False)
    # selected_participants = ["UiPj"]
    selected_participants = ["Cclz", "uxml"]
    # selected_participants = ["1BAf", "KWso"]
    # selected_participants = ["1BAf", "KWso", "gN1w", "uxmL"]

    selected_participants = ["4MLe", "5nNG"]

    print(f"Selected participants: {selected_participants}")

    # Filter the DataFrame to include only the selected participants
    df_backup = df_backup[df_backup["id"].isin(selected_participants)]
else:
    dbscan_crash_paricipants = [
        "UiPj",
    ]
    df_backup = df_backup[~df_backup["id"].isin(dbscan_crash_paricipants)]
    print("process all participants except dbscan crash participants")
    print(f"discarded participants: {dbscan_crash_paricipants}")

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

bins = np.arange(0, 72, 0.5)
(df_loc["time_to_next_sample"] / 3600).plot.hist(bins=bins, alpha=0.5, label="original")
(df_loc["time_to_next_clipped"] / 3600).plot.hist(bins=bins, alpha=0.5, label="clipped")

plt.yscale("log")
plt.vlines([24], 1, 1e8, colors="red", linestyles="dashed")
# sns.histplot(df_loc["time_to_next_clipped"] / 60, bins=50, kde=True)
plt.xlabel("Time to next GPS sample (hours)")


# %%
bins = np.arange(0, 70, 0.5)
(df_loc["time_to_next_sample"] / 60).plot.hist(bins=bins, alpha=0.5, label="original")
plt.yscale("log")
plt.xlabel("Time to next GPS sample (minutes)")

# %%
bins = np.arange(0, 310, 1)
(df_loc["time_to_next_sample"]).plot.hist(bins=bins, alpha=0.5, label="original")
plt.yscale("log")
plt.xlabel("Time to next GPS sample (seconds)")

# %%
bins = np.logspace(np.log2(1), np.log2(24 * 3600), num=100, base=2)
bins

# %%
bins = np.arange(0, 24 * 3600, 1)
bins = np.logspace(np.log10(1), np.log10(24 * 3600), num=100, base=10)
(df_loc["time_to_next_sample"]).plot.hist(bins=bins, alpha=0.5, label="original")
plt.xlabel("Time to next GPS sample (seconds)")
plt.xscale("log")
plt.yscale("log")

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
# ### TODO remove this section later

# %%
## TODO remove this
from sklearn.cluster import DBSCAN, HDBSCAN
import fast_hdbscan

df = df_loc[df_loc["id"] == df_loc["id"].unique()[0]].copy()


# %%
import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Compact helper functions for Spherical <-> Cartesian conversion
to_3d = lambda lat, lon: np.c_[
    np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
    np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
    np.sin(np.radians(lat)),
]
to_2d = lambda v: (
    np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1))),
    np.degrees(np.arctan2(v[:, 1], v[:, 0])),
)

# 2. Find the spatial mode (rounding to 2 decimals to group nearby points)
mode_lat, mode_lon = df[["Latitude", "Longitude"]].round(2).value_counts().idxmax()

# 3. Calculate the rigid 3D rotation from the Mode to Berlin
rot, _ = R.align_vectors(to_3d([52.52], [13.405]), to_3d([mode_lat], [mode_lon]))

# 4. Apply the rotation to the whole dataset and map back to Lat/Lon columns
df["Berlin_Lat"], df["Berlin_Lon"] = to_2d(
    rot.apply(to_3d(df["Latitude"], df["Longitude"]))
)

df["Latitude"], df["Longitude"] = df["Berlin_Lat"], df["Berlin_Lon"]

# %%
# clustering_model = DBSCAN(eps=100 / 6371000, min_samples=60, metric="haversine")
clustering_model = fast_hdbscan.HDBSCAN(min_cluster_size=60, metric="haversine")
# clustering_model = HDBSCAN(min_cluster_size=60, metric="haversine")
cluster_labels = clustering_model.fit_predict(
    np.radians(df[["Latitude", "Longitude"]]),
    sample_weight=df["time_to_next_clipped"].fillna(0) / 60,
)
cluster_labels
# remap the labels to be consecutive
# shuffle cluster labels, but not -1
unique_labels = np.unique(cluster_labels)
unique_labels = unique_labels[unique_labels != -1]  # Exclude noise label
print(f"Unique cluster labels after excluding noise: {unique_labels}")
print(f"{len(unique_labels)} unique labels after excluding noise")
new_unique_labels = np.arange(len(unique_labels))
np.random.seed(42)
np.random.shuffle(unique_labels)
label_mapping = dict(zip(unique_labels, new_unique_labels))
label_mapping[-1] = -1  # Ensure noise label remains -1
cluster_labels_mapped = np.array([label_mapping[label] for label in cluster_labels])
cluster_labels = cluster_labels_mapped

# %%
cluster_labels

# %%
cluster_labels.min(), cluster_labels.max()

# %%
df

# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px


# Plot cluster label counts
label_counts = pd.Series(cluster_labels, name="cluster").value_counts().sort_index()

plt.figure(figsize=(7, 4))
label_counts.plot(kind="bar")
plt.title("Cluster Label Counts")
plt.xlabel("Cluster label")
plt.ylabel("Number of points")
plt.tight_layout()
plt.show()

# %%
df

# %%
import plotly.express as px
import plotly.colors as pc

# 1. Prepare the data for discrete coloring
# Converting to string ensures Plotly treats them as distinct categories (legend entries)
plot_df = df[["Latitude", "Longitude", "timestamp_start", "time_to_next_sample"]].copy()
plot_df["cluster_label"] = cluster_labels

# 2. Build the Custom Color Map
unique_clusters = sorted([c for c in plot_df["cluster_label"].unique() if c != -1])

# Generate Turbo colors for the actual clusters
# We sample N colors from the Turbo scale
turbo_colors = px.colors.sample_colorscale("Turbo", len(unique_clusters))

# Construct the map: -1 is Gray, others are Turbo
color_map = {"-1": "rgba(200, 200, 200, 0.5)"}  # Light Gray with transparency
for label, color in zip(unique_clusters, turbo_colors):
    color_map[str(label)] = color

# 3. Create the Interactive Plot
fig = px.scatter(
    plot_df,
    x="Longitude",
    y="Latitude",
    size=np.clip(
        plot_df["time_to_next_sample"].fillna(0) / 60, 2, 60
    ),  # Size by time to next sample (in minutes)
    color=plot_df["cluster_label"].astype(str),  # Ensure it's treated as categorical
    color_discrete_map=color_map,
    category_orders={"cluster_label": ["-1"] + [str(c) for c in unique_clusters]},
    title="Interactive Cluster Map",
    labels={"cluster_label": "Cluster ID"},
    hover_data={
        "cluster_label": True,
        "Latitude": True,
        "Longitude": True,
        "timestamp_start": True,
        "time_to_next_sample": True,
    },
    opacity=0.7,
    render_mode="webgl",  # Use WebGL for high performance if you have many points
)


fig.show()

# %%
import plotly.express as px
import numpy as np

# 1. Prepare the data
plot_df = df[
    ["Berlin_Lat", "Berlin_Lon", "timestamp_start", "time_to_next_sample"]
].copy()
plot_df["cluster_label"] = cluster_labels

# --- THE NO-DEPENDENCY PROJECTION TO KILOMETERS ---
# Find the center of your rotated data
center_lat = plot_df["Berlin_Lat"].mean()
center_lon = plot_df["Berlin_Lon"].mean()

# Convert degrees to distance (km) from the center point
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))

plot_df["X_km"] = (plot_df["Berlin_Lon"] - center_lon) * km_per_deg_lon
plot_df["Y_km"] = (plot_df["Berlin_Lat"] - center_lat) * km_per_deg_lat
# --------------------------------------------------

# 2. Build the Custom Color Map
unique_clusters = sorted([c for c in plot_df["cluster_label"].unique() if c != -1])
turbo_colors = px.colors.sample_colorscale("Turbo", len(unique_clusters))

color_map = {"-1": "rgba(200, 200, 200, 0.5)"}
for label, color in zip(unique_clusters, turbo_colors):
    color_map[str(label)] = color

# 3. Create the Interactive Plot
fig = px.scatter(
    plot_df,
    x="X_km",  # Now plotting pure kilometers
    y="Y_km",  # Now plotting pure kilometers
    size=np.clip(
        plot_df["time_to_next_sample"].fillna(0) / 60, 1, 60
    ),  # Size by time to next sample (in minutes)
    color=plot_df["cluster_label"].astype(str),
    color_discrete_map=color_map,
    category_orders={"cluster_label": ["-1"] + [str(c) for c in unique_clusters]},
    title="Interactive Cluster Map (Local Flat Plane in km)",
    labels={
        "cluster_label": "Cluster ID",
        "X_km": "Distance East/West (km)",
        "Y_km": "Distance North/South (km)",
    },
    hover_data={
        "cluster_label": True,
        "Berlin_Lat": ":.4f",
        "Berlin_Lon": ":.4f",
        "X_km": ":.2f",
        "Y_km": ":.2f",
        "timestamp_start": True,
        "time_to_next_sample": True,
    },
    opacity=0.7,
    render_mode="webgl",
)

# CRITICAL: Lock the aspect ratio so 1km X is visually equal to 1km Y
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(width=800, height=800)
fig.show()

# %% [markdown]
# ### gps_features cluster & home extract

# %%
# TODO fine-tune the parameters, rn they're from 03_JITAI notebook
home_extractor = gps_features.HomeClusterExtractor(
    df_loc,
    speed_limit=1.4,
    max_distance=150,
    # epsilon=100 / 6371000,  # 100 / kms_per_radian # TODO Müller uses 30
    epsilon=100,  # 100 / kms_per_radian # TODO Müller uses 30
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
# geodata_clusters["local_day"] = geodata_clusters["local_timestamp_start"].dt.floor("d")
logging.info("Home cluster extraction completed")

df_gps_metrics = gps_features.calculate_metrics(
    geodata_clusters, group_by=["id", "local_day"]
)
logging.info("GPS metrics calculated and stored in df_gps_metrics")

# df_gps_transition = gps_features.calculate_transition_time(
#     geodata_clusters, group_by=["id", "local_day"]
# )

# df_gps_daily = df_gps_metrics.merge(
#     df_gps_transition, on=["id", "local_day"], how="outer"
# ).merge(df_gps_coverage, on=["id", "local_day"], how="outer")
df_gps_daily = df_gps_metrics

print(f"GPS daily rows: {len(df_gps_daily)}")
print(f"GPS daily columns: {len(df_gps_daily.columns)}")
print(f"Unique customers: {df_gps_daily['id'].nunique()}")
print(
    f"Date range: {df_gps_daily['local_day'].min()} to {df_gps_daily['local_day'].max()}"
)
df_gps_daily.head()
# TODO double check the home cluster extraction

# %%
geodata_clusters

# %%

# %%
geodata_clusters

# %%
home_extractor.df["stationary"]

# %%
assert np.all(
    (home_extractor.df["transition"] == 1) | (home_extractor.df["stationary"])
)
assert not np.any(
    (home_extractor.df["transition"] == 1) & (home_extractor.df["stationary"])
)

# %%
df_gps_daily.to_feather("temp_gps_daily_metrics.feather")


# %%
logging.info("GPS daily metrics saved to temp_gps_daily_metrics.feather")

# %%
df_gps_daily = pd.read_feather("temp_gps_daily_metrics.feather")

# %% [markdown]
# ### random checks

# %%
geodata_clusters

# %%
geodata_clusters["cluster"].min()

# %%
geodata_clusters[geodata_clusters["local_day"] != geodata_clusters["day_gps"]]
geodata_clusters[geodata_clusters["local_hour"] != geodata_clusters["hour_gps"]]

# %%
np.any(geodata_clusters["cluster"] != geodata_clusters["clusterID"])

# %%
geodata_clusters

# %% [markdown]
# ## df gps coverage analysis

# %% [markdown]
# ### day availibility

# %%
first_data_day = df_loc_all.groupby(["id"])["local_day"].min()
df = df_loc_all.merge(
    first_data_day, on="id", how="left", suffixes=("", "_first_data_day")
)
df["days_since_first_data"] = (df["local_day"] - df["local_day_first_data_day"]).dt.days
df

# %%
bins = np.arange(0, df["days_since_first_data"].max() + 1, 1)
df["days_since_first_data"].plot.hist(bins=bins)
plt.xlabel("Days Since First GPS Data")
plt.ylabel("Number of records")
# plt.xlim(0, 60)

# %%
df.groupby("days_since_first_data")["id"].nunique().plot()
plt.xlabel("Days Since First GPS Data")
plt.ylabel("Number of Unique Participants")
plt.xlim(0, 60)

# %% [markdown]
# ### within days coverage (during the first 2 weeks)

# %% [markdown]
# ### visualization plan
# 1. Start with a quick distribution view of `GPS_coverage_pct` across all participant-days.
# 2. Plot the daily median and interquartile range of `GPS_coverage_pct` over calendar time.
# 3. Show participant-level average coverage to compare between participants (top 20 by data volume for readability).
# 4. Add compact views of mean gap (`GPS_mean_gap_seconds`) as distribution and daily trend.
# 5. Add compact views of max gap (`GPS_max_gap_seconds`) as distribution and daily trend.

# %%
df_gps_coverage

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plot_df = df_gps_coverage.copy()
plot_df = plot_df.dropna(
    subset=[
        "GPS_coverage_pct",
        "GPS_mean_gap_seconds",
        "GPS_max_gap_seconds",
        "local_day",
        "id",
    ]
)
plot_df["local_day"] = pd.to_datetime(plot_df["local_day"])
plot_df["GPS_mean_gap_hours"] = plot_df["GPS_mean_gap_seconds"] / 3600
plot_df["GPS_max_gap_hours"] = plot_df["GPS_max_gap_seconds"] / 3600

fig, axes = plt.subplots(1, 7, figsize=(38, 4))

sns.histplot(plot_df["GPS_coverage_pct"], bins=25, kde=True, ax=axes[0])
axes[0].set_title("Coverage distribution")
axes[0].set_xlabel("GPS coverage (%)")

sns.lineplot(
    data=plot_df,
    x="local_day",
    y="GPS_coverage_pct",
    estimator="median",
    errorbar=("pi", 50),
    ax=axes[1],
)
axes[1].set_title("Daily median coverage (IQR)")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("GPS coverage (%)")

participant_mean = (
    plot_df.groupby("id", as_index=False)
    .agg(mean_coverage=("GPS_coverage_pct", "mean"), n_days=("local_day", "nunique"))
    .sort_values("n_days", ascending=False)
    .head(20)
)
participant_mean = participant_mean.sort_values("mean_coverage", ascending=False)

sns.barplot(data=participant_mean, x="mean_coverage", y="id", ax=axes[2])
axes[2].set_title("Participant mean coverage (top 20 by days)")
axes[2].set_xlabel("Mean GPS coverage (%)")
axes[2].set_ylabel("id")


sns.histplot(
    plot_df["GPS_mean_gap_hours"],
    bins=np.linspace(0, 1, 101),
    kde=True,
    ax=axes[3],
    kde_kws={"cut": 0},
)
axes[3].set_title("Mean gap distribution")
axes[3].set_xlabel("Mean gap (hours)")
axes[3].set_xlim(0, 1)

sns.lineplot(
    data=plot_df,
    x="local_day",
    y="GPS_mean_gap_hours",
    estimator="median",
    errorbar=("pi", 50),
    ax=axes[4],
)
axes[4].set_title("Daily median mean-gap (IQR)")
axes[4].set_xlabel("Day")
axes[4].set_ylabel("Mean gap (hours)")

sns.histplot(plot_df["GPS_max_gap_hours"], bins=25, kde=True, ax=axes[5])
axes[5].set_title("Max gap distribution")
axes[5].set_xlabel("Max gap (hours)")

sns.lineplot(
    data=plot_df,
    x="local_day",
    y="GPS_max_gap_hours",
    estimator="median",
    errorbar=("pi", 50),
    ax=axes[6],
)
axes[6].set_title("Daily median max-gap (IQR)")
axes[6].set_xlabel("Day")
axes[6].set_ylabel("Max gap (hours)")

plt.tight_layout()

# %%
# Distribution of average number of GPS records per participant
avg_records_per_participant = (
    df_gps_coverage.groupby("id", observed=True)["GPS_raw_points"]
    .mean()
    .reset_index(name="avg_daily_records")
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(
    avg_records_per_participant["avg_daily_records"], bins=100, kde=True, ax=ax
)
ax.set_title("Distribution of avg daily GPS records per participant")
ax.set_xlabel("Avg daily GPS records")
ax.set_ylabel("Number of participants")
plt.tight_layout()

# %%
