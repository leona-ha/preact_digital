# ---
# jupyter:
#   jupytext:
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

# %%
import matplotlib.pyplot as plt

# %%
from pyprojroot import (
    here,
)  # define relative paths to the project root (working directory)
from pathlib import Path
import sys
from datetime import date
import pandas as pd
import gc
import os
import glob
import numpy as np
import pickle
import plotly.express as px

# --- Paths / imports -------------------------------------------------

# relative project root
PROJECT_ROOT = (
    here()
)  # '.here' is located as invisible file in the project root working directory
PREPROCESSING_DIR = PROJECT_ROOT / "functions" / "preprocessing"
for p in (PROJECT_ROOT, PREPROCESSING_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

# from server_config import datapath, proj_sheet, preprocessed_path, raw_path#, backup_path
datapath = "/sc-projects/sc-proj-cc15-preact/SP6"
raw_path = datapath + "/raw"

from functions.preprocessing.infer_timeoffset import (
    create_utcday_tzoffset_df,
    merge_fill_tz,
)

# --- Dates ------------------------------------------------------------
today_str = date.today().strftime("%d%m%Y")
today_day = pd.Timestamp.today().normalize()
today_str = "18052026"

# --- Path -------------------------------------------------------------

datapath = Path(raw_path) / f"export_tiki_{today_str}"
# print("!!!!!!OVERRIDE!!!!!!")
# datapath = Path(raw_path) / "tiki_backup_files" / f"export_tiki_31032026"
print(datapath)

# %%
# actual passive + ema_data

# define the pattern for passive data files
file_pattern = os.path.join(datapath, "epoch_part*.csv")

# use glob to find all matching files
file_list = glob.glob(file_pattern)

# sort the file list for consistent ordering
file_list.sort()

print(len(file_list), "files found")

# %%

# concatenate all CSV files into a single DataFrame
df_complete = pd.concat(
    (pd.read_csv(f, encoding="latin-1", low_memory=False) for f in file_list),
    ignore_index=True,
)


# %%
print(len(file_list), "files read and concatenated into a single DataFrame")
file_list

# %%
df_complete.head()

# %%
customers = df_complete["customer"].unique()
customers.sort()
print(customers)

# %%
customer_t = "C7jp4KtfPgrmyRah"
customer_l = "KlerjVsK5Z9AoWnZ"
"KlerjVsK5Z9AoWnZ"
df_tl = df_complete[df_complete.customer.isin([customer_t, customer_l])]
# df = df_complete[df_complete.customer.isin([customer_l])]
print(df_tl.shape)
df_tl = df_tl[df_tl["type"].isin(["Latitude", "Longitude"])]
print(df_tl.shape)

# %%
df_l = df_complete[df_complete["customer"] == customer_l]
df_t = df_complete[df_complete["customer"] == customer_t]
print(df_l.shape, df_t.shape)

# %%
df_tl.customer.unique()

# %%
plot_df = df_tl.copy()
plot_df["participant"] = plot_df["customer"].map({customer_t: "T", customer_l: "L"})

x = pd.to_datetime(plot_df.startTimestamp, unit="ms")
y = np.ones(len(x)) + np.random.normal(0, 0.1, size=len(x))  # add some jitter for better visualization
fig = px.scatter(
    plot_df,
    x=x,
    y=y,
    color="participant",
    color_discrete_map={"T": "#1f77b4", "L": "#ff7f0e"},
    title="Timestamps of Latitude and Longitude entries for T and L",
)
fig.show()

# %%
df_tl

# %%
df_tl["timestamp_start"] = pd.to_datetime(df_tl["startTimestamp"], unit="ms", utc=True)
df_tl["local_timestamp_start"] = df_tl["timestamp_start"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
df_tl["float_value"] = df_tl["doubleValue"]
df_tl.rename(columns={"customer": "id", "type": "modality"}, inplace=True)

# %%
# df_gps_raw = df_backup[df_backup["modality"].isin(["Latitude", "Longitude"])].copy()

lat = (
    df_tl.loc[
        df_tl["modality"].eq("Latitude"),
        ["id", "timestamp_start", "local_timestamp_start", "float_value"],
    ]
    .rename(columns={"float_value": "Latitude"})
    .copy()
)
lon = (
    df_tl.loc[
        df_tl["modality"].eq("Longitude"),
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
df_loc = df_loc_all
df_loc = df_loc.sort_values(by=["id", "timestamp_start"])


# %%
df_loc["time_to_next_clipped"].mean(), df_loc["time_to_next_clipped"].median()

# %%
df_loc["dist_to_next_sample"].plot.hist(bins=np.arange(0, 100_000, 100))
plt.yscale("log")
plt.show()

# %%
plt.scatter(
    df_loc["time_to_next_sample"], df_loc["dist_to_next_sample"], alpha=0.2, s=1
)
plt.yscale("log")
plt.xscale("log")
plt.hlines(
    [1000, 30_000],
    xmin=df_loc["time_to_next_sample"].min(),
    xmax=df_loc["time_to_next_sample"].max(),
    colors="red",
    linestyles="dashed",
)
plt.vlines(
    [60, 10 * 60, 3600, 12 * 3600, 24 * 3600],
    ymin=df_loc["dist_to_next_sample"].min(),
    ymax=df_loc["dist_to_next_sample"].max(),
    colors="blue",
    linestyles="dashed",
)
plt.xlabel("Time to next sample (clipped, log scale)")
plt.ylabel("Distance to next sample (log scale)")

# %%
earth_radius = 6_371_000 # in meters
df_loc["lat_rad"] = np.deg2rad(df_loc["Latitude"])
df_loc["lon_rad"] = np.deg2rad(df_loc["Longitude"])
lat = df_loc["lat_rad"]
lon = df_loc["lon_rad"]
df_loc["x"] = earth_radius * np.cos(lat) * np.cos(lon)
df_loc["y"] = earth_radius * np.cos(lat) * np.sin(lon)
df_loc["z"] = earth_radius * np.sin(lat)



# %%
def haversine_distance_to_next(df, groupby="id"):
    earth_radius = 6_371_000  # in meterss
    df.sort_values(by=[groupby, "timestamp_start"]).copy()
    df["lat_rad"] = np.deg2rad(df["Latitude"])
    df["lon_rad"] = np.deg2rad(df["Longitude"])
    gby = df.groupby(groupby)

    dlat = -gby["lat_rad"].diff(-1)
    dlon = -gby["lon_rad"].diff(-1)

    lat = gby["lat_rad"].shift(0)
    lat_next = gby["lat_rad"].shift(-1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat_next) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return earth_radius * c


# %%
df_loc["lat_rad"].shift(1)

# %%
# compute also the distance to the next sample from x y z

df_loc_groupby_id = df_loc.groupby("id", observed=True)
# df_loc["dist_to_next_sample"] = np.sqrt(
#     df_loc
# )
df_loc["dist_to_next_sample"] = np.linalg.norm(
    -df_loc_groupby_id[["x", "y", "z"]].diff(-1), axis=1
)

# 4. Calculate the final distance
df_loc["haversine_dist_to_next_sample"] = haversine_distance_to_next(df_loc)

df_loc["time_to_next_sample"] = (
    -df_loc_groupby_id["timestamp_start"].diff(-1)
).dt.total_seconds()


df_loc["time_to_next_clipped"] = df_loc["time_to_next_sample"].clip(upper=12 * 3600)
df_loc["time_to_next_clipped"] = df_loc["time_to_next_clipped"].fillna(0)
df_loc["time_to_next_sample_hours"] = np.round(df_loc["time_to_next_sample"] / 3600, 2)

df_loc["is_next_sample_far"] = (
    df_loc["dist_to_next_sample"] > 1000
)  # example threshold of 1000 meters

df_loc["sample_weight"] = np.where(
    df_loc["is_next_sample_far"],
    df_loc["time_to_next_clipped"].clip(upper=60),
    df_loc["time_to_next_clipped"],
)

print(df_loc.shape)

df_loc

# %%
df_loc["dist_to_next_sample"].describe()

# %%
(df_loc["haversine_dist_to_next_sample"] - df_loc["dist_to_next_sample"]).describe()

# %%
np.cos(df_loc_groupby_id["Latitude"].shift(0))

# %% [markdown]
#

# %%
df_loc["speed"] = df_loc["dist_to_next_sample"] / df_loc["time_to_next_sample"]

# %%
df_loc["speed"].describe()
np.log10(df_loc["speed"].replace(0, 1e-8)).plot.hist(bins=100)

# %%
from sklearn.cluster import DBSCAN, HDBSCAN
# import fast_hdbscan
import plotly.graph_objects as go


# %%
df = df_loc[df_loc["id"] == "C7jp4KtfPgrmyRah"]
print("Taking only T data, as there's no L's data in the export.")

is_only_two_weeks = False
if is_only_two_weeks:
    print("Filtering to only two weeks of labeled data.")
    df = df[
        (df["local_timestamp_start"] > pd.Timestamp("2026-03-23"))
        & (df["local_timestamp_start"] < pd.Timestamp("2026-04-07"))
    ]
df = df.sort_values(by="timestamp_start")

# %%
df.head()

# %%
points_3d = df[["x", "y", "z"]]

# 1. Center of mass
C = np.mean(points_3d, axis=0)

# 2. Normal vector (normalized)
n = C / np.linalg.norm(C)

# 3. Create U and V basis vectors for the 2D plane
up = np.array([0.0, 0.0, 1.0])
if np.abs(np.dot(up, n)) > 0.999:
    up = np.array([0.0, 1.0, 0.0])

u = np.cross(up, n)
u = u / np.linalg.norm(u)
v = np.cross(n, u)

# 4. Center the points by subtracting the centroid
centered_points = points_3d - C

# 5. Project onto the 2D plane
x_2d = centered_points @ u
y_2d = centered_points @ v

df["x_flat"] = x_2d
df["y_flat"] = y_2d

# %%
import time

# fhdb = fast_hdbscan.HDBSCAN(
#     min_cluster_size=1 * 3600, min_samples=1000, metric="euclidean"
# )
db = DBSCAN(eps=100, min_samples=2 * 3600, metric="euclidean", algorithm="ball_tree")
db_haversine = DBSCAN(
    eps=100 / earth_radius,
    min_samples=2 * 3600,
    metric="haversine",
    algorithm="ball_tree",
)

# force C order
# cluster_labels = fhdb.fit_predict(
#     np.ascontiguousarray(df[["x", "y", "z"]]),
#     sample_weight=df["time_to_next_clipped"],
# )

t0 = time.time()
cluster_labels = db.fit_predict(
    df[["x", "y", "z"]],
    sample_weight=df["sample_weight"],
)
t1 = time.time()
db_euclidean_time = t1 - t0

rads = np.deg2rad(df[["Longitude", "Latitude"]])
t0 = time.time()
cluster_labels_haversine = db_haversine.fit_predict(
    rads,
    sample_weight=df["sample_weight"],
)
t1 = time.time()
db_haversine_time = t1 - t0

print(f"DBSCAN with Euclidean metric took {db_euclidean_time:.2f} seconds")
print(f"DBSCAN with Haversine metric took {db_haversine_time:.2f} seconds")

df["cluster"] = cluster_labels
df["cluster_haversine"] = cluster_labels_haversine
print(df["cluster"].value_counts(), df["cluster_haversine"].value_counts())

# %%
pd.Series(cluster_labels).value_counts(), pd.Series(cluster_labels_haversine).value_counts()

# %%
plot_df

# %%
# plot 2d
# show also local_timestamp_start as hover data
plot_df = df[
      (df["local_timestamp_start"] > pd.Timestamp("2026-03-28"))
    & (df["local_timestamp_start"] < pd.Timestamp("2026-03-28T23:59"))
].copy()
# plot_df = df.copy()
print(plot_df.shape)

fig = px.scatter(
    plot_df,
    x="x_flat",
    y="y_flat",
    color=plot_df["cluster"].astype(str),  # convert to string for discrete coloring
    title="DBSCAN Clustering of GPS Data (Customer C7jp4KtfPgrmyRah) - 2D",
    size=np.clip(plot_df["time_to_next_clipped"] / 3600, 0.5, 12),
    hover_data={
        "local_timestamp_start": True,
        "dist_to_next_sample": True,
        "time_to_next_clipped": True,
        "time_to_next_sample_hours": True,
        "speed": True,
        "sample_weight": True,
    },
    opacity=0.7,
)
fig.add_trace(
    go.Scatter(
        x=plot_df["x_flat"],
        y=plot_df["y_flat"],
        mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        name="Chronological Path",
        hoverinfo="skip",
        showlegend=False,
    )
)

# aspect ratio 1:1
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_layout(width=600, height=600)
# TODO add legend in the plot about what the size means
fig.show()

# %%
plot_df
# TODO alternative home cluster selection method - weighting the long stays

# %%
print(df.groupby('local_day')["local_hour"].max())

# %%
# count the number of samples per local day
print(df["local_day"].value_counts().sort_index())

# %%

# %%

# %% [markdown]
# ### home cluster estimation

# %%
# "sample weight" is the assumed time duration that a sample represents
df["assumed_duration"] = df["sample_weight"]
df["assumed_local_endtime"] = df["local_timestamp_start"] + pd.to_timedelta(df["assumed_duration"], unit="s")

start_night_hour = 23
end_night_hour = 6

# %%
import datetime

def compute_row_night_time(row):
    # Ensure inputs are parsed as pandas Timestamps
    start = pd.to_datetime(row['local_timestamp_start'])
    end = pd.to_datetime(row['assumed_local_endtime'])
    
    # Safety check for missing or invalid intervals
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.Timedelta(0)
    
    total_night_duration = pd.Timedelta(0)
    
    # Look back 1 day prior to start date to capture stays 
    # that occur during the early morning window (00:00 - 06:00)
    current_date = start.date() - pd.Timedelta(days=1)
    # current_date = start.date() - datetime.timedelta(days=1)
    end_date = end.date()
    
    while current_date <= end_date:
        # Define the night window for the current iteration
        night_start = pd.Timestamp.combine(current_date, datetime.time(23, 0))
        night_end = pd.Timestamp.combine(current_date + datetime.timedelta(days=1), datetime.time(6, 0))
        
        # Calculate the intersection of the stay interval and the night window
        overlap_start = max(start, night_start)
        overlap_end = min(end, night_end)
        
        if overlap_start < overlap_end:
            total_night_duration += (overlap_end - overlap_start)
            
        current_date += datetime.timedelta(days=1)
        
    return total_night_duration


# %%
df["night_duration"] = df.apply(compute_row_night_time, axis=1)

# %%
df.groupby(["id", "cluster"])["night_duration"].sum()

# %%
# 1. Get the total night duration for every id and cluster combination
# (Using reset_index() makes it easier to work with the columns)
cluster_summary = df.groupby(["id", "cluster"])["night_duration"].sum().reset_index()

# 2. Pick the cluster with the maximum night duration for each 'id'
home_clusters = cluster_summary.loc[cluster_summary.groupby("id")["night_duration"].idxmax()]

# Rename the column for clarity (Optional)
home_clusters = home_clusters.rename(columns={"cluster": "home_cluster"})

print(home_clusters[["id", "home_cluster", "night_duration"]])
