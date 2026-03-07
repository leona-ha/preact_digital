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




# # %load_ext autoreload
# # %autoreload 2
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
debug = False
# debug = False
if debug: # take subset of participants
    # filter to random subset of participants
    unique_participants = df_backup["customer"].unique()
    print(f"Total unique participants: {len(unique_participants)}")

    # np.random.seed(42)  # For reproducibility
    # selected_participants = np.random.choice(unique_participants, size=100, replace=False)
    # selected_participants = ["UiPj"]
    selected_participants = ["Cclz"]
    # selected_participants = ["1BAf", "KWso"]
    # selected_participants = ["1BAf", "KWso", "gN1w", "uxmL"]
    
    print(f"Selected participants: {selected_participants}")

    # Filter the DataFrame to include only the selected participants
    df_backup = df_backup[df_backup["customer"].isin(selected_participants)]
else:
    dbscan_crash_paricipants = ["UiPj",]
    df_backup = df_backup[~df_backup["customer"].isin(dbscan_crash_paricipants)]
    print("process all participants except dbscan crash participants")
    print(f"discarded participants: {dbscan_crash_paricipants}")

# %% [markdown]
# ## GPS

# %%
df_gps_raw = df_backup[df_backup["type"].isin(["Latitude", "Longitude"])].copy()


lat = (
    df_gps_raw.loc[
        df_gps_raw["type"].eq("Latitude"),
        ["customer", "startTimestamp", "local_start_time", "doubleValue"],
    ]
    .rename(columns={"doubleValue": "Latitude"})
    .copy()
)
lon = (
    df_gps_raw.loc[
        df_gps_raw["type"].eq("Longitude"),
        ["customer", "startTimestamp", "local_start_time", "doubleValue"],
    ]
    .rename(columns={"doubleValue": "Longitude"})
    .copy()
)

df_loc_all = lat.merge(
    lon, on=["customer", "startTimestamp", "local_start_time"], how="inner"
)
df_loc_all = df_loc_all.dropna(subset=["Latitude", "Longitude", "local_start_time"])
df_loc_all["local_day"] = df_loc_all["local_start_time"].dt.floor("d")
df_loc_all["local_hour"] = df_loc_all["local_start_time"].dt.hour



# %%
df_loc_all.shape

# %%

# # Restrict to each participant's first two weeks of GPS data
# first_obs_per_customer = df_loc_all.groupby("customer", observed=True)["local_start_time"].transform("min")
# df_loc = df_loc_all[
#     df_loc_all["local_start_time"] < (first_obs_per_customer + pd.Timedelta(days=30))
# ].copy()

# # Restrict to each participant's first year for now
# TODO
first_obs_per_customer = df_loc_all.groupby("customer", observed=True)["local_start_time"].transform("min")
df_loc = df_loc_all[
    df_loc_all["local_start_time"] < (first_obs_per_customer + pd.Timedelta(days=365))
].copy()

# df_loc = df_loc_all.copy()
# df_loc
# %%

gps_raw_points = (
    df_loc.groupby(["customer", "local_day"], observed=True)
    .size()
    .reset_index(name="GPS_raw_points")
)

gps_hour_counts = (
    df_loc.groupby(["customer", "local_day", "local_hour"], observed=True)
    .size()
    .reset_index(name="points_in_hour")
)

gps_hourly_summary = (
    gps_hour_counts.groupby(["customer", "local_day"], observed=True)
    .agg(
        GPS_hours_with_data=("local_hour", "nunique"),
        GPS_points_per_hour_mean=("points_in_hour", "mean"),
        GPS_points_per_hour_median=("points_in_hour", "median"),
        GPS_points_per_hour_std=("points_in_hour", "std"),
    )
    .reset_index()
)


def _gap_stats(group: pd.DataFrame) -> pd.Series:
    times = group["local_start_time"].sort_values()
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
    df_loc.groupby(["customer", "local_day"], observed=True)
    .apply(_gap_stats)
    .reset_index()
)

df_gps_coverage = gps_raw_points.merge(
    gps_hourly_summary, on=["customer", "local_day"], how="outer"
).merge(gps_gap_stats, on=["customer", "local_day"], how="outer")

df_gps_coverage["GPS_coverage_pct"] = (
    df_gps_coverage["GPS_hours_with_data"] / 24.0 * 100.0
)

# %%
logging.info("GPS coverage metrics calculated and merged into df_gps_coverage")
print(df_gps_coverage.shape)
df_gps_coverage.head()
# logging.info(f"GPS coverage metrics calculated for {df_gps_coverage['customer'].nunique()} customers and {len(df_gps_coverage)} customer-days")
# %%
df_loc.head()

# %%
df_loc.groupby("customer", observed=True).size().sort_values(ascending=False)

# %%
# TODO fine-tune the parameters, rn they're from 03_JITAI notebook
home_extractor = gps_features.HomeClusterExtractor(
    df_loc,
    speed_limit=1.4,
    max_distance=150,
    epsilon=100 / 6371000, # 100 / kms_per_radian # TODO Müller uses 30
    min_samples=10, # Müller uses 3 but we have more data
    min_nights_obs=4,
    min_f_home=0.5,
    clustering_method="dbscan",
    normalize_min_samples=False,
    min_data_points=50,
    n_jobs=64,
)

geodata_clusters = home_extractor.run()
# geodata_clusters["local_day"] = geodata_clusters["local_start_time"].dt.floor("d")
logging.info("Home cluster extraction completed")

df_gps_metrics = gps_features.calculate_metrics(
    geodata_clusters, group_by=["customer", "local_day"]
)
logging.info("GPS metrics calculated and stored in df_gps_metrics")

# df_gps_transition = gps_features.calculate_transition_time(
#     geodata_clusters, group_by=["customer", "local_day"]
# )

# df_gps_daily = df_gps_metrics.merge(
#     df_gps_transition, on=["customer", "local_day"], how="outer"
# ).merge(df_gps_coverage, on=["customer", "local_day"], how="outer")
df_gps_daily = df_gps_metrics

print(f"GPS daily rows: {len(df_gps_daily)}")
print(f"GPS daily columns: {len(df_gps_daily.columns)}")
print(f"Unique customers: {df_gps_daily['customer'].nunique()}")
print(
    f"Date range: {df_gps_daily['local_day'].min()} to {df_gps_daily['local_day'].max()}"
)
df_gps_daily.head()
# TODO double check the home cluster extraction

# %%
2+2

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
first_data_day = df_loc_all.groupby(["customer"])["local_day"].min()
df = df_loc_all.merge(first_data_day, on="customer", how="left", suffixes=("", "_first_data_day"))
df["days_since_first_data"] = (df["local_day"] - df["local_day_first_data_day"]).dt.days
df

# %%
bins = np.arange(0, df["days_since_first_data"].max() + 1, 1)
df["days_since_first_data"].plot.hist(bins=bins)
plt.xlabel("Days Since First GPS Data")
plt.ylabel("Number of records")
# plt.xlim(0, 60)

# %%
df.groupby("days_since_first_data")["customer"].nunique().plot()
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
        "customer",
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
    plot_df.groupby("customer", as_index=False)
    .agg(mean_coverage=("GPS_coverage_pct", "mean"), n_days=("local_day", "nunique"))
    .sort_values("n_days", ascending=False)
    .head(20)
)
participant_mean = participant_mean.sort_values("mean_coverage", ascending=False)

sns.barplot(data=participant_mean, x="mean_coverage", y="customer", ax=axes[2])
axes[2].set_title("Participant mean coverage (top 20 by days)")
axes[2].set_xlabel("Mean GPS coverage (%)")
axes[2].set_ylabel("Customer")


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
    df_gps_coverage.groupby("customer", observed=True)["GPS_raw_points"]
    .mean()
    .reset_index(name="avg_daily_records")
)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(avg_records_per_participant["avg_daily_records"], bins=100, kde=True, ax=ax)
ax.set_title("Distribution of avg daily GPS records per participant")
ax.set_xlabel("Avg daily GPS records")
ax.set_ylabel("Number of participants")
plt.tight_layout()

# %%
