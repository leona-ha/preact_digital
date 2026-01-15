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

# %% [markdown]
# # Steps Aggregate

# %%
import os
import sys

import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# %load_ext autoreload
# %autoreload 2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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


# %%
# backup_path = preprocessed_path_freezed + "/backup_data_passive_actual.feather"
backup_path = (
    "/sc-projects/sc-proj-cc15-preact/SP6/raw/backup_passive_recent.feather"  # new file
)
df_backup = pd.read_feather(backup_path)
print(df_backup.shape)
df_backup.head()

# %%
# remove customer with two for_ids for simplicity
# TODO investigate later
customer_with_two_for_ids = "OmAV"
df_backup = df_backup[df_backup["customer"] != customer_with_two_for_ids]

# %%
df_backup.head()

# %% [markdown]
# ## Steps expansion and analysis

# %%
# Similar to HeartRate analysis, expand the Steps dataset
df_steps_original = df_backup[(df_backup["type"] == "Steps")]
df_steps_original = df_steps_original[
    [
        "customer",
        "for_id",
        # "timezoneOffset",
        "startTimestamp",
        "endTimestamp",
        "local_start_time",
        "local_end_time",
        "doubleValue",
    ]
]
df_steps_original = df_steps_original.rename(columns={"doubleValue": "Steps"})
df_steps_original

# %%
df_steps_original["start_end"] = (df_steps_original["endTimestamp"] - df_steps_original["startTimestamp"]).dt.total_seconds()

# %%
df_steps_original["start_end"].max()

# %%
fig, axs = plt.subplots(3, 1, figsize=(16, 20), sharey=True)
# plt.subplot(2, 1, 1)
xs_ends = [24*3600, 700, 130]
df_steps_original["start_end"].plot.hist(
    bins=np.arange(0, 24 * 3600 + 1, 120), log=True, figsize=(10, 6), ax=axs[0]
)
axs[0].set_xlim(0, xs_ends[0])

# plt.subplot(2, 1, 2)
# df_steps_original["start_end"].plot.hist(bins=np.arange(0, 3600 + 1, 1), log=True, figsize=(10, 6), density=False)
df_steps_original["start_end"].plot.hist(
    bins=np.arange(0, 700 + 1, 1), log=True, ax=axs[1]
)
axs[1].set_xlim(0, xs_ends[1])
# df_steps_original["start_end"].plot.hist(bins=np.arange(0, 660 + 1, 1), log=True, figsize=(10, 6), density=False)
# df_steps_original["start_end"].plot.hist(bins=np.arange(0, 610 + 1, 1), log=True, figsize=(10, 6), density=False)
df_steps_original["start_end"].plot.hist(
    bins=np.arange(0, 130 + 1, 1), log=True, figsize=(10, 6), ax=axs[2]
)
axs[2].set_xlim(0, xs_ends[2])
plt.suptitle("Histogram of Steps start_end")
plt.xlabel("Session Duration (seconds)")
# plt.xlim(0, 2000)

# %%
# based on the above -> cutoff sessions at 10 minutes (600 seconds)
cutoff_seconds = 600
print((df_steps_original["start_end"] > cutoff_seconds).sum())
df_steps_original = df_steps_original[df_steps_original["start_end"] <= cutoff_seconds]
df_steps_original.shape

# %%
print(
    f"{(df_steps_original['startTimestamp'].dt.second != 0).sum() / len(df_steps_original):.2%}"
)

# %%
bins = np.arange(0, 61, 1) - 0.5  # 0 to 59 seconds
df_steps_original["startTimestamp"].dt.second.describe(
    [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
)
plt.figure(figsize=(10, 6))
plt.hist(
    df_steps_original["startTimestamp"].dt.second,
    bins=bins,
    alpha=0.5,
    edgecolor="black",
)
plt.hist(
    df_steps_original["endTimestamp"].dt.second, bins=bins, alpha=0.5, edgecolor="black"
)
plt.title("Distribution of Start Timestamp Seconds for Steps Data")
plt.xlabel("Second (0-59)")
plt.ylabel("Frequency")
plt.yscale("log")  # Use logarithmic scale for better visibility
plt.grid(True, alpha=0.3)
plt.xticks(np.arange(0, 60, 5))
plt.tight_layout()
plt.show()

# %% [markdown]
# such a samll number of samples got different offset than 0 (less than 2%), and 99% of samples durations are withing (59,61) range -- more than 90% is exactly 60s
#
# therefore, just round the seconds 

# %%
df_steps_original["duration"] = (
    (df_steps_original["endTimestamp"] - df_steps_original["startTimestamp"])
    .dt.total_seconds()
    .fillna(0)
    .astype(int)
)
df_steps_original["duration"].describe(
    [
        0.003,
        0.004,
        0.005,
        0.0075,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.99,
        0.993,
        0.994,
        0.995,
        0.9975,
        0.999,
    ]
)

# %%

# %%
mask_under_59 = df_steps_original["duration"] < 59
mask_over_61 = df_steps_original["duration"] > 61
print(
    f"Percentage of samples with duration not in [59, 61]: {((mask_under_59 | mask_over_61).sum() / len(df_steps_original)) * 100:.2f}%"
)
df_steps_original["SPM"] = df_steps_original["Steps"] / (
    df_steps_original["duration"] / 60
)  # StepsPerMinute (SPM)
df_steps_original["SPM"].describe(
    [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995]
)

# %%
df_steps_original[mask_over_61]

# %%
df_backup[
    (pd.Timestamp("2024-08-04T10:00", tz="utc") < df_backup["startTimestamp"])
    & (df_backup["startTimestamp"] < pd.Timestamp("2024-08-04T23:30", tz="utc"))
    & (df_backup["customer"] == "yeVc")
]

# %%
df_backup

# %%
df_steps_original.SPM.describe()

# %%
df_steps_original

# %%
df_steps_original["start_end"].plot.hist(
    bins=24*1000, log=True, figsize=(10, 6), density=False
)
# plt.xlim(0, 1000)

# %%
df_steps_original[mask_under_59].Steps.describe()


# %%
df_steps_original.Steps.describe()


# %%
df_steps_original[mask_under_59].SPM.describe()


# %%
df_steps_original[mask_over_61].SPM.describe()


# %%
df_steps_original[~mask_over_61].SPM.describe()


# %%
df_steps_original["SPM"][mask_under_59].describe(
    [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995]
)

# %%
df_steps_original["SPM"][mask_over_61].describe(
    [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.995]
)

# %%
df_steps_original["startTimestamp_minute"] = df_steps_original[
    "startTimestamp"
].dt.round("min")
# (df_steps["duration"] / 60).round(0).astype
df_steps_original["duration_minutes"] = np.maximum(
    1, (df_steps_original["duration"] / 60).round(0).astype(int)
)
df_steps_original["duration_minutes"].min()

# %%
df_steps_expanded = df_steps_original.loc[
    df_steps_original.index.repeat(df_steps_original["duration_minutes"])
].copy()
df_steps_expanded

# %%
df_steps_original["startTimestamp_minute"].dt.second.describe()

# %%
df_steps_expanded["time_offset"] = df_steps_expanded.groupby(level=0).cumcount()
df_steps_expanded["timestamp"] = df_steps_expanded[
    "startTimestamp_minute"
] + pd.to_timedelta(df_steps_expanded["time_offset"], unit="min")

df_steps_expanded["local_timestamp"] = df_steps_expanded["local_start_time"].dt.round(
    "min"
) + pd.to_timedelta(df_steps_expanded["time_offset"], unit="min")
df_steps_expanded

# %%
df_steps_original.shape

# %%
# Check for overlapping entries in Steps data
df_steps_expanded_groupby = df_steps_expanded.groupby(["customer", "timestamp"])

df_steps_expanded_groupby.size().value_counts().sort_index()

# %%
df_steps_size = df_steps_expanded_groupby.size().reset_index(name="n_repeat")
# drop all the rows where count is 1
df_steps_size = df_steps_size[df_steps_size["n_repeat"] > 1]
df_steps_size.sort_values(by="n_repeat", ascending=False)

# %%
df_steps_size["n_repeat"].describe(
    [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99]
)

# %%
df_steps_size["n_additional"] = df_steps_size["n_repeat"] - 1
df_steps_size.groupby("customer")["n_additional"].sum().sort_values(ascending=False)
customer_steps_additional = (
    df_steps_size.groupby("customer")["n_additional"].sum().sort_values(ascending=False)
)
# Get all customers from df_backup and ensure they're included with 0 if not in customer_steps_additional
all_customers = df_backup["customer"].unique()
customer_steps_additional = customer_steps_additional.reindex(
    all_customers, fill_value=0
).sort_values(ascending=False)

print(customer_steps_additional.describe([]).round(2))
customer_steps_additional.quantile(
    [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
    interpolation="linear",
).round(2)

# %%
plt.figure(figsize=(12, 8))
plt.bar(range(len(customer_steps_additional)), customer_steps_additional.values)

# Create percentage-based x-axis labels (reversed from 100% to 0%)
n_customers = len(customer_steps_additional)
percentage_ticks = np.arange(
    0, n_customers, max(1, n_customers // 20)
)  # Show ~20 ticks
percentage_labels = [f"{100 - (i / n_customers) * 100:.0f}%" for i in percentage_ticks]

plt.xticks(percentage_ticks, percentage_labels)
plt.xlabel("Patients Percentile (100% = highest duplicates, 0% = lowest duplicates)")
plt.ylabel("Number of Additional Duplicate Steps Measurements (seconds)")
plt.title("Additional Duplicate Steps Measurements by Customer")
plt.grid(True, alpha=0.3)

# Add horizontal line at 100,000 additional measurements
# plt.axhline(y=100000, color='red', linestyle='--', linewidth=2, label='100,000 threshold')

# Find the position where customer_steps_additional crosses 100,000
threshold_index = (customer_steps_additional > 100000).sum()
threshold_percentage = 100 - (threshold_index / n_customers) * 100

# Add vertical line at the threshold position
# plt.axvline(x=threshold_index, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Add text annotation
# plt.text(threshold_index + 20, 100000 * 1.5, f'{threshold_percentage:.1f}% of patients\nhave <100k duplicates each',
#  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.legend()
plt.tight_layout()

print(f"Steps Threshold analysis:")
print(f"Number of customers with >100,000 additional duplicates: {threshold_index}")
print(
    f"Percentage of customers with >100,000 additional duplicates: {threshold_percentage:.1f}%"
)
print(
    f"Percentage of customers with ≤100,000 additional duplicates: {100 - threshold_percentage:.1f}%"
)

# Also show top 20 customers with most duplicates
print("\nTop 20 customers with most additional Steps duplicates:")
customer_steps_additional_top20 = customer_steps_additional.head(20)
print(customer_steps_additional_top20)

plt.xlim(
    -1, len(customer_steps_additional) / 2
)  # Set x-axis limits to cover all customers
plt.show()

# Print customer names with >100,000 additional duplicates for easy copying
customers_steps_over_threshold = customer_steps_additional[
    customer_steps_additional > 100000
].index.tolist()
print(
    f"Customers with >100,000 additional Steps duplicates ({len(customers_steps_over_threshold)} total):"
)
print(customers_steps_over_threshold)

# %%
# Remove duplicates by taking the average where there are multiple entries per timestamp


df_steps_avg = df_steps_expanded_groupby.agg(
    {
        "for_id": "first",
        "SPM": "mean",
        "local_timestamp": "first",
        # "timezoneOffset": "first",
        # "startTimestamp": "first",
        # "endTimestamp": "max",
        # "local_endTimestamp": "max",
    }
).reset_index()

df_steps_min = df_steps_expanded_groupby.agg(
    {
        "SPM": "min",
    }
).reset_index()

df_steps_max = df_steps_expanded_groupby.agg(
    {
        "SPM": "max",
    }
).reset_index()

df_steps_avg

# %%
diff_steps = df_steps_max["SPM"] - df_steps_min["SPM"]
print(
    f"Proportion of repeated Steps data that have different values: {(diff_steps > 0).sum() / (len(df_steps_expanded) - len(df_steps_avg)):.2%}"
)
print("Out of this, the following are the statistics of the differences:")
diff_steps[diff_steps > 0].describe([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# %% [markdown]
# ## Steps Aggregations
# at this point we've got nice df_steps_avg, so we finally can perform aggregations
#
# SPM is just cadence in other words
#
# there's a problem distinguishing the zero steps and missing data

# %%
df_steps = df_steps_avg.copy()
df_steps
df_steps["customer"] = df_steps["customer"].astype("category")
df_steps.sort_values(by=["customer", "timestamp"], inplace=True)

# %% [markdown]
# additionally to the below, we can also include the dst change day, to prevent from overcounting repeating hours, and timezone changes, but my guess is that the effect will be _minimal_

# %%
df_steps["utc_day"] = df_steps["timestamp"].dt.floor("d")
df_steps["utc_hour"] = df_steps["timestamp"].dt.floor("h")
df_steps["local_day"] = df_steps["local_timestamp"].dt.floor("d")
df_steps["local_hour"] = df_steps["local_timestamp"].dt.floor("h")

# the above should preferably be from the same source - either local_timestamp or timestamp
# if from local_timestamp, issue might arise due to timezone changes & DST
# if from timestamp splitting over days might be innacurate
# maybe i'll figure out a way with t_day from local_timestamp and t_hour from timestamp

# # ! DST
df_steps["isnighttime"] = df_steps["local_timestamp"].dt.hour < 6  # 00:00 - 05:59
df_steps

# %%
# Create daily steps aggregates
df_steps_daily = (
    # TODO maybe change to local day?
    df_steps.groupby(["customer", "local_day"], observed=True)
    .agg(
        for_id=("for_id", "first"),
        # local_day=("local_day", "first"), # ! doesn't make much sense, one can have two different ones
        StepsInDay=("SPM", "sum"),
        # TODO change it for .apply and make all the quantiles and max computed in one run
        SPM_25pct=("SPM", lambda x: x.quantile(0.25)),
        SPM_50pct=("SPM", lambda x: x.quantile(0.50)),
        SPM_75pct=("SPM", lambda x: x.quantile(0.75)),
        SPM_max=("SPM", "max"),
        SPM_count=(
            "SPM",
            "count",
        ),  # total number of minues for which we've got the data
        SPM_mean=(
            "SPM",
            "mean",
        ),  # it's not the average steps/min in the day, for that we would need to fill data with 0s; use StepsInDay/(24*60) to get that value
        SPM_std=("SPM", "std"),
        SPM_skew=("SPM", "skew"),
        SPM_kurtosis=("SPM", lambda x: x.kurtosis()),
        # SPM_min=('SPM', 'min'), # doesn't make much sense in Steps (0)
        StepsAtNight_sum=(
            "SPM",
            lambda x: x[df_steps.loc[x.index, "isnighttime"]].sum(),
        ),
        StepsAtNight_mean=(
            "SPM",
            lambda x: x[df_steps.loc[x.index, "isnighttime"]].mean(),
        ),
    )
    .reset_index()
)
df_steps_daily["StepsAtNight_mean"] = df_steps_daily["StepsAtNight_mean"].fillna(0)

df_steps_daily

# %%
# Create hourly steps aggregates
df_steps_hourly = (
    df_steps.groupby(["customer", "local_hour"], observed=True)
    .agg(
        local_day=("local_day", "first"),
        for_id=("for_id", "first"),
        Steps_inHour=("SPM", "sum"),
        SPM_max_inHour=("SPM", "max"),
        SPM_count_inHour=("SPM", "count"),
        SPM_mean_inHour=("SPM", "mean"),
        SPM_std_inHour=("SPM", "std"),
        SPM_skew_inHour=("SPM", "skew"),
        SPM_kurtosis_inHour=("SPM", lambda x: x.kurtosis()),
        # SPM_min=('SPM', 'min'),
    )
    .reset_index()
)

assert df_steps_hourly["local_day"].equals(df_steps_hourly["local_hour"].dt.floor("d"))
df_steps_hourly["clock_hour"] = df_steps_hourly["local_hour"].dt.hour

df_steps_hourly

# %%
df_steps_daily = df_steps_daily.merge(
    df_steps_hourly.groupby(["customer", "local_day"], observed=True).agg(
        StepsPerHour=("Steps_inHour", "mean"),
        SPM_max_avgbyHour=("SPM_max_inHour", "mean"),
        SPM_mean_avgbyHour=("SPM_mean_inHour", "mean"),
        SPM_std_avgbyHour=("SPM_std_inHour", "mean"),
        SPM_skew_avgbyHour=("SPM_skew_inHour", "mean"),
        SPM_kurtosis_avgbyHour=("SPM_kurtosis_inHour", "mean"),
    ),
    on=["customer", "local_day"],
)

# %%
# Most Active hour
df_steps_daily.merge(
    # df_steps_daily = df_steps_daily.merge(
    df_steps_hourly.loc[
        df_steps_hourly.groupby(["customer", "local_day"], observed=True)[
            "Steps_inHour"
        ].idxmax()
    ][
        [
            "customer",
            "local_day",
            "Steps_inHour",
            "clock_hour",
            "SPM_max_inHour",
            "SPM_mean_inHour",
        ]
    ].rename(
        columns={
            "Steps_inHour": "Steps_in_most_active_hour",
            "clock_hour": "most_active_hour",  # hour with most steps
            "SPM_max_inHour": "max_spm_in_most_active_hour",
            "SPM_mean_inHour": "avg_spm_in_most_active_hour",
        }
    ),
    on=["customer", "local_day"],
)

# %%
df_steps_hourly["Steps_inHour_cumsum"] = df_steps_hourly.groupby(
    ["customer", "local_day"], observed=True
)["Steps_inHour"].cumsum()


def percentile_hours(group, percentiles=(0.25, 0.5, 0.75)):
    total_steps = group["Steps_inHour_cumsum"].iloc[-1]
    result = {}
    for p in percentiles:
        idx = group["Steps_inHour_cumsum"].searchsorted(p * total_steps)
        result[f"hour_reach_{int(p * 100)}pct_Steps_cumsum"] = group.iloc[idx][
            "local_hour"
        ].hour
    return pd.Series(result)


df_steps_daily = df_steps_daily.merge(
    df_steps_hourly.groupby(["customer", "local_day"], observed=True).apply(
        percentile_hours, include_groups=False
    ),
    on=["customer", "local_day"],
)

# %%
df_steps_daily.columns

# %% [markdown]
# inspired by RADAR MDD
#
# | Column Name | Description |
# |-------------|-------------|
# | customer | Customer/participant identifier |
# | local_day | Day timestamp (floor to day) UTC |
# | for_id | Record identifier |
# | StepsInDay | Total number of walked steps within the day |
# | SPM_25pct | 25th percentile of daily steps per minute distribution |
# | SPM_50pct | 50th percentile of daily steps per minute distribution |
# | SPM_75pct | 75th percentile of daily steps per minute distribution |
# | SPM_max | Maximum steps per minute along all day |
# | SPM_count | Number of minutes with step data available |
# | SPM_mean | Mean steps per minute along all day (among available records) |
# | SPM_std | Standard deviation of steps per minute along all day |
# | SPM_skew | Skewness of steps per minute along all day |
# | SPM_kurtosis | Kurtosis of steps per minute along all day |
# | StepsAtNight_sum | Sum of steps per minute during nighttime (00:00-05:59) |
# | StepsAtNight_mean | Mean steps per minute during nighttime (00:00-05:59) |
# | StepsPerHour | Mean of hourly step sums (sum of steps per minute, averaged by hour) |
# | SPM_max_avgbyHour | Maximum steps per minute, averaged by hour |
# | SPM_mean_avgbyHour | Mean steps per minute, averaged by hour |
# | SPM_std_avgbyHour | Standard deviation of steps per minute, averaged by hour |
# | SPM_skew_avgbyHour | Skewness of steps per minute, averaged by hour |
# | SPM_kurtosis_avgbyHour | Kurtosis of steps per minute, averaged by hour |
# | Steps_in_most_active_hour | Maximum of the hourly sum of steps along all day |
# | most_active_hour | Most active hour (hour with maximum hourly sum of steps) |
# | max_spm_in_most_active_hour | Maximum step cadence during the most active hour |
# | avg_spm_in_most_active_hour | Average step cadence during the most active hour |
# | hour_reach_25pct_Steps_cumsum | Hour at which 25th percentile of daily steps occurred (cumulative) |
# | hour_reach_50pct_Steps_cumsum | Hour at which 50th percentile of daily steps occurred (cumulative) |
# | hour_reach_75pct_Steps_cumsum | Hour at which 75th percentile of daily steps occurred

# %% [markdown]
# | No. | Column Name | Description |
# |-----|-------------|-------------|
# |  1  | `id` | Unique identifier wearable and ema data within subproject 6 (SP6) |
# |  2  |`for_id` | Unique identifier across all PREACT subprojects and redcap |
# |  3  |`date` | Day timestamp (floor to day) UTC |
# |  4  | `n_steps_day` | Total number of walked steps within the day |
# |  5  | `spm_25_steps` | 25th percentile of daily steps per minute distribution |
# |  6  | `spm_50_steps` | 50th percentile of daily steps per minute distribution |
# |  7  | `spm_75_steps` | 75th percentile of daily steps per minute distribution |
# |  8  | `spm_max_steps` | Maximum steps per minute along all day |
# |  9  | `spm_count_steps` | Number of minutes with step data available |
# |  10 | `spm_mean_steps` | Mean steps per minute along all day (among available records) |
# |  11 | `spm_std_steps` | Standard deviation of steps per minute along all day |
# |  12 | `spm_skew_steps` | Skewness of steps per minute along all day |
# |  13 | `spm_kurtosis_steps` | Kurtosis of steps per minute along all day |
# |  14 | `night_sum_steps` | Sum of steps per minute during nighttime (00:00-05:59) |
# |  15 | `night_mean_steps` | Mean steps per minute during nighttime (00:00-05:59) |
# |  16 | `n_hour_steps` | Mean of hourly step sums (sum of steps per minute, averaged by hour) |
# |  17 | `spm_max_avghr_steps` | Maximum steps per minute, averaged by hour |
# |  18 | `spm_mean_avghr_steps` | Mean steps per minute, averaged by hour |
# |  19 | `spm_std_avghr_steps` | Standard deviation of steps per minute, averaged by hour |
# |  20 | `spm_skew_avghr_steps` | Skewness of steps per minute, averaged by hour |
# |  21 | `spm_kurtosis_avghr_steps` | Kurtosis of steps per minute, averaged by hour |
# |  22 | `n_steps_activehr_steps` | Maximum of the hourly sum of steps along all day |
# |  23 | `timestamp_max_activehr_steps` | Most active hour (hour with maximum hourly sum of steps) |
# |  24 | `max_spm_activehr_steps` | Maximum step cadence during the most active hour |
# |  25 | `mean_spm_activehr_steps` | Average step cadence during the most active hour |
# |  26 | `dailysteps_25perc_steps` | Hour at which 25th percentile of daily steps occurred (cumulative) |
# |  27 | `dailysteps_50perc_steps` | Hour at which 50th percentile of daily steps occurred (cumulative) |
# |  28 | `dailysteps_75perc_steps` | Hour at which 75th percentile of daily steps occurred

# %% [markdown]
# #### Example plot

# %%
# example plot
df_steps_plot = df_steps_avg[
    (df_steps_avg["customer"] == "05kz")
    & (df_steps_avg["timestamp"] >= pd.to_datetime("2023-10-19", utc=True))
    & (df_steps_avg["timestamp"] < pd.to_datetime("2023-10-22", utc=True))
]
df_steps_plot

plt.figure(figsize=(12, 6))
plt.scatter(
    df_steps_plot["timestamp"], df_steps_plot["SPM"], linewidth=0.5
)  # scatter vs line plot
plt.title(
    f"Steps Data for Customer {df_steps_plot['customer'].iloc[0]} on {df_steps_plot['timestamp'].dt.date.iloc[0]} onwards"
)
plt.xlabel("Time")
plt.ylabel("Steps per Second")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# %%
