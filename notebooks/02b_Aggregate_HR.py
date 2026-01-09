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
# # Heart Rate (HR) Aggregate

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

# %% [markdown]
# ## Heart Rate

# %% [markdown]
# ### Make every record last one second (expand the dataframe)

# %% [markdown]
# to get rid of overlapping events, we need to "expand" the dataset, so that one row last only one second (the smalles time duration)

# %%
# df = df_backup[(df_backup["type"] == "HeartRate") & (df_backup["customer"].isin(["4MLe","kVhY"]))]
df = df_backup[(df_backup["type"] == "HeartRate")]
df = df[["customer", "startTimestamp", "endTimestamp", "longValue"]]
df = df.rename(columns={"longValue": "HeartRate"})


# %%
df["duration"] = (
    (df["endTimestamp"] - df["startTimestamp"]).dt.total_seconds().fillna(0).astype(int)
)
df["duration"].describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# %%
df_expanded = df.loc[df.index.repeat(df["duration"])].copy()
df_expanded

# %%
df_expanded["time_offset"] = df_expanded.groupby(level=0).cumcount()
df_expanded["timestamp"] = df_expanded["startTimestamp"] + pd.to_timedelta(
    df_expanded["time_offset"], unit="s"
)

# %% [markdown]
# ### Analysis 

# %% [markdown]
# as we can see, there are over 1M overlapping entries

# %%
# df_expanded_groupby = df_expanded.groupby(['customer', 'timestamp'])
df_expanded_groupby = df_expanded.groupby(["timestamp", "customer"])

df_expanded_groupby.size().value_counts().sort_index()

# %%
df_size = df_expanded_groupby.size().reset_index(name="n_repeat")
# drop all the rows where count is 1
df_size = df_size[df_size["n_repeat"] > 1]
df_size.sort_values(by="n_repeat", ascending=False)


# %%
df_size["n_repeat"].describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99])

# %%
df_size["n_additional"] = df_size["n_repeat"] - 1
df_size.groupby("customer")["n_additional"].sum().sort_values(ascending=False)
customer_additional = (
    df_size.groupby("customer")["n_additional"].sum().sort_values(ascending=False)
)
# Get all customers from df_backup and ensure they're included with 0 if not in customer_additional
all_customers = df_backup["customer"].unique()
customer_additional = customer_additional.reindex(
    all_customers, fill_value=0
).sort_values(ascending=False)

print(customer_additional.describe([]).round(2))
customer_additional.quantile(
    [0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0],
    interpolation="linear",
).round(2)

# %% [markdown]
# by setting threshold of n_additional = 100_000 sec, we keep 97% of patients 

# %%
plt.figure(figsize=(12, 8))
plt.bar(range(len(customer_additional)), customer_additional.values)

# Create percentage-based x-axis labels (reversed from 100% to 0%)
n_customers = len(customer_additional)
percentage_ticks = np.arange(
    0, n_customers, max(1, n_customers // 20)
)  # Show ~20 ticks
percentage_labels = [f"{100 - (i / n_customers) * 100:.0f}%" for i in percentage_ticks]

plt.xticks(percentage_ticks, percentage_labels)
plt.xlabel("Patients Percentile (100% = highest duplicates, 0% = lowest duplicates)")
plt.ylabel("Number of Additional Duplicate Heart Rate Measurements (seconds)")
plt.title("Additional Duplicate Heart Rate Measurements by Customer")
plt.grid(True, alpha=0.3)
# plt.yscale('log')  # Use logarithmic scale for better visibility

# Add horizontal line at 100,000 additional measurements
plt.axhline(
    y=100000, color="red", linestyle="--", linewidth=2, label="100,000 threshold"
)

# Find the position where customer_additional crosses 100,000
threshold_index = (customer_additional > 100000).sum()
threshold_percentage = 100 - (threshold_index / n_customers) * 100

# Add vertical line at the threshold position
plt.axvline(x=threshold_index, color="red", linestyle="--", linewidth=2, alpha=0.7)

# Add text annotation
plt.text(
    threshold_index + 20,
    100000 * 1.5,
    f"{threshold_percentage:.1f}% of patients\nhave <100k duplicates each",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
)

plt.legend()
plt.tight_layout()

print(f"Threshold analysis:")
print(f"Number of customers with >100,000 additional duplicates: {threshold_index}")
print(
    f"Percentage of customers with >100,000 additional duplicates: {threshold_percentage:.1f}%"
)
print(
    f"Percentage of customers with ≤100,000 additional duplicates: {100 - threshold_percentage:.1f}%"
)

# Also show top 20 customers with most duplicates
print("\nTop 20 customers with most additional duplicates:")
customer_additional_top20 = customer_additional.head(20)
print(customer_additional_top20)

plt.xlim(-1, len(customer_additional) / 2)  # Set x-axis limits to cover all customers
plt.show()


# Print customer names with >100,000 additional duplicates for easy copying
customers_over_threshold = customer_additional[
    customer_additional > 100000
].index.tolist()
print(
    f"Customers with >100,000 additional duplicates ({len(customers_over_threshold)} total):"
)
print(customers_over_threshold)

# %%
# Remove duplicates by taking the average where there are multiple entries per timestamp

df_avg = df_expanded_groupby.agg(
    {
        "HeartRate": "mean",
    }
).reset_index()

df_min = df_expanded_groupby.agg(
    {
        "HeartRate": "min",
    }
).reset_index()

df_max = df_expanded_groupby.agg(
    {
        "HeartRate": "max",
    }
).reset_index()

df_avg

# %%
diff = df_max["HeartRate"] - df_min["HeartRate"]
print(
    f"proportion of repeated data that have different values: {(diff > 0).sum() / (len(df_expanded) - len(df_avg)):.2%}"
)
print("out of this, the following are the statistics of the differences:")
diff[diff > 0].describe([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
# TODO resolve what to do with this next, about ~80 of repeated

# %% [markdown]
# ### Continue with HeartRate aggregation

# %%
# df = df_avg[df_avg["customer"].isin(["1BAf", "lAHE", "4MLe", "kVhY"])]
df_hr = df_avg

# %%
df_hr["HR_zone_resting"] = df_hr["HeartRate"] < 60
df_hr["HR_zone_moderate"] = (60 <= df_hr["HeartRate"]) & (df_hr["HeartRate"] < 100)
df_hr["HR_zone_vigorous"] = 100 <= df_hr["HeartRate"]

# %%
# Create hourly heart rate averages
df_hr["t_hour"] = df_hr["timestamp"].dt.floor("h")
df_hr_hourly = (
    df_hr.groupby(["customer", "t_hour"])
    .agg(
        HR_count=("HeartRate", "count"),
        HR_mean=("HeartRate", "mean"),
        HR_std=("HeartRate", "std"),
        HR_min=("HeartRate", "min"),
        HR_max=("HeartRate", "max"),
        HR_skew=("HeartRate", "skew"),
        HR_kurtosis=("HeartRate", lambda x: x.kurtosis()),
        HR_zone_resting=("HR_zone_resting", "sum"),
        HR_zone_moderate=("HR_zone_moderate", "sum"),
        HR_zone_vigorous=("HR_zone_vigorous", "sum"),
    )
    .reset_index()
)

df_hr_hourly

# %%
# Create daily heart rate averages
df_hr["t_day"] = df_hr["timestamp"].dt.floor("d")
df_hr_daily = (
    df_hr.groupby(["customer", "t_day"])
    .agg(
        HR_count=("HeartRate", "count"),
        HR_mean=("HeartRate", "mean"),
        HR_std=("HeartRate", "std"),
        HR_min=("HeartRate", "min"),
        HR_max=("HeartRate", "max"),
        HR_skew=("HeartRate", "skew"),
        HR_kurtosis=("HeartRate", lambda x: x.kurtosis()),
        HR_zone_resting=("HR_zone_resting", "sum"),
        HR_zone_moderate=("HR_zone_moderate", "sum"),
        HR_zone_vigorous=("HR_zone_vigorous", "sum"),
    )
    .reset_index()
)

df_hr_daily

# %%
# this stopped working TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and Timestamp
# df_plot = df_avg[(df_avg["customer"] == "05kz") & (df_avg["timestamp"] >= pd.to_datetime("2023-10-19"))
#                  & (df_avg["timestamp"] < pd.to_datetime("2023-10-22"))]
# df_plot

# plt.figure(figsize=(12, 6))
# plt.scatter(df_plot['timestamp'], df_plot['HeartRate'], linewidth=0.5) # scatter vs line plot
# plt.title(f'Heart Rate Data for Customer {df_plot["customer"].iloc[0]} on {df_plot["timestamp"].dt.date.iloc[0]} onwards')
# plt.xlabel('Time')
# plt.ylabel('Heart Rate (BPM)')
# plt.xticks(rotation=45)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# %%
