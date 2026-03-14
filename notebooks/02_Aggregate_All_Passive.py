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
# # Aggregate all passive data

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
from pathlib import Path




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
    proj_sheet,
    preprocessed_path,
)
from functions.preprocessing.ema_mappings import clean_heart_rate_data
from functions.preprocessing.aggregation import compute_sleep_sessions

from functions.preprocessing import gps_features
from functions.preprocessing.ema_mappings import run_ema_mappings
from functions.preprocessing.missing_data import summarize_missing_data

# from functions.preprocessing.aggregation import aggregate_all_passive
from functions.preprocessing.aggregation import (
    aggregate_sleep_daily,
    aggregate_hr_daily,
    aggregate_steps_daily,
    aggregate_activity_daily,
    aggregate_elevation_daily,
    aggregate_floors_daily,
)


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
debug = True
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON!!!!!!!!!!!!!")
    ids_to_keep = df_backup["id"].unique()[100:110]  # Keep only the first 10 unique IDs for debugging
    df_backup = df_backup[df_backup["id"].isin(ids_to_keep)].copy()
    print(f"Unique IDs kept for debugging: {ids_to_keep}")
    print(f"Data shape after filtering for debug: {df_backup.shape}")

# %%
df_backup.head()

# %% [markdown]
# ### Filter passive records by EMA post end (+14 days)

# %%
# check how many unique days of data we have per ids
df_backup.groupby("id")["start_date"].nunique().sort_values(ascending=False)


# %% [markdown]
# #### Filter by redcap

# %%
def get_last_avail_redcap_date_per_participant(
    for_df,
    zert_df,
    participant_col="for_id",
    date_cols=None,
    col_priority=None,
):
    if date_cols is None:
        date_cols = [
            "ema_begin_baseline",
            "ema_begin_t20",
            "ema_begin_post",
            "ema_begin_passive",
            "ema_completed_date",
            "ema_end_baseline",
            "ema_end_t20",
            "ema_end_post",
            "ema_end_passive",
        ]

    if col_priority is None:
        col_priority = [
            "ema_completed_date",
            "ema_end_passive",
            "ema_end_post",
            "ema_end_t20",
            "ema_end_baseline",
        ]

    onboarding = pd.concat([for_df, zert_df], axis=0, ignore_index=True)

    available_date_cols = [c for c in date_cols if c in onboarding.columns]
    if not available_date_cols:
        return pd.DataFrame(
            columns=[participant_col, "last_avail_date_redcap", "last_avail_colname"]
        )

    onboarding = onboarding[[participant_col, *available_date_cols]].copy()
    onboarding = onboarding.groupby(participant_col, as_index=False).agg(
        lambda s: s.dropna().iloc[0] if not s.dropna().empty else pd.NA
    )

    ordered_date_cols = [c for c in col_priority if c in available_date_cols] + [
        c for c in available_date_cols if c not in col_priority
    ]
    priority_rank = {c: i for i, c in enumerate(ordered_date_cols)}

    tmp = onboarding[[participant_col, *ordered_date_cols]].copy()
    tmp[ordered_date_cols] = tmp[ordered_date_cols].apply(
        pd.to_datetime, errors="coerce"
    )

    tmp["last_avail_date_redcap"] = tmp[ordered_date_cols].max(axis=1)
    is_last = tmp[ordered_date_cols].eq(tmp["last_avail_date_redcap"], axis=0)
    tmp["last_avail_colname"] = is_last.idxmax(axis=1).where(is_last.any(axis=1), pd.NA)
    tmp["last_avail_col_priority"] = (
        tmp["last_avail_colname"].map(priority_rank).fillna(len(priority_rank))
    )

    last_avail_date_per_participant = (
        tmp.sort_values(
            [participant_col, "last_avail_date_redcap", "last_avail_col_priority"],
            ascending=[True, False, True],
            na_position="last",
            kind="mergesort",
        )
        .groupby(participant_col, as_index=False)
        .head(1)[[participant_col, "last_avail_date_redcap", "last_avail_colname"]]
        .sort_values(participant_col)
        .reset_index(drop=True)
    )

    return last_avail_date_per_participant


# %%
redcap_root = Path("/home/milu10/src/tiki_code/tmp/SP6/redcap")
for_ema_onboarding_13012026 = pd.read_csv(
    redcap_root / "ema_onboarding/FOR_ema_onboarding_13012026.csv"
)
zert_for_ema_onboarding_13012026 = pd.read_csv(
    redcap_root / "ema_onboarding/ZERT_FOR_ema_onboarding_13012026.csv"
)

# %%
last_avail_redcap = get_last_avail_redcap_date_per_participant(
    for_ema_onboarding_13012026, zert_for_ema_onboarding_13012026
)
last_avail_redcap["last_avail_colname"].value_counts(dropna=False)

# %%
if "for_id" not in df_backup.columns:
    raise KeyError("df_backup is missing required column: for_id")

if "startTimestamp" in df_backup.columns:
    ts_col = "startTimestamp"
# elif "local_timestamp_start" in df_backup.columns:
#     ts_col = "local_timestamp_start"
else:
    raise KeyError("df_backup is missing both startTimestamp and local_timestamp_start")

last_avail_redcap = last_avail_redcap.copy()
last_avail_redcap["last_avail_date_redcap"] = pd.to_datetime(
    last_avail_redcap["last_avail_date_redcap"], errors="coerce", utc=True
)
last_avail_redcap["redcap_cutoff_plus21"] = (
    last_avail_redcap["last_avail_date_redcap"] + pd.Timedelta(days=21)
)

# df_backup["for_id"] = df_backup["for_id"].astype(str).str.strip()
# df_backup[ts_col] = pd.to_datetime(df_backup[ts_col], errors="coerce", utc=True)

if "redcap_cutoff_plus21" in df_backup.columns:
    df_backup = df_backup.drop(columns=["redcap_cutoff_plus21"])

n_records_before = len(df_backup)
df_backup = df_backup.merge(
    last_avail_redcap[["for_id", "redcap_cutoff_plus21"]],
    on="for_id",
    how="left",
)

df_backup = df_backup[
    df_backup["redcap_cutoff_plus21"].isna()
    | (df_backup[ts_col] <= df_backup["redcap_cutoff_plus21"])
].copy()

print(f"Records before REDCap +21d filtering: {n_records_before:_d}")
print(f"Records after REDCap +21d filtering: {len(df_backup):_d}")
print(f"Records removed: {n_records_before - len(df_backup):_d}")

# %%
df_backup.groupby("id")["start_date"].nunique().sort_values(ascending=False)

# %% [markdown]
# #### Filter by monitoring spreadsheet

# %%
monitoring_url = f"https://docs.google.com/spreadsheets/d/{proj_sheet}/export?format=csv"
# monitoring_csv_fallback = os.path.join(preprocessed_path, "monitoring_data.csv")

monitoring_source = monitoring_url
df_monitoring = pd.read_csv(monitoring_url, low_memory=False)

if "ema_post_end" not in df_monitoring.columns and "Ende EMA Post" in df_monitoring.columns:
    df_monitoring = df_monitoring.rename(columns={"Ende EMA Post": "ema_post_end"})

if "for_id" not in df_monitoring.columns:
    candidate_for_id_cols = ["FOR_ID", "forid", "record_id", "Record ID"]
    found_for_id_cols = [col for col in candidate_for_id_cols if col in df_monitoring.columns]
    if found_for_id_cols:
        df_monitoring = df_monitoring.rename(columns={found_for_id_cols[0]: "for_id"})
    else:
        raise KeyError("Could not find a for_id column in monitoring sheet.")

if "ema_post_end" not in df_monitoring.columns:
    raise KeyError("Could not find ema_post_end (or Ende EMA Post) in monitoring sheet.")

df_monitoring["for_id"] = df_monitoring["for_id"].astype(str).str.strip()
df_monitoring["ema_post_end"] = pd.to_datetime(
    df_monitoring["ema_post_end"], dayfirst=True, errors="coerce"
).dt.date

df_post_end = (
    df_monitoring.dropna(subset=["for_id", "ema_post_end"])
    .groupby("for_id", as_index=False)["ema_post_end"]
    .max()
)
df_post_end["post_cutoff_day"] = (
    pd.to_datetime(df_post_end["ema_post_end"], errors="coerce") + pd.Timedelta(days=14)
).dt.date

if "for_id" not in df_backup.columns:
    raise KeyError("df_backup is missing required column: for_id")

df_backup["for_id"] = df_backup["for_id"].astype(str).str.strip()
df_backup["local_timestamp_start"] = pd.to_datetime(df_backup["local_timestamp_start"], errors="coerce")
df_backup["local_day_filter"] = df_backup["local_timestamp_start"].dt.date

n_records_before = len(df_backup)
df_backup = df_backup.merge(df_post_end[["for_id", "post_cutoff_day"]], on="for_id", how="left")
df_backup = df_backup[
    df_backup["post_cutoff_day"].isna()
    | (df_backup["local_day_filter"] <= df_backup["post_cutoff_day"])
].copy()
# df_backup = df_backup.drop(columns=["post_cutoff_day", "local_day_filter"])

print(f"Loaded monitoring data from: {monitoring_source}")
print(f"Records before post-end filtering: {n_records_before}")
print(f"Records after post-end filtering: {len(df_backup)}")
print(f"Records removed: {n_records_before - len(df_backup)}")

# %%
df_backup.head()

# %%
# df_backup.groupby("id")["start_date"].nunique().sort_values(ascending=False)
tmp_for_id = df_backup.groupby("for_id")["start_date"].nunique().sort_values(ascending=False)

# %%
for forid, count in tmp_for_id.items():
    print(f"{forid}: {count} unique days")

# %% [markdown]
# ### check the other types

# %%
type_coverage = (
    df_backup.groupby("modality")["id"].nunique().reset_index(name="unique_ids")
)
type_coverage["percentage_ids"] = (
    type_coverage["unique_ids"] / df_backup["id"].nunique()
) * 100
type_coverage = type_coverage.merge(
    df_backup["modality"].value_counts().reset_index(name="total_entries"),
    on="modality",
    how="left",
)
type_coverage.sort_values(by="total_entries", ascending=False, inplace=True)
type_coverage.reset_index(drop=True, inplace=True)
type_coverage


# %%

# %%
df_backup.groupby("id")["modality"].value_counts()

print(f"Total id-type combinations: {len(df_backup.groupby('id')["modality"].value_counts())}")
print(f"Unique ids: {df_backup['id'].nunique()}")
print(f"Unique types: {df_backup["modality"].nunique()}")

# %% [markdown]
# ## Single modality aggregations

# %% [markdown]
# ### Sleep

# %%
# cProfile.run('aggregate_sleep_daily(df_backup)')

# %%
df_sleep_daily = aggregate_sleep_daily(df_backup)
df_sleep_daily.head()

# %% [markdown]
# ### Heart Rate

# %%
df_hr_daily = aggregate_hr_daily(df_backup, include_zero_hours=False)


# %%
pd.set_option('display.max_columns', None)
print(df_hr_daily.head())
# df_backup.head()

# %%
print(df_hr_daily.shape)
df_hr_daily.head()

# %%

# %% [markdown]
# ### Steps

# %%
# cProfile.run('aggregate_steps_daily(df_backup)')

# %%
df_steps_daily = aggregate_steps_daily(df_backup)
# TODO add the number of hours with zero steps per day as a feature / number of hours with steps > 0 per day
df_steps_daily.head()

# %%
# cProfile.run('aggregate_activity_daily(df_backup)', 'agg_activity.prof')

# %% [markdown]
# ### Activity

# %%
df_activity_daily = aggregate_activity_daily(df_backup)
df_activity_daily.head()
# %% [markdown]
# ### Elevation

# %%
df_elevation_daily = aggregate_elevation_daily(df_backup)
df_elevation_daily.head()
# %% [markdown]
# ### Floors
# %%
df_floors_daily = aggregate_floors_daily(df_backup)
df_floors_daily.head()
# %% [markdown]
# ## Aggregate all
# Inside, it runs the functions from above and merge all the dataframes.

# %%
def merge_dataframes(
    dataframes: list[pd.DataFrame], on=("id", "local_day")
) -> pd.DataFrame:
    """Merge a list of dataframes on 'id' and 'local_day' using outer join."""

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=list(on), how="outer"),
        dataframes,
    )
    return merged_df


# %%
# # Aggregate all passive data

# df_daily = df_sleep_daily.merge(df_hr_daily, on=["id", "local_day"], how="outer")
# df_daily = df_daily.merge(df_steps_daily, on=["id", "local_day"], how="outer")
# df_daily = df_daily.merge(df_activity_daily, on=["id", "local_day"], how="outer")
# df_daily = df_daily.merge(df_elevation_daily, on=["id", "local_day"], how="outer")
# df_daily = df_daily.merge(df_floors_daily, on=["id", "local_day"], how="outer")

# %%
# or just call aggregate_all_passive
# df_daily = aggregate_all_passive(df_backup)
df_daily = merge_dataframes(
    [
        df_sleep_daily,
        df_hr_daily,
        df_steps_daily,
        df_activity_daily,
        df_elevation_daily,
        df_floors_daily,
    ]
)
df_daily

# %%
print(f"Total rows: {len(df_daily)}")
print(f"Total columns: {len(df_daily.columns)}")
print(f"Unique ids: {df_daily['id'].nunique()}")
print(f"Date range: {df_daily['local_day'].min()} to {df_daily['local_day'].max()}")

# %%
df_daily

# %%
# save the aggregated daily data
passive_daily_dir = Path("/sc-projects/sc-proj-cc15-preact/SP6/preprocessed/passive/daily/")
output_path = passive_daily_dir / "daily_aggregated_all_passive.feather"


# %%

df_daily.to_feather(output_path)
print(f"Aggregated daily data saved to: {output_path}")

# %%
for key, df in {
    "sleep": df_sleep_daily,
    "hr": df_hr_daily,
    "steps": df_steps_daily,
    "activity": df_activity_daily,
    "elevation": df_elevation_daily,
    "floors": df_floors_daily,
}.items():
    print(f"{key}: {len(df)} rows, columns: {list(df.columns)}")

    single_save_path = passive_daily_dir / f"daily_agg_{key}.parquet"
    df.to_parquet(single_save_path)
    print(f"{key} daily data saved to: {single_save_path}")

# %%
df_daily = pd.read_feather(output_path)

# %% [markdown]
# ## HR Coverage/Missingness Analysis

# %%
entries_per_day = df_daily.groupby("local_day").size().reset_index(name="entries_per_day").sort_values(by="local_day")

plt.figure(figsize=(16, 6))
plt.plot(entries_per_day['local_day'], entries_per_day['entries_per_day'], 
         color='#3498db', linewidth=1.5, alpha=0.8)
plt.fill_between(entries_per_day['local_day'], entries_per_day['entries_per_day'], 
                 alpha=0.3, color='#3498db')
plt.title('Number of id with any records per Day', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Number of Entries', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)
plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()

# %%
df_daily.groupby("id").agg({"local_day": ["min", "max", "count"]})

# %%
# Check days without HR data (HR_raw_records is NA)
df_no_hr = df_daily[df_daily['HR_raw_records'].isna()].copy()

print("=" * 60)
print("DAYS WITHOUT HR DATA (NA) ANALYSIS")
print("=" * 60)
print(f"Total days without HR data: {len(df_no_hr):,}")
print(f"Unique ids affected: {df_no_hr['id'].nunique()}")
print(f"Date range: {df_no_hr['local_day'].min()} to {df_no_hr['local_day'].max()}")
print("=" * 60)

# Per-id breakdown
no_hr_per_id = df_no_hr.groupby('id').size().reset_index(name='days_without_hr')
print(f"\nDays without HR per id:")
print(f"  Mean: {no_hr_per_id['days_without_hr'].mean():.1f}")
print(f"  Median: {no_hr_per_id['days_without_hr'].median():.1f}")
print(f"  Min: {no_hr_per_id['days_without_hr'].min()}")
print(f"  Max: {no_hr_per_id['days_without_hr'].max()}")
print("=" * 60)

# Check what other data exists on days without HR
other_data_cols = ['SPM_count', 'longest_total_sleep_time', 'ACTIVE_n_sessions', 
                   'total_elevation_gain', 'total_floors_climbed']
available_cols = [col for col in other_data_cols if col in df_no_hr.columns]

print(f"\nOther data available on days WITHOUT HR:")
for col in available_cols:
    n_available = df_no_hr[col].notna().sum()
    pct = (n_available / len(df_no_hr)) * 100
    print(f"  {col}: {n_available:,} days ({pct:.1f}%)")
print("=" * 60)

# %%
# Visualize days without HR over time
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Days Without HR Data Analysis', fontsize=14, fontweight='bold')

# Top: Daily count of ids without HR
hr_na_per_day = df_no_hr.groupby('local_day').size().reset_index(name='n_ids_no_hr')
axes[0].bar(hr_na_per_day['local_day'], hr_na_per_day['n_ids_no_hr'],
            color='#e74c3c', alpha=0.7, edgecolor='black', width=0.8)
axes[0].set_ylabel('Number of ids Without HR', fontsize=11)
axes[0].set_title('Daily Count of ids Without HR Data', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Bottom: Per-id days without HR (sorted)
no_hr_sorted = no_hr_per_id.sort_values('days_without_hr', ascending=True)
axes[1].barh(range(len(no_hr_sorted)), no_hr_sorted['days_without_hr'],
             color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(no_hr_sorted)))
axes[1].set_yticklabels(no_hr_sorted['id'], fontsize=8)
axes[1].set_xlabel('Days Without HR Data', fontsize=11)
axes[1].set_ylabel('id ID', fontsize=11)
axes[1].set_title('Days Without HR Data per id', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# %%
# Sample some days without HR to inspect what data IS available
print("Sample of days WITHOUT HR data (first 20 rows):")
sample_cols = ['id', 'local_day', 'HR_raw_records', 'SPM_count', 
               'longest_total_sleep_time', 'ACTIVE_n_sessions']
available_sample_cols = [col for col in sample_cols if col in df_no_hr.columns]
df_no_hr[available_sample_cols].head(20)

# %%
# Per-id: percentage of available days with NO HR data
id_hr_na_pct = df_daily.groupby('id').agg(
    total_days=('local_day', 'count'),
    days_with_na=('HR_raw_records', lambda x: x.isna().sum())
).reset_index()
id_hr_na_pct['na_percentage'] = (id_hr_na_pct['days_with_na'] / id_hr_na_pct['total_days']) * 100

# Histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(id_hr_na_pct['na_percentage'], bins=100, color='#e74c3c', edgecolor='black', alpha=0.7)
ax.axvline(id_hr_na_pct['na_percentage'].mean(), color='darkred', linestyle='--', linewidth=2, 
           label=f"Mean: {id_hr_na_pct['na_percentage'].mean():.1f}%")
ax.axvline(id_hr_na_pct['na_percentage'].median(), color='darkgreen', linestyle='--', linewidth=2, 
           label=f"Median: {id_hr_na_pct['na_percentage'].median():.1f}%")
ax.set_xlabel('Percentage of Days Without HR Data (%)', fontsize=11)
ax.set_ylabel('Number of ids', fontsize=11)
ax.set_title('Distribution of HR Data Missingness per id', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nSummary Statistics:")
print(f"ids with 0% NA (complete HR coverage): {(id_hr_na_pct['na_percentage'] == 0).sum()}")
print(f"ids with 100% NA (no HR data): {(id_hr_na_pct['na_percentage'] == 100).sum()}")
print(f"ids with >50% NA: {(id_hr_na_pct['na_percentage'] > 50).sum()}")

# %%
# Number of days with HR data per id
# #! not counting ids with zero days of HR data 
days_per_id = df_daily[df_daily['HR_raw_records'].notna()].groupby('id').size().reset_index(name='n_days')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(days_per_id['n_days'], bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
ax.axvline(days_per_id['n_days'].mean(), color='darkred', linestyle='--', linewidth=2, 
           label=f"Mean: {days_per_id['n_days'].mean():.1f} days")
ax.axvline(days_per_id['n_days'].median(), color='darkgreen', linestyle='--', linewidth=2, 
           label=f"Median: {days_per_id['n_days'].median():.1f} days")
ax.set_xlabel('Number of Days with HR Data', fontsize=11)
ax.set_ylabel('Number of ids', fontsize=11)
ax.set_title('Distribution of Recording Days per id', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# 1. Population-level distributions of coverage metrics
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('HR Coverage Metrics - Population Distributions', fontsize=16, fontweight='bold')

coverage_cols = [
    ('HR_raw_records', 'Raw Records per Day', '#e74c3c'),
    ('HR_raw_hours_with_records', 'Hours with Records (0-24)', '#c0392b'),
    ('HR_seconds_hours_with_data', 'Hours with Data (0-24)', '#e67e22'),
    ('HR_seconds_per_hour_mean', 'Mean Seconds per Hour', '#d35400'),
    ('HR_seconds_per_hour_median', 'Median Seconds per Hour', '#f39c12'),
    ('HR_raw_records_per_hour_mean', 'Mean Records per Hour', '#e74c3c'),
]

for idx, (col, title, color) in enumerate(coverage_cols):
    ax = axes[idx // 3, idx % 3]
    data = df_daily[col].dropna()
    if len(data) > 0:
        data.hist(bins=50, ax=ax, color=color, edgecolor='black', alpha=0.7)
        median_val = data.median()
        ax.axvline(median_val, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.1f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 2. Per-id coverage heatmap (temporal view)
fig, ax = plt.subplots(figsize=(20, 10))

# Prepare data - pivot to id x day matrix
ids_to_plot = sorted(df_daily['id'].unique())[:20]  # First 20 ids
df_hr_subset = df_daily[df_daily['id'].isin(ids_to_plot)].copy()

# Pivot: rows = ids, columns = days
heatmap_data = df_hr_subset.pivot_table(
    index='id',
    columns='local_day',
    values='HR_seconds_hours_with_data',
    aggfunc='first'
)

# Plot heatmap
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.0f', 
            cbar_kws={'label': 'Hours with HR Data (0-24)'},
            linewidths=0.5, ax=ax, vmin=0, vmax=24)
ax.set_title('HR Data Coverage Heatmap (First 20 ids)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('id ID')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# 3. Per-id summary boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('HR Coverage Distribution Across ids', fontsize=14, fontweight='bold')

# Compute per-id medians
id_summary = df_daily.groupby('id').agg(
    median_raw_records=('HR_raw_records', 'median'),
    median_hours_with_records=('HR_raw_hours_with_records', 'median'),
    median_seconds_per_hour=('HR_seconds_per_hour_mean', 'median')
).reset_index()

box_configs = [
    ('median_raw_records', 'Median Raw Records/Day', '#e74c3c', axes[0]),
    ('median_hours_with_records', 'Median Hours with Records/Day', '#c0392b', axes[1]),
    ('median_seconds_per_hour', 'Median Seconds/Hour', '#e67e22', axes[2]),
]

for col, title, color, ax in box_configs:
    data = id_summary[col].dropna()
    if len(data) > 0:
        bp = ax.boxplot([data], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xticklabels(['All ids'])
        ax.grid(axis='y', alpha=0.3)
        
        # Add summary stats as text
        mean_val = data.mean()
        median_val = data.median()
        ax.text(0.98, 0.98, f'Mean: {mean_val:.1f}\nMedian: {median_val:.1f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# %%
# 4. Time-series coverage plots (sample ids)
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('HR Coverage Over Time (Sample ids)', fontsize=14, fontweight='bold')

sample_ids = df_daily['id'].unique()[:5]  # First 5 ids
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for i, id in enumerate(sample_ids):
    id_data = df_daily[df_daily['id'] == id].sort_values('local_day')
    
    # Plot 1: Hours with records
    axes[0].plot(id_data['local_day'], id_data['HR_raw_hours_with_records'],
                 marker='o', markersize=4, alpha=0.7, label=f'id {id}',
                 color=colors[i])
    
    # Plot 2: Mean seconds per hour
    axes[1].plot(id_data['local_day'], id_data['HR_seconds_per_hour_mean'],
                 marker='o', markersize=4, alpha=0.7, label=f'id {id}',
                 color=colors[i])

axes[0].set_title('Hours with HR Records per Day', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Hours (0-24)')
axes[0].set_ylim(0, 24)
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
axes[0].grid(alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

axes[1].set_title('Mean Seconds of HR Data per Hour', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Seconds per Hour')
axes[1].set_ylim(0, 3600)
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].grid(alpha=0.3)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# %%
# 5. Coverage vs HR quality scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('HR Coverage vs. HR Quality Metrics', fontsize=14, fontweight='bold')

# Filter out NaN values
df_scatter = df_daily.dropna(subset=['HR_raw_hours_with_records', 'HR_std', 
                                         'HR_seconds_per_hour_mean', 'HR_mean'])

# Plot 1: Hours with records vs HR std
axes[0].scatter(df_scatter['HR_raw_hours_with_records'], df_scatter['HR_std'],
                alpha=0.5, c='#e74c3c', s=20, edgecolors='black', linewidths=0.5)
axes[0].set_xlabel('Hours with HR Records', fontsize=11)
axes[0].set_ylabel('HR Std Dev (BPM)', fontsize=11)
axes[0].set_title('Coverage vs HR Variability', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Add correlation coefficient
if len(df_scatter) > 0:
    corr = df_scatter[['HR_raw_hours_with_records', 'HR_std']].corr().iloc[0, 1]
    axes[0].text(0.05, 0.95, f'Corr: {corr:.3f}',
                 transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Mean seconds per hour vs HR mean
axes[1].scatter(df_scatter['HR_seconds_per_hour_mean'], df_scatter['HR_mean'],
                alpha=0.5, c='#3498db', s=20, edgecolors='black', linewidths=0.5)
axes[1].set_xlabel('Mean Seconds per Hour', fontsize=11)
axes[1].set_ylabel('Mean HR (BPM)', fontsize=11)
axes[1].set_title('Data Density vs Mean HR', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# Add correlation coefficient
if len(df_scatter) > 0:
    corr = df_scatter[['HR_seconds_per_hour_mean', 'HR_mean']].corr().iloc[0, 1]
    axes[1].text(0.05, 0.95, f'Corr: {corr:.3f}',
                 transform=axes[1].transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

# %%
# 6. Population-level daily coverage over time
fig, ax = plt.subplots(figsize=(16, 6))

# Aggregate across all ids per day
daily_population = df_daily.groupby('local_day').agg(
    median_hours=('HR_seconds_hours_with_data', 'median'),
    q25_hours=('HR_seconds_hours_with_data', lambda x: x.quantile(0.25)),
    q75_hours=('HR_seconds_hours_with_data', lambda x: x.quantile(0.75)),
    n_ids=('id', 'nunique')
).reset_index()

# Plot median with IQR shading
ax.plot(daily_population['local_day'], daily_population['median_hours'],
        color='#e74c3c', linewidth=2, label='Median')
ax.fill_between(daily_population['local_day'],
                daily_population['q25_hours'],
                daily_population['q75_hours'],
                alpha=0.3, color='#e74c3c', label='IQR (25th-75th percentile)')

ax.set_title('Population-Level HR Coverage Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Hours with HR Data (0-24)', fontsize=11)
ax.set_ylim(0, 24)
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Add secondary y-axis for number of ids
ax2 = ax.twinx()
ax2.plot(daily_population['local_day'], daily_population['n_ids'],
         color='#3498db', linewidth=1, linestyle='--', alpha=0.6, label='N ids')
ax2.set_ylabel('Number of ids', fontsize=11, color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### HR Gap Analysis

# %%
# HR Gap distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('HR Record Gap Statistics (between consecutive raw records)', fontsize=14, fontweight='bold')

gap_cols = [
    ('HR_gap_max_seconds', 'Max Gap per Day', '#e74c3c'),
    ('HR_gap_mean_seconds', 'Mean Gap per Day', '#c0392b'),
    ('HR_gap_median_seconds', 'Median Gap per Day', '#e67e22'),
]

for idx, (col, title, color) in enumerate(gap_cols):
    data = df_daily[col].dropna()
    # clip at 99th percentile for visibility
    clip_val = data.quantile(0.99)
    axes[idx].hist(data.clip(upper=clip_val), bins=60, color=color, edgecolor='black', alpha=0.7)
    axes[idx].axvline(data.median(), color='black', linestyle='--', linewidth=2,
                      label=f'Median: {data.median():.0f}s')
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Seconds')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# %%
# Per-id median HR gap
id_gap = df_daily.groupby('id').agg(
    median_max_gap=('HR_gap_max_seconds', 'median'),
    median_mean_gap=('HR_gap_mean_seconds', 'median'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Per-id Median HR Gaps', fontsize=14, fontweight='bold')

for ax, col, title, color in [
    (axes[0], 'median_max_gap',  'Hist of Daily MAX Gap', '#e74c3c'),
    (axes[1], 'median_mean_gap', 'Hist of Daily MEAN Gap', '#c0392b'),
]:
    data_minutes = (id_gap[col].dropna() / 60)
    ax.hist(data_minutes, bins=40, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(data_minutes.median(), color='black', linestyle='--', linewidth=2,
               label=f'Median: {data_minutes.median():.1f}min')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Number of ids')
    ax.legend()

plt.tight_layout()
plt.show()

# %%
# HR gap over time — does gap quality degrade?
daily_gap_pop = df_daily.groupby('local_day').agg(
    median_mean_gap=('HR_gap_mean_seconds', 'median'),
    q25_mean_gap=('HR_gap_mean_seconds', lambda x: x.quantile(0.25)),
    q75_mean_gap=('HR_gap_mean_seconds', lambda x: x.quantile(0.75)),
).reset_index()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_gap_pop['local_day'], daily_gap_pop['median_mean_gap'],
        color='#e74c3c', linewidth=2, label='Median')
ax.fill_between(daily_gap_pop['local_day'],
                daily_gap_pop['q25_mean_gap'], daily_gap_pop['q75_mean_gap'],
                alpha=0.3, color='#e74c3c', label='IQR')
ax.set_title('Population-Level Mean HR Gap Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Mean Gap Between Records (seconds)')
ax.legend()
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Steps Coverage/Missingness Analysis

# %%
# Days without steps data
df_no_steps = df_daily[df_daily['Steps_raw_records'].isna()].copy()

print("=" * 60)
print("DAYS WITHOUT STEPS DATA (NA) ANALYSIS")
print("=" * 60)
print(f"Total days without steps data: {len(df_no_steps):,}")
print(f"Unique ids affected: {df_no_steps['id'].nunique()}")
print(f"Date range: {df_no_steps['local_day'].min()} to {df_no_steps['local_day'].max()}")

no_steps_per_id = df_no_steps.groupby('id').size().reset_index(name='days_without_steps')
print(f"\nDays without steps per id:")
print(f"  Mean: {no_steps_per_id['days_without_steps'].mean():.1f}")
print(f"  Median: {no_steps_per_id['days_without_steps'].median():.1f}")
print(f"  Min: {no_steps_per_id['days_without_steps'].min()}")
print(f"  Max: {no_steps_per_id['days_without_steps'].max()}")
print("=" * 60)

# %%
# Per-id steps missingness histogram
id_steps_na_pct = df_daily.groupby('id').agg(
    total_days=('local_day', 'count'),
    days_with_na=('Steps_raw_records', lambda x: x.isna().sum())
).reset_index()
id_steps_na_pct['na_percentage'] = (id_steps_na_pct['days_with_na'] / id_steps_na_pct['total_days']) * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(id_steps_na_pct['na_percentage'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(id_steps_na_pct['na_percentage'].mean(), color='darkblue', linestyle='--', linewidth=2,
           label=f"Mean: {id_steps_na_pct['na_percentage'].mean():.1f}%")
ax.axvline(id_steps_na_pct['na_percentage'].median(), color='darkgreen', linestyle='--', linewidth=2,
           label=f"Median: {id_steps_na_pct['na_percentage'].median():.1f}%")
ax.set_xlabel('Percentage of Days Without Steps Data (%)')
ax.set_ylabel('Number of ids')
ax.set_title('Distribution of Steps Data Missingness per id', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"ids with 0% NA: {(id_steps_na_pct['na_percentage'] == 0).sum()}")
print(f"ids with 100% NA: {(id_steps_na_pct['na_percentage'] == 100).sum()}")
print(f"ids with >50% NA: {(id_steps_na_pct['na_percentage'] > 50).sum()}")

# %%
# Steps coverage distributions (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Steps Coverage Metrics — Population Distributions', fontsize=14, fontweight='bold')

steps_cov_cols = [
    ('Steps_raw_records', 'Raw Records per Day', '#3498db'),
    ('Steps_raw_hours_with_records', 'Hours with Records (0-24)', '#2980b9'),
    ('Steps_minutes_per_hour_mean', 'Mean Minutes per Hour', '#1abc9c'),
    ('Steps_gap_mean_seconds', 'Mean Gap Between Records (s)', '#16a085'),
]

for idx, (col, title, color) in enumerate(steps_cov_cols):
    ax = axes[idx // 2, idx % 2]
    data = df_daily[col].dropna()
    clip_val = data.quantile(0.99)
    bins=np.arange(0,24,1)
    ax.hist(data.clip(upper=clip_val), bins=bins, color=color, edgecolor='black', alpha=0.7)
    ax.axvline(data.median(), color='black', linestyle='--', linewidth=2,
               label=f'Median: {data.median():.1f}')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(col)
    ax.legend()

plt.tight_layout()
plt.show()

# %%
# Steps coverage heatmap (first 20 ids)
fig, ax = plt.subplots(figsize=(20, 10))
ids_steps = sorted(df_daily['id'].unique())[:20]
df_steps_sub = df_daily[df_daily['id'].isin(ids_steps)]

heatmap_steps = df_steps_sub.pivot_table(
    index='id', columns='local_day',
    values='Steps_raw_hours_with_records', aggfunc='first'
)
sns.heatmap(heatmap_steps, cmap='YlGnBu', annot=False,
            cbar_kws={'label': 'Hours with Steps Data (0-24)'},
            linewidths=0.5, ax=ax, vmin=0, vmax=24)
ax.set_title('Steps Data Coverage Heatmap (First 20 ids)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('id ID')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Steps coverage vs total daily steps (sanity check)
df_scatter_steps = df_daily.dropna(subset=['Steps_raw_hours_with_records', 'StepsInDay'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Steps: Coverage vs. Volume', fontsize=14, fontweight='bold')

axes[0].scatter(df_scatter_steps['Steps_raw_hours_with_records'], df_scatter_steps['StepsInDay'],
                alpha=0.4, c='#3498db', s=15, edgecolors='black', linewidths=0.3)
axes[0].set_xlabel('Hours with Steps Records')
axes[0].set_ylabel('Total Steps in Day')
axes[0].set_title('Coverage vs Total Steps')
axes[0].grid(alpha=0.3)
if len(df_scatter_steps) > 2:
    r = df_scatter_steps['Steps_raw_hours_with_records'].corr(df_scatter_steps['StepsInDay'])
    axes[0].annotate(f'r = {r:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Comparative: HR vs Steps hours with records per id-day
df_both = df_daily.dropna(subset=['HR_raw_hours_with_records', 'Steps_raw_hours_with_records'])
axes[1].scatter(df_both['HR_raw_hours_with_records'], df_both['Steps_raw_hours_with_records'],
                alpha=0.3, c='#9b59b6', s=15, edgecolors='black', linewidths=0.3)
axes[1].plot([0, 24], [0, 24], 'k--', alpha=0.4, label='y=x')
axes[1].set_xlabel('HR Hours with Records')
axes[1].set_ylabel('Steps Hours with Records')
axes[1].set_title('HR vs Steps Coverage (per day)')
axes[1].set_xlim(0, 24)
axes[1].set_ylim(0, 24)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# load the aggregated daily data
output_path = "/sc-projects/sc-proj-cc15-preact/SP6/preprocessed" + "/passive_daily_aggregated.feather"
df_daily = pd.read_feather(output_path)

# %% [markdown]
# ### check the missing data of `df_daily`

# %%
df_viz = df_daily[
    [
        "id",
        "local_day",
        # sleep TODO - sleep session duration, total sleep time, num_sessions_in_day
        "HR_count",  # it is more the number of seconds with heart rate data than the number of records
        "SPM_count",
        "ACTIVE_n_sessions",
        "BIKE_n_sessions",
        "RUN_n_sessions",
        "WALK_n_sessions",
        "ACTIVE_total_duration",
        "BIKE_total_duration",
        "RUN_total_duration",
        "WALK_total_duration",
        "total_elevation_gain", # TODO include the number of records
        "total_floors_climbed",
    ]
]

# %%

# %%
# Data Availability Heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Prepare data for heatmap - check if data is available (not null)
metrics_to_check = [
    "HR_count", "SPM_count",
    "ACTIVE_n_sessions", "BIKE_n_sessions", "RUN_n_sessions", "WALK_n_sessions",
    "total_elevation_gain", "total_floors_climbed"
]

# Create availability matrix
availability_data = []
for id in sorted(df_viz['id'].unique())[:20]:  # First 20 ids
    id_data = df_viz[df_viz['id'] == id].sort_values('local_day')
    for metric in metrics_to_check:
        availability_data.append({
            'id': id,
            'metric': metric,
            'availability_pct': (id_data[metric].notna().sum() / len(id_data)) * 100
        })

df_availability = pd.DataFrame(availability_data)
heatmap_data = df_availability.pivot(index='id', columns='metric', values='availability_pct')

# Plot heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0], 
            cbar_kws={'label': 'Data Availability (%)'}, linewidths=0.5)
axes[0].set_title('Data Availability by id (First 20 ids)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Metric')
axes[0].set_ylabel('id ID')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Summary statistics table
summary_stats = df_viz[metrics_to_check].describe().T
summary_stats['missing_pct'] = (df_viz[metrics_to_check].isna().sum() / len(df_viz)) * 100
summary_stats = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_pct']]
summary_stats.columns = ['Count', 'Mean', 'Std', 'Min', 'Median', 'Max', 'Missing %']

# Plot table
axes[1].axis('tight')
axes[1].axis('off')
table = axes[1].table(cellText=summary_stats.round(2).values,
                      rowLabels=summary_stats.index,
                      colLabels=summary_stats.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.12]*len(summary_stats.columns))
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code the header
for i in range(len(summary_stats.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code the row labels
for i in range(len(summary_stats)):
    table[(i+1, -1)].set_facecolor('#ecf0f1')
    table[(i+1, -1)].set_text_props(weight='bold')

axes[1].set_title('Summary Statistics for Key Metrics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create subset of key metrics
viz_cols = [
    "id", "local_day",
    "HR_count", "SPM_count",
    "ACTIVE_n_sessions", "BIKE_n_sessions", "RUN_n_sessions", "WALK_n_sessions",
    "ACTIVE_total_duration", "BIKE_total_duration", "RUN_total_duration", "WALK_total_duration",
    "total_elevation_gain", "total_floors_climbed"
]

df_viz = df_daily[viz_cols].copy()

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))
gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Heart Rate Count Distribution
ax1 = fig.add_subplot(gs[0, 0])
df_viz['HR_count'].dropna().hist(bins=50, ax=ax1, color='#e74c3c', edgecolor='black', alpha=0.7)
ax1.set_title('Heart Rate Data Count Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('HR Count (seconds)')
ax1.set_ylabel('Frequency')
ax1.axvline(df_viz['HR_count'].median(), color='darkred', linestyle='--', linewidth=2, label=f'Median: {df_viz["HR_count"].median():.0f}')
ax1.legend()

# 2. Steps Count Distribution
ax2 = fig.add_subplot(gs[0, 1])
df_viz['SPM_count'].dropna().hist(bins=50, ax=ax2, color='#3498db', edgecolor='black', alpha=0.7)
ax2.set_title('Steps Count Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Steps Count')
ax2.set_ylabel('Frequency')
ax2.axvline(df_viz['SPM_count'].median(), color='darkblue', linestyle='--', linewidth=2, label=f'Median: {df_viz["SPM_count"].median():.0f}')
ax2.legend()

# 3. Activity Sessions - Bar plot
ax3 = fig.add_subplot(gs[0, 2])
session_cols = ['ACTIVE_n_sessions', 'BIKE_n_sessions', 'RUN_n_sessions', 'WALK_n_sessions']
session_totals = df_viz[session_cols].sum()
colors_sessions = ['#9b59b6', '#1abc9c', '#f39c12', '#2ecc71']
session_totals.plot(kind='bar', ax=ax3, color=colors_sessions, edgecolor='black', alpha=0.8)
ax3.set_title('Total Activity Sessions by Type', fontsize=12, fontweight='bold')
ax3.set_ylabel('Total Sessions')
ax3.set_xticklabels(['Active', 'Bike', 'Run', 'Walk'], rotation=45)
ax3.grid(axis='y', alpha=0.3)

# 4. Activity Duration - Bar plot
ax4 = fig.add_subplot(gs[1, 0])
duration_cols = ['ACTIVE_total_duration', 'BIKE_total_duration', 'RUN_total_duration', 'WALK_total_duration']
duration_totals = df_viz[duration_cols].sum() / 60  # Convert to hours
duration_totals.plot(kind='bar', ax=ax4, color=colors_sessions, edgecolor='black', alpha=0.8)
ax4.set_title('Total Activity Duration by Type', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Duration (hours)')
ax4.set_xticklabels(['Active', 'Bike', 'Run', 'Walk'], rotation=45)
ax4.grid(axis='y', alpha=0.3)

# 5. Elevation Gain Distribution
ax5 = fig.add_subplot(gs[1, 1])
df_viz['total_elevation_gain'].dropna().hist(bins=50, ax=ax5, color='#e67e22', edgecolor='black', alpha=0.7)
ax5.set_title('Elevation Gain Distribution', fontsize=12, fontweight='bold')
ax5.set_xlabel('Elevation Gain (meters)')
ax5.set_ylabel('Frequency')
ax5.axvline(df_viz['total_elevation_gain'].median(), color='darkorange', linestyle='--', linewidth=2, 
            label=f'Median: {df_viz["total_elevation_gain"].median():.1f}m')
ax5.legend()

# 6. Floors Climbed Distribution
ax6 = fig.add_subplot(gs[1, 2])
df_viz['total_floors_climbed'].dropna().hist(bins=50, ax=ax6, color='#16a085', edgecolor='black', alpha=0.7)
ax6.set_title('Floors Climbed Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('Floors Climbed')
ax6.set_ylabel('Frequency')
ax6.axvline(df_viz['total_floors_climbed'].median(), color='darkgreen', linestyle='--', linewidth=2, 
            label=f'Median: {df_viz["total_floors_climbed"].median():.1f}')
ax6.legend()

# 7. Time Series - Heart Rate (sample id)
ax7 = fig.add_subplot(gs[2, :])
sample_ids = df_viz['id'].unique()[:5]  # First 5 ids
for i, id in enumerate(sample_ids):
    id_data = df_viz[df_viz['id'] == id].sort_values('local_day')
    ax7.plot(id_data['local_day'], id_data['HR_count'], 
             marker='o', markersize=3, alpha=0.7, label=f'id {id}')
ax7.set_title('Heart Rate Data Count Over Time (Sample ids)', fontsize=12, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('HR Count')
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax7.grid(alpha=0.3)
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

# 8. Time Series - Steps (sample id)
ax8 = fig.add_subplot(gs[3, :])
for i, id in enumerate(sample_ids):
    id_data = df_viz[df_viz['id'] == id].sort_values('local_day')
    ax8.plot(id_data['local_day'], id_data['SPM_count'], 
             marker='o', markersize=3, alpha=0.7, label=f'id {id}')
ax8.set_title('Steps Count Over Time (Sample ids)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Date')
ax8.set_ylabel('Steps Count')
ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax8.grid(alpha=0.3)
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle('Passive Data Daily Aggregation Overview', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize Daily Passive Data

# %%
# Compute missing data summary -> the number of missing days per id per feature

# Get date range per id
id_dates = df_daily.groupby('id')['local_day'].agg(['min', 'max']).reset_index()

# Create complete date range for each id
missing_summary = []
for _, row in id_dates.iterrows():
    id = row['id']
    date_range = pd.date_range(start=row['min'], end=row['max'], freq='D')
    
    # Get actual data for this id
    id_data = df_daily[df_daily['id'] == id].set_index('local_day')
    
    # Count missing days for each feature (column)
    for col in df_daily.columns:
        if col not in ['id', 'local_day']:
            # Reindex to full date range and count nulls
            full_range_data = id_data[col].reindex(date_range)
            n_missing = full_range_data.isna().sum()
            total_days = len(date_range)
            
            missing_summary.append({
                'id': id,
                'feature': col,
                'n_missing_days': n_missing,
                'total_days': total_days,
                'missing_pct': (n_missing / total_days) * 100
            })

df_missing_summary = pd.DataFrame(missing_summary)
print(f"Missing data summary shape: {df_missing_summary.shape}")
df_missing_summary


# %% [markdown]
# ## missing data of `df_backup`

# %%
df_backup["local_timestamp_start"].dt.date

# %%
df_backup["local_day"] = df_backup["local_timestamp_start"].dt.floor('d')

# %%
df_backup

# %%
df = df_backup[df_backup["id"] == "0ePW"].copy()

# %%
df["local_day"]

# %%
df["day_from_study_start"] = (df["local_day"] - df["local_day"].min()).dt.days

# %%
df[df["modality"] == "HeartRate"]
df.groupby("day_from_study_start")

# %%
df[df["modality"] == "HeartRate"]

# %%
df_daily.columns.tolist()

# %%
