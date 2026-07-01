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
# %%
import os
import sys
from functools import reduce
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
    proj_sheet,
    preprocessed_path,
)

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
    ids_to_keep = df_backup["id"].unique()[100:110]
    df_backup = df_backup[df_backup["id"].isin(ids_to_keep)].copy()
    print(f"Unique IDs kept for debugging: {ids_to_keep}")
    print(f"Data shape after filtering for debug: {df_backup.shape}")


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

print(f"Total id-type combinations: {len(df_backup.groupby('id')['modality'].value_counts())}")
print(f"Unique ids: {df_backup['id'].nunique()}")
print(f"Unique types: {df_backup['modality'].nunique()}")

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
# df_daily

# %%
# # save the aggregated daily data
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON: Saving locally inside the repository!!!!!!!!!!!!!")
    output_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
    output_path = output_dir / "daily_aggregated_all_passive_debug.feather"
else:
    passive_daily_dir = Path("/sc-projects/sc-proj-cc15-preact/SP6/preprocessed/passive/daily/")
    output_path = passive_daily_dir / "daily_aggregated_all_passive.feather"

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

    if debug:
        single_save_path = output_dir / f"daily_agg_{key}_debug.parquet"
        df.to_parquet(single_save_path)
        print(f"[DEBUG] {key} daily data saved locally to: {single_save_path}")
    else:
        single_save_path = passive_daily_dir / f"daily_agg_{key}.parquet"
        df.to_parquet(single_save_path)
        print(f"{key} daily data saved to: {single_save_path}")

# %% [markdown]
# ## Summary Visualizations

# %%
# Plot non-null counts of each key metric in df_daily
key_metrics = [
    "HR_count", "SPM_count", "ACTIVE_n_sessions", "total_elevation_gain", "total_floors_climbed"
]
available_metrics = [m for m in key_metrics if m in df_daily.columns]
if available_metrics:
    plt.figure(figsize=(10, 5))
    df_daily[available_metrics].notna().sum().plot(kind="bar", color="#3498db", edgecolor="black")
    plt.title("Number of Day-Records containing each Passive Data Modality")
    plt.ylabel("Non-Null Count")
    plt.xlabel("Modality")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# %%
