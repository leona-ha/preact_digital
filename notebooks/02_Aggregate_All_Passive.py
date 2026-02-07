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

from functions.preprocessing.aggregation import aggregate_all_passive
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

# %% [markdown]
# ### check the other types

# %%

# %%
type_coverage = (
    df_backup.groupby("type")["customer"].nunique().reset_index(name="unique_customers")
)
type_coverage["percentage_customers"] = (
    type_coverage["unique_customers"] / df_backup["customer"].nunique()
) * 100
type_coverage = type_coverage.merge(
    df_backup["type"].value_counts().reset_index(name="total_entries"),
    on="type",
    how="left",
)
type_coverage.sort_values(by="total_entries", ascending=False, inplace=True)
type_coverage.reset_index(drop=True, inplace=True)
type_coverage


# %%

# %%
df_backup.groupby("customer")["type"].value_counts()

print(f"Total customer-type combinations: {len(df_backup.groupby('customer')['type'].value_counts())}")
print(f"Unique customers: {df_backup['customer'].nunique()}")
print(f"Unique types: {df_backup['type'].nunique()}")

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
df_hr_daily = aggregate_hr_daily(df_backup)
df_hr_daily.head()

# %% [markdown]
# ### Steps

# %%
# cProfile.run('aggregate_steps_daily(df_backup)')

# %%
df_steps_daily = aggregate_steps_daily(df_backup)
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
# Aggregate all passive data

df_daily = df_sleep_daily.merge(df_hr_daily, on=["customer", "local_day"], how="outer")
df_daily = df_daily.merge(df_steps_daily, on=["customer", "local_day"], how="outer")
df_daily = df_daily.merge(df_activity_daily, on=["customer", "local_day"], how="outer")
df_daily = df_daily.merge(df_elevation_daily, on=["customer", "local_day"], how="outer")
df_daily = df_daily.merge(df_floors_daily, on=["customer", "local_day"], how="outer")

# %%
# or just call aggregate_all_passive
df_daily = aggregate_all_passive(df_backup)

df_daily

# %%
print(f"Total rows: {len(df_daily)}")
print(f"Total columns: {len(df_daily.columns)}")
print(f"Unique customers: {df_daily['customer'].nunique()}")
print(f"Date range: {df_daily['local_day'].min()} to {df_daily['local_day'].max()}")

# %%
df_daily

# %%
# save the aggregated daily data
output_path = "/sc-projects/sc-proj-cc15-preact/SP6/preprocessed" + "/passive_daily_aggregated.feather"
print(f"Aggregated daily data saved to: {output_path}")
df_daily.to_feather(output_path)

# %%
# load the aggregated daily data
output_path = "/sc-projects/sc-proj-cc15-preact/SP6/preprocessed" + "/passive_daily_aggregated.feather"
df_daily = pd.read_feather(output_path)

# %% [markdown]
# ### check the missing data of `df_daily`

# %%
df_viz = df_daily[
    [
        "customer",
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
for customer in sorted(df_viz['customer'].unique())[:20]:  # First 20 customers
    customer_data = df_viz[df_viz['customer'] == customer].sort_values('local_day')
    for metric in metrics_to_check:
        availability_data.append({
            'customer': customer,
            'metric': metric,
            'availability_pct': (customer_data[metric].notna().sum() / len(customer_data)) * 100
        })

df_availability = pd.DataFrame(availability_data)
heatmap_data = df_availability.pivot(index='customer', columns='metric', values='availability_pct')

# Plot heatmap
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0], 
            cbar_kws={'label': 'Data Availability (%)'}, linewidths=0.5)
axes[0].set_title('Data Availability by Customer (First 20 Customers)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Metric')
axes[0].set_ylabel('Customer ID')
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
    "customer", "local_day",
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

# 7. Time Series - Heart Rate (sample customer)
ax7 = fig.add_subplot(gs[2, :])
sample_customers = df_viz['customer'].unique()[:5]  # First 5 customers
for i, customer in enumerate(sample_customers):
    customer_data = df_viz[df_viz['customer'] == customer].sort_values('local_day')
    ax7.plot(customer_data['local_day'], customer_data['HR_count'], 
             marker='o', markersize=3, alpha=0.7, label=f'Customer {customer}')
ax7.set_title('Heart Rate Data Count Over Time (Sample Customers)', fontsize=12, fontweight='bold')
ax7.set_xlabel('Date')
ax7.set_ylabel('HR Count')
ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax7.grid(alpha=0.3)
plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)

# 8. Time Series - Steps (sample customer)
ax8 = fig.add_subplot(gs[3, :])
for i, customer in enumerate(sample_customers):
    customer_data = df_viz[df_viz['customer'] == customer].sort_values('local_day')
    ax8.plot(customer_data['local_day'], customer_data['SPM_count'], 
             marker='o', markersize=3, alpha=0.7, label=f'Customer {customer}')
ax8.set_title('Steps Count Over Time (Sample Customers)', fontsize=12, fontweight='bold')
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
# Compute missing data summary -> the number of missing days per customer per feature

# Get date range per customer
customer_dates = df_daily.groupby('customer')['local_day'].agg(['min', 'max']).reset_index()

# Create complete date range for each customer
missing_summary = []
for _, row in customer_dates.iterrows():
    customer = row['customer']
    date_range = pd.date_range(start=row['min'], end=row['max'], freq='D')
    
    # Get actual data for this customer
    customer_data = df_daily[df_daily['customer'] == customer].set_index('local_day')
    
    # Count missing days for each feature (column)
    for col in df_daily.columns:
        if col not in ['customer', 'local_day']:
            # Reindex to full date range and count nulls
            full_range_data = customer_data[col].reindex(date_range)
            n_missing = full_range_data.isna().sum()
            total_days = len(date_range)
            
            missing_summary.append({
                'customer': customer,
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
df_backup["local_start_time"].dt.date

# %%
df_backup["local_day"] = df_backup["local_start_time"].dt.floor('d')

# %%
df_backup

# %%
df = df_backup[df_backup["customer"] == "0ePW"].copy()

# %%
df["local_day"]

# %%
df["day_from_study_start"] = (df["local_day"] - df["local_day"].min()).dt.days

# %%
df[df["type"] == "HeartRate"]
df.groupby("day_from_study_start")

# %%
df[df["type"] == "HeartRate"]

# %%
df_daily.columns.tolist()

# %%
