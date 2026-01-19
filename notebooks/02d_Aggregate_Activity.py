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
# # Activity Aggregate

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

# %% [markdown]
# ### Thryve Activity Type mappings

# %%
map_ActivityType = {
    102: "SLEEP", # missing from the official website
    103: "REST",
    104: "ACTIVE",
    105: "WALK",
    106: "RUN",
    107: "BIKE",
    108: "TRANSPORT",
    110: "LEISURE" # no records
}

# %%
map_ActivityTypeDetail1 = {
    203: "MINDFULNESS",
    204: "ACTIVE",
    205: "WALK",
    206: "RUN",
    207: "BIKE",
    208: "TRANSPORT",
    210: "LEISURE",
    211: "BALL_SPORTS",
    212: "GYM_SPORTS",
    213: "FIGHT_SPORTS",
    214: "WATER_SPORTS",
    215: "WINTER_SPORTS",
    216: "ENDURANCE",
    217: "RACING_SPORTS",
    219: "E_SPORTS",
    220: "AIR_SPORTS"
}

# %%
map_ActivityTypeDetail2 = {
    303: "MINDFULNESS",
    304: "ACTIVE",
    305: "WALK",
    306: "RUN",
    307: "BIKE",
    308: "AEROBICS",
    310: "LEISURE",
    313: "GYM_SPORTS",
    314: "PILATES",
    315: "SLEDDING",
    316: "HIIT",
    317: "MOUNTAIN_BIKING",
    320: "WAKEBOARDING",
    321: "SWIMMING",
    322: "DIVING",
    323: "YOGA",
    324: "HIKING",
    325: "CLIMBING",
    326: "CLIMBING_STAIRS",
    327: "DANCING",
    328: "SKATING",
    329: "HORSEBACK_RIDING",
    330: "SKIING",
    331: "SNOWBOARDING",
    332: "ICE_SKATING",
    333: "CROSS_COUNTRY_SKIING",
    334: "SURFING",
    335: "ROWING",
    336: "SAILING",
    337: "SPINNING",
    338: "NORDIC_WALKING",
    339: "RUGBY",
    340: "SOCCER",
    341: "HANDBALL",
    342: "BASEBALL",
    343: "BASKETBALL",
    344: "TENNIS",
    345: "TABLE_TENNIS",
    346: "BADMINTON",
    347: "VOLLEYBALL",
    348: "GOLFING",
    349: "FOOTBALL",
    350: "BOXING",
    351: "MARTIAL_ARTS",
    352: "WRESTLING",
    354: "HOCKEY",
    356: "SOFTBALL",
    357: "PADEL",
    358: "SQUASH",
    359: "CRICKET",
    360: "ZUMBA",
    361: "WHEELCHAIR",
    362: "KAYAKING",
    363: "WALKING_USING_CRUTCHES",
    364: "PARAGLIDING",
    365: "SKYDIVING",
    370: "FISHING",
    371: "HUNTING",
    372: "CURLING",
    373: "FRISBEE",
    374: "LACROSSE",
    378: "STRETCHING",
    379: "MEDITATION",
    380: "HOUSEHOLD",
    381: "FENCING",
    382: "TRAMPOLINING",
    383: "ARCHERY",
    385: "TRIATHLON",
    386: "BOWLING",
    387: "BODYBUILDING",
    388: "BILLIARDS",
    389: "DARTS",
    390: "GYMNASTICS",
    391: "KITE",
    392: "ORIENTEERING",
    393: "POLO",
    394: "SNOWSHOEING",
    395: "KITESURFING",
    396: "RACING",
    397: "MOTOCROSS",
    398: "STANDUP_PADDLEBOARDING",
    399: "WATER_RUNNING",
    400: "MINDFUL_ACTIVITY",
    401: "WHITEWATER_RAFTING",
    402: "HULA_HOOPING",
    403: "CROSSFIT",
    404: "ROWING_MACHINE",
    405: "ELLIPTICAL",
    406: "CANOEING",
    407: "WATER_AEROBICS",
    408: "WINDSURFING",
    409: "ICE_HOCKEY",
    410: "PICKLEBALL",
    411: "WATER_SKIING",
    412: "SKATEBOARDING",
    413: "SNOWKITING",
    414: "E_SPORTS",
    415: "WEIGHT_TRAINING",
    416: "WEIGHT_MACHINE_TRAINING",
    417: "TREADMILL_RUNNING",
    418: "TREADMILL_WALKING",
    419: "RACQUETBALL",
    420: "ROCK_CLIMBING",
    421: "ROPE_JUMPING",
    422: "KETTLEBELL_WORKOUT",
    423: "AIR_SPORTS",
    424: "TRANSPORT",
    425: "WATER_SPORTS",
    426: "BALL_SPORTS",
    427: "WINTER_SPORTS",
    428: "ENDURANCE",
    429: "FIGHT_SPORTS",
}

# %%

# %%
df_backup[df_backup["type"] == "ActivityType"].longValue.value_counts().sort_values(
    ascending=False
).rename(index=map_ActivityType)
# ? 102 is sleep -> google sheet https://docs.google.com/spreadsheets/d/1HLcw2o_qERfLxcVT01D-RCqkhgV2E5llKYNh7cUZgLQ/edit?gid=0#gid=0

# %%
df_backup[
    df_backup["type"] == "ActivityTypeDetail1"
].longValue.value_counts().sort_values(ascending=False).rename(
    index=map_ActivityTypeDetail1
)

# %%
df_backup[
    df_backup["type"] == "ActivityTypeDetail2"
].longValue.value_counts().sort_values(ascending=False).rename(
    index=map_ActivityTypeDetail2
)

# %%
# TODO check it on the customer level - how many customers have activity each type


# %%
df_backup.type.value_counts()

# %% [markdown]
# ## ActivityType

# %% [markdown]
# do also ElevationGain & FloorsClimbed

# %% [markdown]
# exclude sleep here, as it is done separately
#
# - [ ] #TODO, might comeback to SLEEP from ActivityType to "double-check"/measure the reliability of sleep data

# %%
df = df_backup[df_backup["type"] == "ActivityType"].copy()

df["activity_type_str"] = df["longValue"].map(map_ActivityType)

# df = df[df["activity_type_str"] != "SLEEP"].copy()
df = df[~df["activity_type_str"].isin(["SLEEP", "REST"])].copy()

df

# %%
df["local_day"] = df["local_start_time"].dt.floor("d")

# %%
df["duration"] = (df["endTimestamp"] - df["startTimestamp"]).dt.total_seconds() / 60
df["duration"].describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# %%
df

# %% [markdown]
# - [ ] don't know what much else to aggregate here
#     - some peak hours? - don't know how demanding the activities were
#     - first start and last end time maybe?
#     - number of distinct records per day? - probably tells more about the activity classification algorithm than actual activities

# %%
# df.groupby(["customer", "local_day", "activity_type_str"])["duration"].sum().reset_index()
df_agg = df.groupby(["customer", "local_day", "activity_type_str"]).agg(
    total_duration=("duration", "sum"),
    n_sessions=("duration", "count"),          # Number of distinct records
    avg_session_duration=("duration", "mean"), # Average length of a session
    max_session_duration=("duration", "max"),  # Longest single session
    first_start_time=("local_start_time", "min"), # Earliest time activity occurred
    last_start_time=("local_start_time", "max")   # Latest time activity occurred
).reset_index()

df_agg

# %%
# plot the number of records per activity type with time (local_day) on x axis
import seaborn as sns
activity_counts = (
    df.groupby(["local_day", "activity_type_str"])
    .size()
    .reset_index(name="counts")
)
sns.lineplot(data=activity_counts, x="local_day", y="counts", hue="activity_type_str")


# %%

# %%

# %%

# %% [markdown]
# ## ActivityTypeDetail1

# %%
df_detail1 = df_backup[df_backup["type"] == "ActivityTypeDetail1"].copy()

df_detail1["activity_type_str"] = df_detail1["longValue"].map(map_ActivityTypeDetail1)

df_detail1["local_day"] = df_detail1["local_start_time"].dt.floor("d")
df_detail1["duration"] = (df_detail1["endTimestamp"] - df_detail1["startTimestamp"]).dt.total_seconds() / 60

# %%
df_detail1["duration"].describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# %%
df_detail1_agg = df_detail1.groupby(["customer", "local_day", "activity_type_str"]).agg(
    total_duration=("duration", "sum"),
    n_sessions=("duration", "count"),          # Number of distinct records
    avg_session_duration=("duration", "mean"), # Average length of a session
    max_session_duration=("duration", "max"),  # Longest single session
    first_start_time=("local_start_time", "min"), # Earliest time activity occurred
    last_start_time=("local_start_time", "max")   # Latest time activity occurred
).reset_index()

df_detail1_agg

# %%
# plot the number of records per activity type with time (local_day) on x axis
activity_counts_d1 = (
    df_detail1.groupby(["local_day", "activity_type_str"])
    .size()
    .reset_index(name="counts")
)
sns.lineplot(data=activity_counts_d1, x="local_day", y="counts", hue="activity_type_str")


# %% [markdown]
# ## ActivityTypeDetail2

# %%
df_detail2 = df_backup[df_backup["type"] == "ActivityTypeDetail2"].copy()

df_detail2["activity_type_str"] = df_detail2["longValue"].map(map_ActivityTypeDetail2)

df_detail2["local_day"] = df_detail2["local_start_time"].dt.floor("d")
df_detail2["duration"] = (df_detail2["endTimestamp"] - df_detail2["startTimestamp"]).dt.total_seconds() / 60

# %%
df_detail2["duration"].describe([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# %%
df_detail2_agg = df_detail2.groupby(["customer", "local_day", "activity_type_str"]).agg(
    total_duration=("duration", "sum"),
    n_sessions=("duration", "count"),          # Number of distinct records
    avg_session_duration=("duration", "mean"), # Average length of a session
    max_session_duration=("duration", "max"),  # Longest single session
    first_start_time=("local_start_time", "min"), # Earliest time activity occurred
    last_start_time=("local_start_time", "max")   # Latest time activity occurred
).reset_index()

df_detail2_agg

# %%
# plot the number of records per activity type with time (local_day) on x axis
activity_counts_d2 = (
    df_detail2.groupby(["local_day", "activity_type_str"])
    .size()
    .reset_index(name="counts")
)
sns.lineplot(data=activity_counts_d2, x="local_day", y="counts", hue="activity_type_str")
plt.yscale("log")

# %% [markdown]
# ## Compare the aggregated frames
# Compare WALK activity across ActivityType, ActivityTypeDetail1, and ActivityTypeDetail2

# %%
# Filter for WALK in each aggregated frame - include all metrics
walk_agg = df_agg[df_agg["activity_type_str"] == "WALK"][["customer", "local_day", "total_duration", "n_sessions", "first_start_time", "last_start_time"]].copy()
walk_d1 = df_detail1_agg[df_detail1_agg["activity_type_str"] == "WALK"][["customer", "local_day", "total_duration", "n_sessions", "first_start_time", "last_start_time"]].copy()
walk_d2 = df_detail2_agg[df_detail2_agg["activity_type_str"] == "WALK"][["customer", "local_day", "total_duration", "n_sessions", "first_start_time", "last_start_time"]].copy()

print(f"WALK rows - ActivityType: {len(walk_agg)}, Detail1: {len(walk_d1)}, Detail2: {len(walk_d2)}")

# %%
# Merge the three WALK frames on customer and local_day
walk_merged = walk_agg.merge(
    walk_d1, on=["customer", "local_day"], how="outer", suffixes=("_act", "_d1")
).merge(
    walk_d2.rename(columns={
        "total_duration": "total_duration_d2",
        "n_sessions": "n_sessions_d2",
        "first_start_time": "first_start_time_d2",
        "last_start_time": "last_start_time_d2"
    }),
    on=["customer", "local_day"],
    how="outer"
)

walk_merged

# %%
# Visualize: scatter plot comparing total_duration across levels
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

max_act = walk_merged["total_duration_act"].max()
max_d1 = walk_merged["total_duration_d1"].max()

axes[0].scatter(walk_merged["total_duration_act"], walk_merged["total_duration_d1"], alpha=0.3)
axes[0].plot([0, max_act], [0, max_act], 'r--')
axes[0].set_xlabel("ActivityType (min)")
axes[0].set_ylabel("Detail1 (min)")
axes[0].set_title("WALK: ActivityType vs Detail1")

axes[1].scatter(walk_merged["total_duration_act"], walk_merged["total_duration_d2"], alpha=0.3)
axes[1].plot([0, max_act], [0, max_act], 'r--')
axes[1].set_xlabel("ActivityType (min)")
axes[1].set_ylabel("Detail2 (min)")
axes[1].set_title("WALK: ActivityType vs Detail2")

axes[2].scatter(walk_merged["total_duration_d1"], walk_merged["total_duration_d2"], alpha=0.3)
axes[2].plot([0, max_d1], [0, max_d1], 'r--')
axes[2].set_xlabel("Detail1 (min)")
axes[2].set_ylabel("Detail2 (min)")
axes[2].set_title("WALK: Detail1 vs Detail2")

plt.tight_layout()
plt.show()


# %%
# Correlation matrix for total_duration across the three levels
walk_merged[["total_duration_act", "total_duration_d1", "total_duration_d2"]].corr()

# %%
# Correlation matrix for n_sessions across the three levels
walk_merged[["n_sessions_act", "n_sessions_d1", "n_sessions_d2"]].corr()

# %%
# Convert first_start_time and last_start_time to hours since midnight (time of day)
for suffix in ["_act", "_d1", "_d2"]:
    walk_merged[f"first_start_hour{suffix}"] = walk_merged[f"first_start_time{suffix}"].dt.hour + walk_merged[f"first_start_time{suffix}"].dt.minute / 60
    walk_merged[f"last_start_hour{suffix}"] = walk_merged[f"last_start_time{suffix}"].dt.hour + walk_merged[f"last_start_time{suffix}"].dt.minute / 60

# %%
# Correlation matrix for first_start_hour across the three levels
walk_merged[["first_start_hour_act", "first_start_hour_d1", "first_start_hour_d2"]].corr()

# %%
# Correlation matrix for last_start_hour across the three levels
walk_merged[["last_start_hour_act", "last_start_hour_d1", "last_start_hour_d2"]].corr()
