# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: TessaPyEnv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Preprocess 

# %% [markdown]
# This script concatenates most recent passive data with backup data and performs basic preprocessing on passive, EMA and Monitoring data.

# %%
from pyprojroot import here
import sys
sys.path.insert(0, str(here()))

from pathlib import Path
from datetime import date
import pandas as pd
import gc  
import os
import glob
import numpy as np
import pickle
import plotly.express as px

# --- Paths / imports -------------------------------------------------

PROJECT_ROOT = here()

from server_config import base_path, proj_sheet
base_path = Path(base_path)  # convert to Path

raw_path = str(base_path / "raw")
preprocessed_path = str(base_path / "preprocessed")
backup_path = str(base_path / "backup")

from src.preprocessing.infer_timeoffset import (
    create_utcday_tzoffset_df,
    merge_fill_tz,
)
# --- Dates ------------------------------------------------------------
today_str = "03082026"
#today_str = date.today().strftime("%d%m%Y")        
today_day = pd.Timestamp.today().normalize()       

# --- Path -------------------------------------------------------------
datapath = Path(raw_path) / f"export_tiki_{today_str}"  

# %% [markdown]
# ## 1. Passive Data

# %% [markdown]
# ### 1.1 Load most recent passive data

# %%
# actual passive + ema_data

# define the pattern for passive data files
file_pattern = os.path.join(datapath, "epoch_part*.csv")

# use glob to find all matching files
file_list = glob.glob(file_pattern)

# sort the file list for consistent ordering
file_list.sort()

# concatenate all CSV files into a single DataFrame
df_complete = pd.concat((pd.read_csv(f, encoding="latin-1", low_memory=False) for f in file_list), ignore_index=True)


# %%
# Extract customer identifier and reduce to first 4 characters
df_complete["customer"] = df_complete.customer.str.split("@").str.get(0)
df_complete["customer"] = df_complete["customer"].str[:4]

for col in ["startTimestamp", "endTimestamp"]:
    df_complete[col] = (
        pd.to_datetime(df_complete[col], utc=True, errors="coerce", unit="ms")
    )

#df_complete

# %%
df_complete = df_complete[['customer', 'startTimestamp', 'endTimestamp', 'timezoneOffset', 'type',
       'stringValue', 'booleanValue', 'doubleValue', 'longValue']]

# %% [markdown]
# ### 1.2 Load big backup dataset

# %%
# merge with backup data
backup_path = preprocessed_path + "/backup_passive_05052025.parquet"

# backup_path = preprocessed_path + "/backup_passive_last.feather"
df_backup = pd.read_parquet(backup_path)

# %%
# make independent copies of both DataFrames to avoid SettingWithCopyWarning (future modifications do not affect any original DataFrame)
df_backup = df_backup.copy()
df_complete = df_complete.copy()

# convert booleanValue to boolean[pyarrow] dtype before concatenation so that it can be saved to .feather later
# alternative to "boolean[pyarrow]" is "boolean", but it is experimental and may change in future pandas versions
df_backup['booleanValue'] = df_backup['booleanValue'].astype('boolean[pyarrow]')
df_complete['booleanValue'] = df_complete['booleanValue'].astype('boolean[pyarrow]')

# %%
# latest timestamp from the backup dataset
latest_timestamp = df_backup['startTimestamp'].max()

# filter from df_complete only those entries that are newer than what’s already in the backup
df_complete_filtered = df_complete[df_complete['startTimestamp'] > latest_timestamp]

# %% [markdown]
# ### 1.3 Concat Backup and most recent data

# %%
# update the backup by concatenating only the newly filtered rows from df_complete, creating an up-to-date backup
df_backup_recent = pd.concat([df_backup, df_complete_filtered], ignore_index=True)

# %% [markdown]
# ### 1.4 Rename variable names and create additional columns 

# %%
# define a clear mapping for rename columns
rename_columns = {"customer": "id",
                  "type": "modality",
                  "startTimestamp": "timestamp_start",
                  "endTimestamp": "timestamp_end",
                  "booleanValue": "boolean_value",
                  "doubleValue": "double_value",
                  "longValue": "long_value",
                  "timezoneOffset": "timezone_offset"}

# apply renaming
df_backup_recent = df_backup_recent.rename(columns=rename_columns)

# %%
# create a unified float_value column:
# use 'doubleValue' where available (more precise), otherwise use 'longValue'
df_backup_recent['float_value'] = df_backup_recent['double_value'].fillna(df_backup_recent['long_value'])


# %%
# drop original value columns that have been unified into 'float_value' + 'stringValue' (because only ECG data are stored as string for period March - November 2023) + 'createdtAt'
df_backup_recent = df_backup_recent.drop(columns=['double_value', 'long_value', 'stringValue'])


# %%
# create a time_interval (duration in seconds) column
df_backup_recent['time_interval'] = (
    df_backup_recent['timestamp_end'] - df_backup_recent['timestamp_start']
).dt.total_seconds()

# create a start_date and start_hour column
df_backup_recent['start_date']  = df_backup_recent['timestamp_start'].dt.normalize()
df_backup_recent['start_hour'] = df_backup_recent['timestamp_start'].dt.hour

# %% [markdown]
# ### 1.5 Infer Timezone Offset

# %%
df_tz = create_utcday_tzoffset_df(df_backup_recent)


# %%
df_tz["inferred_tzoffset_timedelta"] = pd.to_timedelta(
    df_tz["inferred_tzoffset"], unit="min"
)

# %%
df_backup_recent = df_backup_recent.merge(
    df_tz,
    left_on=["id", "start_date"],
    right_on=["id", "day"],
    how="left",
)
#df_backup_recent.drop(columns=["day"], inplace=True)  # remove day from df_tz

# %%
df_backup_recent["local_timestamp_start"] = (
    df_backup_recent["timestamp_start"] + df_backup_recent["inferred_tzoffset_timedelta"]
).dt.tz_localize(None)

df_backup_recent["local_timestamp_end"] = (
    df_backup_recent["timestamp_end"] + df_backup_recent["inferred_tzoffset_timedelta"]
).dt.tz_localize(None)


# %%
assert df_backup_recent.inferred_tzoffset.isna().sum() == 0, (
    "There are missing inferred timezone offsets!"
)

# %%
# just to make sure that we don't use them anymore later
del df_complete_filtered
del df_complete

# %% [markdown]
# Next:
#
# 1. Since we want to include 'for_id' and 'study_version' in our `passive_data` data frame, we need to extract these data from the monitoring sheet. This is done in section 2
#
# 2. Additionally the `monitoring_data` data frame is set up in section 2
#

# %% [markdown]
# ## 2. Monitoring data

# %%
# import data
df_monitoring = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{proj_sheet}/export?format=csv")

# %%
# get an overview of the monitoring data
#df_monitoring.head()

# %%
df_monitoring = df_monitoring.copy()

df_monitoring.rename(columns = {"Pseudonym": "id",
                                "FOR_ID": "for_id",
                                "EMA_ID": "ema_id", 
                                "Status": "study_status",
                                "Studienversion":"study_version", 
                                "Start EMA Baseline": "ema_base_start", 
                                "Ende EMA Baseline": "ema_base_end", 
                                "Freischaltung/ Start EMA T20": "ema_t20_start",
                                "Ende EMA T20":"ema_t20_end", 
                                "Freischaltung/ Start EMA Post":"ema_post_start",
                               "Ende EMA Post":"ema_post_end", 
                                "T20=Post":"t20_post" }, 
                     inplace=True)

df_monitoring = df_monitoring[['for_id', 'ema_id', 'id', 'study_version', 'study_status',
       't20_post', 'ema_base_start', 'ema_base_end', 'ema_t20_start', 'ema_t20_end',
       'ema_post_start', 'ema_post_end']]

df_monitoring["id"] = df_monitoring["id"].str[:4]
df_monitoring["for_id"] = df_monitoring.for_id.str.strip()

df_monitoring["ema_base_start"] = pd.to_datetime(
    df_monitoring["ema_base_start"], dayfirst=True, errors="coerce", utc=True
)
df_monitoring["ema_base_end"] = pd.to_datetime(
    df_monitoring["ema_base_end"], dayfirst=True, errors="coerce", utc=True
)


# %% [markdown]
# ### 2.1 Merge relevant columns with passive data

# %%
df_monitoring_short = df_monitoring[["id", "for_id","study_version"]]

# %% [markdown]
# #### 2.2 Final `passive_data` data frame

# %%
df_backup_recent = df_backup_recent.merge(df_monitoring_short, on="id", how="right")

# %%

manual_for_map = {
    "asYVE": "FOR11011",
    "DWmH": "FOR11015",
    "dT0F": "FOR13008",
    "yv0Q": "FOR13012",
    "w0Ep": "FOR14051",
    "P0Tz": "FOR11156",
}

df_backup_recent["for_id"] = df_backup_recent["for_id"].fillna(df_backup_recent["id"].map(manual_for_map))

# %%
# ensure data types are coded correctly
df_backup_recent['boolean_value'] = df_backup_recent['boolean_value'].astype('boolean[pyarrow]')
df_backup_recent['study_version'] = df_backup_recent['study_version'].astype('string')
df_backup_recent['modality'] = df_backup_recent['modality'].astype('string')
df_backup_recent['id'] = df_backup_recent['id'].astype('category')

# %%
# Get a list of columns to drop (all columns not in keep_cols)
keep_cols_passive = ['id','for_id', 'modality', 'timestamp_start','timestamp_end',
    'local_timestamp_start', 'local_timestamp_end','time_interval', 'float_value', 'boolean_value','start_date', 
    'start_hour', "timezone_offset", 'study_version']

# final passive data frame
df_passive_final = df_backup_recent[keep_cols_passive]

# %% [markdown]
# ## 3. EMA data

# %% [markdown]
# #### 3.1 Load, match and rename relevant data from separate .csv files

# %%

# Beispiel: datapath = Path("/pfad/zum/verzeichnis")
session        = pd.read_csv(datapath / "questionnaireSession.csv", low_memory=False)
answers        = pd.read_csv(datapath / "answers.csv", low_memory=False)
choice         = pd.read_csv(datapath / "choice.csv", low_memory=False)
questions      = pd.read_csv(datapath / "questions.csv", low_memory=False)
questionnaire  = pd.read_csv(datapath / "questionnaires.csv", low_memory=False)  # ohne Komma!


# %%
# session data
study_mapping = {24: 0, 25: 0, 33: 1, 34: 1, 38: 2, 39: 2}
chronotype_mapping = {24: 0, 25: 1, 33: 0, 34: 1, 38: 0, 39: 1} 
session["user"] = session["user"].str[:4]
session.rename(columns = {"id":"session_id",
                          "user":"id",
                          "completedAt": "timestamp_beep_completion", 
                          "createdAt": "timestamp_item_completion", 
                          "expirationTimestamp": "timestamp_beep_expiration",
                          "sessionRun":"beep_num_run",
                          "study":"schedule_chronotype"}, inplace=True)
session['measurement_burst'] = session['schedule_chronotype'].map(study_mapping)
session['schedule_chronotype'] = session['schedule_chronotype'].map(chronotype_mapping)
# Parse epoch‑ms columns as UTC and drop tz info -> naive UTC
for col in ["timestamp_item_completion", "timestamp_beep_completion", "timestamp_beep_expiration"]:
    session[col] = (
        pd.to_datetime(session[col], unit="ms", utc=True, errors="coerce")
    )

df_sess = session[["id","session_id", "measurement_burst", "beep_num_run", "timestamp_item_completion", "timestamp_beep_completion", "schedule_chronotype", "timestamp_beep_expiration"]]

# %%
study_mapping = {24: 0, 25: 0, 33: 1, 34: 1, 38: 2, 39: 2}
chronotype_mapping = {24: 0, 25: 1, 33: 0, 34: 1, 38: 0, 39: 1} 

answers["user"] = answers["user"].str[:4]
answers = answers[["user", "questionnaire", "study", "question", "element",
                   "createdAt", "order", "questionnaireSession"]]

answers["createdAt"] = (
    pd.to_datetime(answers["createdAt"], unit="ms", utc=True, errors="coerce")
)
answers['measurement_burst'] = answers['study'].map(study_mapping)
answers['schedule_chronotype'] = answers['study'].map(chronotype_mapping)

answers.rename(columns={"user": "id", 
                        "createdAt": "timestamp_item_completion",
                        "questionnaire": "beep_type",
                        "question":"item_code_map",
                        "order":"item_order",
                        "questionnaireSession":"session_id",
                        }, inplace=True)
answers.drop(columns=["study"], inplace=True)

# %%
# item description data
choice = choice[["element", "choice_id", "text", "question"]]
choice.rename(columns={"text":"response_text",
                       "choice_id":"response", 
                       "question":"item_code_map"}, inplace=True)

# %%
# question description data
questions = questions[["id", "title"]]
questions.rename(columns={"id":"item_code_map",
                          "title":"item"}, inplace=True)

# %%
questionnaire = questionnaire[["id", "name"]]
questionnaire.rename(columns={"id":"beep_type",
                              "name":"beep_type_name"}, inplace=True)

# %%
answer_merged = pd.merge(answers, choice, on= ["item_code_map","element"])
answer_merged = pd.merge(answer_merged, questions, on= "item_code_map")
answer_merged = pd.merge(answer_merged, questionnaire, on= "beep_type")
answer_merged["date"] = answer_merged["timestamp_item_completion"].dt.normalize()

# %% [markdown]
# #### 3.2 Calculate auxiliary variables

# %%
df_monitoring_ema = df_monitoring[["id", "for_id","study_version", 'ema_base_start', 'ema_base_end',
       'ema_t20_start', 'ema_t20_end', 'ema_post_start', 'ema_post_end', 't20_post']]

# %%
bursts = [("base", 0), ("t20", 1), ("post", 2)]

out = []
for burst_name, burst_code in bursts:
    tmp = df_monitoring_ema[[
        "id", "for_id", "study_version", "t20_post",
        f"ema_{burst_name}_start", f"ema_{burst_name}_end"
    ]].copy()

    tmp = tmp.rename(columns={
        f"ema_{burst_name}_start": "ema_burst_start",
        f"ema_{burst_name}_end":   "ema_burst_end",
    })
    tmp["measurement_burst"] = burst_code
    out.append(tmp)

df_monitoring_ema_long = (
    pd.concat(out, ignore_index=True)
      # optional: drop rows where the burst is entirely missing
      .dropna(subset=["ema_burst_start", "ema_burst_end"], how="all")
      .sort_values(["id", "measurement_burst"])
      .reset_index(drop=True)
)


# %%
answer_merged = pd.merge(answer_merged, df_monitoring_ema_long, on = ["id", "measurement_burst"])

# %%
df_ema_content = answer_merged.copy()

# %%
meta_cols = ['id','for_id','date','response_text','item_code_map','beep_type' ,'beep_type_name',
              'element', 'item_order', 'session_id', 'measurement_burst']
df_ema_meta = df_ema_content[meta_cols].copy()

# %%
df_sess_short = df_sess[["id", "session_id",  "beep_num_run",'timestamp_beep_completion']].copy()

# %%
df_ema_meta = df_ema_meta.merge(df_sess_short, on=["id", "session_id"], how="left")

# %% [markdown]
# #### 3.3 Calculate EMA coverage

# %%
df_sess_short_compliance = df_sess[["id", "session_id", 'timestamp_beep_completion']].copy()

# %%
df_ema_content = df_ema_content.merge(df_sess_short_compliance, on=["id", "session_id"], how="left")

# %%
df_ema_content["n_beeps_completed_per_burst"] = (
    df_ema_content
    .groupby(["measurement_burst", "id"])["timestamp_beep_completion"]
    .transform("nunique"))

# %% [markdown]
# #### 3.3 Calculate auxiliary variables

# %%
#df_ema_content = answer_merged.copy()

# %%
# 1. Date and Time Manipulations
df_ema_content['weekday'] = df_ema_content['timestamp_item_completion'].dt.day_name()


# 1a. Season
def get_season(month):
    if month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:
        return 4

df_ema_content['season'] = df_ema_content['timestamp_item_completion'].dt.month.apply(get_season)

# 1b. Time of Day
def get_time_of_day(hour):
    if 5 <= hour < 8:
        return 1
    elif 8 <= hour < 12:
        return 2
    elif 12 <= hour < 17:
        return 3
    elif 17 <= hour < 21:
        return 4
    else:
        return 5

df_ema_content['time_of_day'] = df_ema_content['timestamp_item_completion'].dt.hour.apply(get_time_of_day)
df_ema_content['item'] = df_ema_content['item'].str.replace('_morning', '', regex=False)

# 3. Weekend Indicator
df_ema_content['weekend'] = df_ema_content['weekday'].isin(['Saturday', 'Sunday']).astype(int)

# 4. Questionnaire Number
df_ema_content['nr_beep_daily'] = df_ema_content['beep_type_name'].str.extract(r'(\d+)').astype(float)

# 5. Count unique questionnaires per day
df_ema_content['n_beeps_completed_per_day'] = (
    df_ema_content.groupby(['measurement_burst', 'id', 'date'])['beep_type_name']
                  .transform('nunique')
)

# 6. Unique Day Identifier
df_ema_content['quest_nr_str'] = df_ema_content['nr_beep_daily'].fillna('unknown').astype(str)
df_ema_content['beep_per_person_id'] = (
    df_ema_content['date'].dt.strftime('%Y%m%d') + '_' + df_ema_content['quest_nr_str']
)

# %%
# --- manual exception: ID mASG ---
# reassign misclassified burst 2 data to burst 1 

start_fix = pd.Timestamp("2025-08-21", tz="UTC")
end_fix   = pd.Timestamp("2025-09-05", tz="UTC")

mask_mASG_fix = (
    (df_ema_content["id"] == "mASG") &
    (df_ema_content["measurement_burst"] == 2) &
    (df_ema_content["date"].between(start_fix, end_fix))
)

df_ema_content.loc[mask_mASG_fix, "measurement_burst"] = 1


# %%
# 7. (ersetzt) Relative Start/End pro Phase & Customer
df_ema_content['ema_relative_start'] = (
    df_ema_content.groupby(['id', 'measurement_burst'])['date'].transform('min')
)
df_ema_content['ema_relative_end'] = (
    df_ema_content.groupby(['id', 'measurement_burst'])['date'].transform('max')
)

# 8. Absolute & Relative Day Index
df_ema_content['absolute_day_index'] = (
    df_ema_content['date'] - df_ema_content['ema_relative_start']
).dt.days + 1

df_ema_content['relative_day_index'] = (
    df_ema_content.groupby(['id', 'measurement_burst'])['date']
                  .rank(method='dense').astype(int)
)


# %%
# 9. Filter absolute_day_index > 16
max_allowed_days = 16
#df_ema_content = df_ema_content[df_ema_content['absolute_day_index'] <= max_allowed_days].reset_index(drop=True)

# 10. Check
high_indices = df_ema_content[df_ema_content['absolute_day_index'] > max_allowed_days]
high_indices_id = high_indices.id.unique().tolist()
if not high_indices.empty:
    print("Warning: High absolute_day_index vorhanden:", high_indices['id'].unique())
else:
    print("All entries have absolute_day_index <= 16.")



# %%
df_max_day = (
    df_ema_content.loc[
        df_ema_content["id"].isin(high_indices_id) & (df_ema_content["absolute_day_index"] > 16),
        ["id", "measurement_burst", "ema_relative_start", "ema_relative_end", "absolute_day_index", "beep_per_person_id"]
    ]
    .groupby(["id", "measurement_burst", "ema_relative_start", "ema_relative_end"], as_index=False)
    .agg(
        max_absolute_day_index=("absolute_day_index", "max"),
        n_unique_beeps_after_day16=("beep_per_person_id", "nunique"),
    )
    .sort_values(["id", "measurement_burst", "max_absolute_day_index"])
)

df_max_day_burst0 = df_max_day.loc[df_max_day["measurement_burst"] == 0]
df_max_day_burst1 = df_max_day.loc[df_max_day["measurement_burst"] == 1]
df_max_day_burst2 = df_max_day.loc[df_max_day["measurement_burst"] == 2]


# %% [markdown]
# CLUSTER PLOTTING TO DECIDE HOW TO PROCEED WITH MAX DAY INDEX > 16
# * spot days before/after the intended 16 day window
# * gaps or irregular participation
# * whether 'extra' days are meaningful or just noise

# %%
def plot_ema_clusters(
    df_ema_content,
    df_monitoring,
    df_ids_to_keep,
    burst_number,
    start_col,
    monitoring_id_col="id",
    ema_id_col="id",
    burst_col="measurement_burst",
    date_col="date"
):
    
    """
    Plot daily EMA beep counts per participant for a given measurement burst,
    with dashed lines for:
    - day 16 after first observed activation
    - scheduled EMA start date from df_monitoring
    Parameters
    ----------
    df_ema_content : pd.DataFrame
        EMA content dataframe.
    df_monitoring : pd.DataFrame
        Monitoring dataframe containing scheduled start dates.
    df_ids_to_keep : pd.DataFrame
        Dataframe containing IDs to retain (e.g. df_max_day_burst1).
    burst_number : int
        Measurement burst to plot (e.g. 1 or 2).
    start_col : str
        Column in df_monitoring with the scheduled EMA start date.
    monitoring_id_col : str
        ID column in df_monitoring before shortening.
    ema_id_col : str
        ID column in df_ema_content and df_ids_to_keep.
    burst_col : str
        Burst column in df_ema_content.
    date_col : str
        Date column in df_ema_content.
    """

    # --- Step 1: filter ema data ---
    df_ema = df_ema_content.copy()
    df_ema[date_col] = pd.to_datetime(df_ema[date_col], errors="coerce")
    df_ema[ema_id_col] = df_ema[ema_id_col].astype(str).str.strip()

    ids_to_keep = (
        df_ids_to_keep[ema_id_col]
        .astype(str)
        .str.strip()
        .drop_duplicates()
    )

    filtered = df_ema[
        (df_ema[burst_col] == burst_number) &
        (df_ema[ema_id_col].isin(ids_to_keep))
    ].copy()

    filtered = filtered.sort_values([ema_id_col, date_col])

    filtered["day_in_burst"] = (
        filtered.groupby(ema_id_col)[date_col]
        .transform(lambda x: (x - x.min()).dt.days + 1)
    )

   # --- Step 2: prepare scheduled starts --- 
    monitoring = df_monitoring.copy()
    monitoring[monitoring_id_col] = (
        monitoring[monitoring_id_col].astype(str).str.split("@").str.get(0).str[:4].str.strip()
    )
    monitoring[start_col] = pd.to_datetime(monitoring[start_col], errors="coerce")

    start_map = (
        monitoring[[monitoring_id_col, start_col]]
        .dropna(subset=[monitoring_id_col])
        .drop_duplicates(subset=[monitoring_id_col])
        .rename(columns={monitoring_id_col: ema_id_col})
    )

    # --- Step 3: merge onto filtered burst and id ---
    filtered = filtered.merge(start_map,on=ema_id_col,how="left")

    # --- Step 4: count beeps per day per id ---
    daily_counts = (
        filtered
        .groupby([ema_id_col, date_col, "day_in_burst", start_col], dropna=False)
        .size()
        .reset_index(name="n_beeps")
    )

    # --- Step 5: plot one figure per id ---
    for pid, sub in daily_counts.groupby(ema_id_col):
        fig = px.scatter(
            sub,
            x=date_col,
            y="n_beeps",
            title=f"id {pid} | burst {burst_number}",
            labels={
                date_col: "Date",
                "n_beeps": "Number of beeps"
            }
        )

        # black dashed line: 16th day after first observed activation
        cutoff_date = sub[date_col].min() + pd.Timedelta(days=15)
        fig.add_vline(
            x=cutoff_date,
            line_dash="dash"
        )

        # orange dashed line: scheduled EMA start
        ema_start = sub[start_col].iloc[0]
        if pd.notna(ema_start):
            fig.add_vline(
                x=ema_start,
                line_dash="dash",
                line_color="orange"
            )

        fig.show()

    return daily_counts, filtered


# %%
# CLUSTER PLOTTING T20
daily_counts_b1, filtered_b1 = plot_ema_clusters(
    df_ema_content=df_ema_content,
    df_monitoring=df_monitoring,
    df_ids_to_keep=df_max_day_burst1,
    burst_number=1,
    start_col="ema_t20_start"
)


# %%
# CLUSTER PLOTTING TPost
daily_counts_b2, filtered_b2 = plot_ema_clusters(
    df_ema_content=df_ema_content,
    df_monitoring=df_monitoring,
    df_ids_to_keep=df_max_day_burst2,
    burst_number=2,
    start_col="ema_post_start"
)

# %%
# remove beeps after day 16 for the measurement burst 0
df_ema_content = df_ema_content.loc[
    ~(
        (df_ema_content["measurement_burst"] == 0) &
        (df_ema_content["absolute_day_index"] > 16)
    )
].copy()

# %%
# remove out of phase data for measurement burst 1 and 2 (self-activation was possible, leading to false-activation of study phases)
# rule: keep all data from scheduled start (T20 and TPost) counting upwards to day 16, cut everything else
# exception 1: one case in which the last day with recorded ema beeps ends one day before scheduled start -> decided to keep (t20)
# exception 2: one case in which ema recording started a few days after offically scheduled start -> decided to keep (t20)
# exception 3: one case in which ema recording was self-activated one day before scheduled start -> decided to keep this day (tpost)

# keep raw data unchanged
df_ema_clean = df_ema_content.copy()

# flagged IDs only 
ids_burst1 = set(df_max_day_burst1["id"].astype(str).str.strip())
ids_burst2 = set(df_max_day_burst2["id"].astype(str).str.strip())

# prepare EMA data 
df_ema_clean["id"] = df_ema_clean["id"].astype(str).str.strip()
df_ema_clean["date"] = (pd.to_datetime(df_ema_clean["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize())

# prepare monitoring data
df_monitoring_clean = df_monitoring.copy()
df_monitoring_clean["id_short"] = (df_monitoring_clean["id"].astype(str).str.split("@").str[0].str[:4].str.strip())
df_monitoring_clean["ema_t20_start"] = (pd.to_datetime(df_monitoring_clean["ema_t20_start"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize())
df_monitoring_clean["ema_post_start"] = (pd.to_datetime(df_monitoring_clean["ema_post_start"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize())

# merge scheduled starts onto EMA data
df_ema_clean = df_ema_clean.merge(
    df_monitoring_clean[["id_short", "ema_t20_start", "ema_post_start"]]
    .drop_duplicates("id_short")
    .rename(columns={"id_short": "id"}),
    on="id",
    how="left"
)

# manual exceptions 
exception_id_burst1_before_start = "fx8B"
exception_id_burst1_keep_after_start = "p1zH"
exception_id_post = "p1zH"

# define keep rules
keep_burst1 = (
    # normal rule: T20 start until day 16
    df_ema_clean["date"].between(
        df_ema_clean["ema_t20_start"],
        df_ema_clean["ema_t20_start"] + pd.Timedelta(days=15)
    )
    |
    # exception fx8B: keep data before T20 start
    (
        (df_ema_clean["id"] == exception_id_burst1_before_start) &
        (df_ema_clean["date"] < df_ema_clean["ema_t20_start"])
    )
    |
    # exception p1zH: keep all T20/burst 1 data after scheduled T20 start
    (
        (df_ema_clean["id"] == exception_id_burst1_keep_after_start) &
        (df_ema_clean["date"] >= df_ema_clean["ema_t20_start"])
    )
)


# manual exception for burst 2 
keep_burst2 = (
    df_ema_clean["date"].between(
        df_ema_clean["ema_post_start"],
        df_ema_clean["ema_post_start"] + pd.Timedelta(days=15)
    )
    |
    # exception p1zH: keep exactly one day before scheduled TPost start
    (
        (df_ema_clean["id"] == exception_id_post) &
        (df_ema_clean["date"] == df_ema_clean["ema_post_start"] - pd.Timedelta(days=1))
    )
)


# explanation: keep the normal TPost window plus exactly one day before scheduled TPost start for p1zH

# --- apply cut only to flagged IDs ---
df_ema_clean = df_ema_clean.loc[
    ~(((df_ema_clean["measurement_burst"] == 1) & (df_ema_clean["id"].isin(ids_burst1)) & ~keep_burst1)
        |
        ((df_ema_clean["measurement_burst"] == 2) & (df_ema_clean["id"].isin(ids_burst2)) & ~keep_burst2)
    )
].copy()


# optional: drop scheduled start columns again
df_ema_clean = df_ema_clean.drop(columns=["ema_t20_start", "ema_post_start"])



# %%
# sanity check: compare rows and IDs before vs after cut
def summarize_cut_t20_post(before_df, after_df, burst_col="measurement_burst", id_col="id", bursts=(1, 2)):
    before = before_df.copy()
    after = after_df.copy()

    before[id_col] = before[id_col].astype(str).str.strip()
    after[id_col] = after[id_col].astype(str).str.strip()

    before = (before[before[burst_col].isin(bursts)]
        .groupby(burst_col)
        .agg(rows_before=(id_col, "size"),
            ids_before=(id_col, "nunique")))

    after = (after[after[burst_col].isin(bursts)]
        .groupby(burst_col)
        .agg(rows_after=(id_col, "size"),
            ids_after=(id_col, "nunique")))

    summary = pd.concat([before, after], axis=1).fillna(0).astype(int)
    summary["rows_removed"] = summary["rows_before"] - summary["rows_after"]
    summary["ids_removed"] = summary["ids_before"] - summary["ids_after"]

    return summary.reset_index()

summary = summarize_cut_t20_post(df_ema_content, df_ema_clean)
print(summary)

# %%
# please double check if the removal of T20 and TPost out of phase activation is running correctly (therefore run everything safely on df_ema_clean): 
# check if the table summary and clustering is consistent, if not adjust the code.

# if everything is working properly change 'df_ema_clean' to 'df_ema_content' so that the notebook continues with the cleaned df
df_ema_content = df_ema_clean.copy()

# %%
# manual correction
replacements = {
    ('mASG', 1): 81,
    ('9UHX', 1): 70,
    ('LQFF', 1): 96,
    ('uxmL', 1): 114,
    ('mASG', 2): 80
}

for (pid, burst), value in replacements.items():
    
    mask = (
        (df_ema_content['id'] == pid) &
        (df_ema_content['measurement_burst'] == burst)
    )
    
    df_ema_content.loc[
        mask,
        'n_beeps_completed_per_burst'
    ] = value


# %%
# 11. Questionnaire Counter
df_unique = df_ema_content.drop_duplicates(subset=['id', 'measurement_burst', 'beep_per_person_id']).copy()
df_unique['relative_beep_counter'] = df_unique.groupby(['id', 'measurement_burst']).cumcount() + 1
df_ema_content = df_ema_content.merge(
    df_unique[['id', 'measurement_burst', 'beep_per_person_id', 'relative_beep_counter']],
    on=['id', 'measurement_burst', 'beep_per_person_id'],
    how='left'
)

# 12. Missing Data behandeln
df_ema_content['measurement_burst'] = df_ema_content['measurement_burst'].fillna('unknown')
df_ema_content['absolute_day_index'] = df_ema_content['absolute_day_index'].where(
    df_ema_content['ema_relative_start'].notna(), np.nan
)

# %% [markdown]
# ### 3.4 merge the inferred tz offsets

# %%
# uncomment if want to run this cell multiple times
# if "infered_tzoffset" in df_sess.columns:
#     print("Dropping existing 'inferred_tzoffset' columns from df_sess")
#     df_sess = df_sess.drop(columns=["inferred_tzoffset", "inferred_tzoffset_source", "inferred_tzoffset_timedelta"])
# if "inferred_tzoffset" in df_ema_meta.columns:
#     print("Dropping existing 'inferred_tzoffset' columns from df_ema_meta")
#     df_ema_meta = df_ema_meta.drop(columns=["inferred_tzoffset", "inferred_tzoffset_source", "inferred_tzoffset_timedelta"])
# if "inferred_tzoffset" in df_ema_content.columns:
#     print("Dropping existing 'inferred_tzoffset' columns from df_ema_content")
#     df_ema_content = df_ema_content.drop(columns=["inferred_tzoffset", "inferred_tzoffset_source", "inferred_tzoffset_timedelta"])


df_ema_meta = merge_fill_tz(
    df_ema_meta, df_tz, day_col="date", customer_col="id"
)
# df_tz expects the date column to be timezone-aware (UTC), so we need to localize it before merging
df_ema_content["date"] = df_ema_content["date"].dt.tz_localize("utc")
df_ema_content = merge_fill_tz(
    df_ema_content, df_tz, day_col="date", customer_col="id"
)

# %%

df_ema_content.drop(columns=['response_text','item_code_map','beep_type' ,'beep_type_name',
              'element', 'item_order', 'session_id'], inplace=True) 

# %% [markdown]
# ### 3.5 Rename Affect Item names

# %% [markdown]
# If you want to use the old item names ('panas_') just comment out this section.

# %%
# Create a dictionary for affect mapping
affect_map = {
    "panas_attentiveness": "attentive",
    "panas_joviality1": "cheerful",
    "panas_joviality2": "happy",
    "panas_selfassurance": "self_confident",
    "panas_serenity1": "relaxed",
    "panas_serenity2": "calm",
    "panas_fear1": "anxious",
    "panas_fear2": "nervous",
    "panas_guilt1": "ashamed",
    "panas_guilt2": "dissatisfied_myself",
    "panas_hostility1": "irritable",
    "panas_hostility2": "angry",
    "panas_loneliness": "lonely",
    "panas_sadness1": "downcast",
    "panas_sadness2": "sad",
    "panas_shyness": "shy",
    "panas_fatigue": "fatigue"
}

# Replace the names in the 'item' column based on the directory
df_ema_content["item"] = df_ema_content["item"].replace(affect_map)

# %%
# sanity check 1:
df_ema_content['item'].unique()

# %%
# sanity check 2:
old_items = {
    "panas_selfassurance", "panas_joviality2", "panas_fatigue",
    "panas_joviality1", "panas_fear1", "panas_hostility2",
    "panas_serenity2", "panas_shyness", "panas_hostility1",
    "panas_guilt1", "panas_fear2", "panas_sadness1",
    "panas_guilt2", "panas_loneliness", "panas_serenity1",
    "panas_sadness2", "panas_attentiveness",
    "er_intensity", "er_control", "er_distraction",
    "er_reappraisal", "er_rumination", "er_relaxation",
    "er_suppression", "er_acceptance", "situation1",
    "situation2", "event_general", "event_social1",
    "ta_behavioral_2", "ta_kognitiv", "ta_kognitiv_2",
    "ta_behavioral", "physical_health", "ecg_control",
    "event_social2", "event_social3"
}

new_items = set(df_ema_content["item"].unique())

print(len(old_items))
print(len(new_items))

# %% [markdown]
# ### Export passive, EMA and Monitoring

# %%
backup_path = raw_path + "/backup_passive_recent.feather"
df_passive_final.to_feather(backup_path)

preprocessed_path_final = preprocessed_path + "/backup_passive_recent.feather"
df_passive_final.to_feather(preprocessed_path_final)

#preprocessed_path_freezed_final = preprocessed_path_freezed + "/code_check" + "/backup_passive_recent.parquet"
#df_passive_final.to_parquet(preprocessed_path_freezed_final)
ema_save_path = str(Path(preprocessed_path) / "ema")

with open(ema_save_path + '/ema_meta.pkl', 'wb') as file:
    pickle.dump(df_ema_meta, file)

    
with open(preprocessed_path + '/monitoring_data.pkl', 'wb') as file:
    pickle.dump(df_monitoring, file)

    
with open(ema_save_path + '/ema_content.pkl', 'wb') as file:
    pickle.dump(df_ema_content, file)

# %%

# Export ema meta as CSV
df_ema_path = ema_save_path + '/ema_meta.csv'
df_ema_meta.to_csv(df_ema_path, index=False)

# Export df_monitoring as CSV
df_monitoring_csv_path = preprocessed_path + '/monitoring_data.csv'
df_monitoring.to_csv(df_monitoring_csv_path, index=False)

# Export df_ema_content as CSV
df_ema_content_csv_path = ema_save_path + '/ema_content.csv'
df_ema_content.to_csv(df_ema_content_csv_path, index=False)

print("Exported df_ema_meta, df_monitoring, and df_ema_content as CSV files.")

# Export df_ema_content as CSV to freezed for data check
#df_ema_content_csv_path = preprocessed_path_freezed +'/code_check' +'/ema_content_recent.csv'
#df_ema_content.to_csv(df_ema_content_csv_path, index=False)



# %%
