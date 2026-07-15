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
# # Resting-state ECG: Import and Preprocessing

# %% [raw] vscode={"languageId": "raw"}
# last change: 2026-04-01 

# %% [markdown]
# ## ECG preprocessing pipeline for PREACT-digital  
#
# Details:
# * 2x/day, 30 sec during active EMA assessment (T0, T20, TPost), while each active assessment includes max. 16 days
#   
# * sampling rate: 300 Hz (i.e., 300 samples/1 sec => one sample every ~3.33 ms)
#   
# * 30 sec -> 300 x 30 = 9000 data points per ecg session
#
# * the raw data export provides values in V:
#     - 0.001 V ($10^{-3}$) = 1 mV
#     - 0.000001 V ($10^{-6}$) = 1 µV microvolt
#     - example: 0.000012 V ($10^{-5}$) = 0.012 mV = 12 µV
#
#
# Pipeline (wip):
#
# 1. **Import data:** Variation in data format   
#     - raw data as StringValues 'ECGVoltageOffset' with [t, v] from 14.03.2023 - 20.11.2023 (one string per ecg session -> data need to be extracted)  
#     - raw data as single '.csv' files [t, v] per ecg session from 17.11.2023 - today
#
# 2. **Concatenate:** separate 30s snippets per participant  -> this is not implemented so far
#
# 3. **Preprocessing & HRV computation:** NeuroKit  
#     - Clean signal (remove noise and artifacts) & bandpass filtering  
#     - R-Peak detection    
#     - RR Interval computation   
#     - HRV Metric (e.g. time-domain metric: RMSSD (ms)) based on 30 sec assessment
#
# 4. **Run Quality Control (QC):** generate a summary table (value range: min, max, mean, sd; sample size; range of missings in time series)  
#
# 5. **Export:** ecg features as `.parquet` file   
#        
#
# #### ECG Amplitudes not clear defined for wearables but can be lower compared to lab-assessments
#
# | ECG Component     | Range (mV) | 
# |------------------|-------------------|
# | Baseline noise   | 0.005 – 0.05 mV   | 
# | P wave           | 0.05 – 0.2 mV     | 
# | QRS complex      | 0.3 – 1.5 mV      | 
# | T wave           | 0.05 – 0.3 mV     | 
#
#
# #### Formula to convert V in mV (1 V are 1000 mV): 
#
# $$
# \text{V} \times 1000 = \text{mV}
# $$
#
# example:
#
# $$
# 0.000012\ \text{V} \times 1000 = 0.012\ \text{mV}
# $$

# %% [markdown]
# For data preprocessing and HRV computation the neurophysiological toolkit [Neurokit2](https://neuropsychology.github.io/NeuroKit/functions/ecg.html#) was used ([Makowski et al. 2021](https://link.springer.com/article/10.3758/s13428-020-01516-y)). See also [Pham et al. 2021](https://www.mdpi.com/1424-8220/21/12/3998) for a step-by-step guide for HRV analysis using Neurokit2.

# %% [markdown]
# -----------------------

# %% [markdown]
# ## TO-DOES (for Michal):
#
# Ideally work through the TO DO's in the proposed order:
#
# * ~~ecg data from November 2023 until today~~:
#     - ~~loading all .csv files from 17.11.2023 - until today takes hours if you just concatenat them. Instead the idea was to create batches. however, I created 3-moth batches with sub-batches and tried to concatenate these but this is also too slow because each single file contains 9000 rows and we have 676686 files with 9000 rows each (this is larger than the batches we concatenate in the data_load script with the very first passive data backup). Please find out if this is really the best way in terms of memory and adjust it if you find something more feasible and faster~~
#
#     - ~~some rows in the .csv files have more than 9000 data points (72000 means 8x 9000, 54000 means 6x 9000, 45000 means 5x 9000, 27000 means 3x 9000, 18000 means 2x 9000 -> after inspecting single files, it seems so that some data are stored more than once but it is not completely clear (needs more investigation). every single file should only contain 9000 data points!~~
#
#     - ~~the latest .csv file is from 30.09.2025 -> either this is a code bug OR an export bug from thrive - please double check (some 2026 folders seem to include only Nokia_300.csv files with timestamps from 2025 and 2024)~~
#
# * ~~Concatenate the cleaned raw data from March - Nov 2023 with the once from Nov23 - today~~
#
# * Double check and potentially adjust timezone if necessary (I didn't checked that to date)
#
# * ~~Run the preproc loop for all data and plot the violin plot to investigate data loss for RMSSD (HRV metric)~~
#
# * If it's high overall, go back into the preproc loop and investigate which parameters you could adjust and play around (ideally based on literature evidence so we can cite and argue why we chose which parameter in which size)
#
# * When you have a final version add the following columns to the data frame or make sure they are already included: ID, FOR-ID, timestamp, measurement_burst, ecg_session_nr (e.g. b1_d1_ecg_ses1, b1_d1_ecg_ses2), ema ecg control item - TODO check for the new data
#
# * ~~export as .parquet~~
#
#
# Further TO-DO's that can be tackled afterwards:
# * probably a few assessments will get lost by that because they are not directly followed by an active assessment (ema) but just activated randomly -> do we want to keep ecg sessions out of beep activation and flag them as 'ecg out of beep activation' vs 'triggered by beep' in a separate column?
# * participants can start the ecg more than once after the beep tells them to start ecg -> if more than one session is available for hrv computation for a beep how to decide which one to keep?
# * compute additional features? (NeuroKit2 enables to infer more)

# %% [markdown]
# ---

# %% vscode={"languageId": "python"}
from pyprojroot import here
import sys
sys.path.insert(0, str(here()))

from pathlib import Path
from collections import defaultdict
from datetime import date
import gc
import os
import re
import glob
import pickle
import json
import hashlib


import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
import time
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------------------------------
#  Define paths
# ------------------------------------

PROJECT_ROOT = here()

from server_config import proj_sheet, base_path
raw_path = base_path / "raw"
ecg_path = raw_path / "ecg"
backup_path = base_path / "backup"
preprocessed_path = base_path / "preprocessed"
ema_path = preprocessed_path / "ema_content.pkl"


# %% [markdown]
# ## 1. Import
#
# Variation in ECG data formats during data collection requires two separate data import steps indicated as 1.1 and 1.2 followed by harmonization into a single format.

# %% [markdown]
# ### 1.1 Import raw data as StringValues from 14.03.2023 - 20.11.2023  
#
# **Period:** 14 March 2023 until 20 November 2023  
# **Description:** ECG data are stored as stringValue   
# * (one string (= 1 row) contains 300 samples, i.e. all samples for 1 sec)   
# * located within the overall passive datase  
# * type 'RawECGVoltage'  
#
#
# path: `/sc-projects/sc-proj-cc15-preact/SP6/backup/first_backup/tiki_backup_2023-03-14_1.csv ...` [latest Update 2025-09-21]
#

# %% vscode={"languageId": "python"}
# define the pattern for passive data files
file_pattern = os.path.join(backup_path, "first_backup/tiki_backup_*.csv")

# use glob to find all matching files
file_list = glob.glob(file_pattern)

# sort the file list for consistent ordering
file_list.sort()

# %% vscode={"languageId": "python"}
# define the dtype for columns that are known to be problematic
dtype_spec = {
    "startTimestamp": "str",  # load as string initially
    "endTimestamp": "str",  # load as string initially
}

# create a list to hold all the dataframes
all_dfs = []

# loop over the files and read them with date parsing
for filename in file_list:
    df = pd.read_csv(
        filename, dtype=dtype_spec, low_memory=False
    )  # low-memory=False: read all files at once, then infer types (more reliable, avoids risk of inaccurate data types but uses more memory)

    # parse timestamps for THIS file
    for col in ["startTimestamp", "endTimestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], utc=True, format="%Y-%m-%dT%H:%M:%S%z", errors="coerce"
            )

    # derive date & index from THIS filename
    m = re.match(
        r"tiki_backup_(\d{4}-\d{2}-\d{2})_(\d+)\.csv", os.path.basename(filename)
    )
    if m:
        df["date"] = m.group(1)
        df["index"] = int(m.group(2))

    all_dfs.append(df)

# concatenate all dataframes
df_1 = pd.concat(all_dfs, ignore_index=True)

# optionally, drop the 'date' and 'index' columns if they are no longer needed
df_1.drop(columns=["date", "index"], inplace=True)


# %% vscode={"languageId": "python"}
df_1.head()

# %% vscode={"languageId": "python"}
# sanity check: all IDs included? CHECK
len(df_1["customer"].unique())

# %% vscode={"languageId": "python"}
# sanity check: entire period included? CHECK
min_starttimestamp = df_1["startTimestamp"].min()
max_endtimestamp = df_1["endTimestamp"].max()
print(min_starttimestamp)
print(max_endtimestamp)

# 2023-03-14 13:13:20+00:00
# 2023-11-21 08:05:00+00:00

# %% vscode={"languageId": "python"}
# quick sanity check
"stringValue" in df_1.columns

# %% vscode={"languageId": "python"}
# datetime is still UTC+01:00
df_1.dtypes

# %% vscode={"languageId": "python"}
# adjust datetime to UTC
for col in ["startTimestamp", "endTimestamp"]:
    df_1[col] = pd.to_datetime(df_1[col], utc=True, errors="coerce", unit="ms")

# extract date and hour part
df_1["start_day"] = df_1.startTimestamp.dt.normalize()
df_1["start_hour"] = df_1.startTimestamp.dt.hour

# extract customer identifier
df_1["customer"] = df_1.customer.str.split("@").str.get(0)
df_1["customer"] = df_1["customer"].str[:4]


# %% vscode={"languageId": "python"}
# rename columns
df_1.rename(
    columns={
        "customer": "id",
        "type": "modality",
        "startTimestamp": "timestamp_start",
        "endTimestamp": "timestamp_end",
    },
    inplace=True,
)

# only keep relevant variables
df_1 = df_1[
    [
        "id",
        "timestamp_start",
        "timestamp_end",
        # "timezoneOffset", # there's only berlin time specified, so probably useless
        "modality",
        "stringValue",
        "booleanValue",
        "doubleValue",
        "longValue",
    ]
]


# %% vscode={"languageId": "python"}
# filter ecg data
df_1_filtered = df_1[df_1["modality"] == "RawECGVoltage"].copy()

df_1_filtered.head(3)

# %% vscode={"languageId": "python"}
# very first ECG session is April 27 2023
df_1_filtered["timestamp_start"].min()

# %% vscode={"languageId": "python"}
# total unique IDs
print("All modalities:", df_1["id"].nunique())

# only ECG
print(
    "ECG:", df_1_filtered[df_1_filtered["modality"] == "RawECGVoltage"]["id"].nunique()
)

print("--------------------")

# unique IDs by each modality
# print(df_1.groupby("modality")["id"].nunique())


# %% [markdown]
# #### ECGVoltageOffset
#
# * resting-state ECG data are stored as stringValue for this time period
# * convert stringValue in two separate variables:
#   * `t` = relative time [sec] within recording window 
#   * `v` = microvoltage (electrical potential difference between two electrodes) -> changes in voltage over time (amplitude of the electrical signal at a given moment)
#  
# One string (= 1 row) contains 300 samples, i.e. all samples for 1 sec)

# %% vscode={"languageId": "python"}
# convert stringValues in two separte variables: t and v



def extract_t_v_lists(json_str):
    try:
        # replace decimal commas in numbers only (match digits with commas)
        fixed_str = re.sub(r"(\d),(\d)", r"\1.\2", json_str)
        parsed = json.loads(fixed_str)
        ecg_data = parsed.get("ECGVoltageOffset", [])
        t_list = [float(entry["t"]) for entry in ecg_data]
        v_list = [float(entry["v"]) for entry in ecg_data]
        return pd.Series([t_list, v_list])
    except Exception as e:
        print("Failed to parse row:", e)
        return pd.Series([None, None])


# apply to the DataFrame
df_1_filtered[["t", "v"]] = df_1_filtered["stringValue"].apply(extract_t_v_lists)

# sanity check: ensure lists are the same length in both columns
assert all(df_1_filtered["t"].str.len() == df_1_filtered["v"].str.len()), (
    "Mismatch in t and v list lengths"
)


# %% vscode={"languageId": "python"}
print(len(df_1_filtered))
df_1_filtered.head()

# %% [markdown]
# Edge cases (sessions differ from 9000 samples / 30 seconds):
# - `8lwi` 2023-07-11 16:19 - the last second is 240, not 300 Hz -> drop this ecg session
# - `nhdk` 2023-11-16 13:10 - the recording is a bit too long, drop the additional seconds (keep the first 30 seconds / 9000 samples)
# - `0NEG` 2023-10-09 22:18 - the first 20 seconds of samples is duplicated, drop the duplicated for this patient and day before concatenating
#

# %% vscode={"languageId": "python"}
# concatenate contiguous ECG chunks into one packed row per session (no explode)
# a new session starts when current timestamp_start != previous timestamp_end for the same id


def _concat_lists(series):
    out = []
    for arr in series:
        if isinstance(arr, list):
            out.extend(arr)
    return out


# keep the original simple concat workflow
_session_work = df_1_filtered.sort_values(
    ["id", "timestamp_start", "timestamp_end"]
).copy()

# edge-case fix 1 (0NEG, 2023-10-09): duplicated first seconds -> drop duplicate seconds before concat
mask_0neg_day = (_session_work["id"] == "0NEG") & (
    _session_work["timestamp_start"].dt.date == pd.Timestamp("2023-10-09").date()
)
_0neg_day = (
    _session_work.loc[mask_0neg_day]
    .sort_values(["timestamp_start", "timestamp_end"])
    .drop_duplicates(subset=["timestamp_start"], keep="first")
)
_session_work = pd.concat(
    [_session_work.loc[~mask_0neg_day], _0neg_day], ignore_index=True
).sort_values(["id", "timestamp_start", "timestamp_end"])

_session_work["prev_end"] = _session_work.groupby("id")["timestamp_end"].shift()
_session_work["new_session"] = _session_work["timestamp_start"].ne(
    _session_work["prev_end"]
)
_session_work["session_id"] = _session_work.groupby("id")["new_session"].cumsum()

df_1_sessions = _session_work.groupby(["id", "session_id"], as_index=False).agg(
    timestamp_start=("timestamp_start", "min"),
    timestamp_end=("timestamp_end", "max"),
    modality=("modality", "first"),
    t=("t", _concat_lists),
    v=("v", _concat_lists),
    n_chunks=("t", "size"),
)

# edge-case fix 2 (8lwi, 2023-07-11 16:19): last second has 240 samples -> drop session
mask_8lwi_bad = (df_1_sessions["id"] == "8lwi") & (
    df_1_sessions["timestamp_start"] == pd.Timestamp("2023-07-11 16:19:56+00:00")
)
df_1_sessions = df_1_sessions.loc[~mask_8lwi_bad].copy()

# edge-case fix 3 (nhdk, 2023-11-16 13:10): recording too long -> keep first 30 seconds only
mask_nhdk_long = (df_1_sessions["id"] == "nhdk") & (
    df_1_sessions["timestamp_start"] == pd.Timestamp("2023-11-16 13:10:58+00:00")
)
df_1_sessions.loc[mask_nhdk_long, "t"] = df_1_sessions.loc[mask_nhdk_long, "t"].apply(
    lambda arr: arr[:9000] if isinstance(arr, list) else arr
)
df_1_sessions.loc[mask_nhdk_long, "v"] = df_1_sessions.loc[mask_nhdk_long, "v"].apply(
    lambda arr: arr[:9000] if isinstance(arr, list) else arr
)
df_1_sessions.loc[mask_nhdk_long, "n_chunks"] = 30
df_1_sessions.loc[mask_nhdk_long, "timestamp_end"] = df_1_sessions.loc[
    mask_nhdk_long, "timestamp_start"
] + pd.Timedelta(seconds=30)

df_1_sessions["duration_s"] = (
    df_1_sessions["timestamp_end"] - df_1_sessions["timestamp_start"]
).dt.total_seconds()
df_1_sessions["n_samples"] = df_1_sessions["t"].str.len()

# sanity check: t and v lists must stay aligned
assert (df_1_sessions["t"].str.len() == df_1_sessions["v"].str.len()).all()

# list -> numpy array
df_1_sessions["t"] = df_1_sessions["t"].apply(
    lambda lst: np.array(lst, dtype=np.float32)
)
df_1_sessions["v"] = df_1_sessions["v"].apply(
    lambda lst: np.array(lst, dtype=np.float32)
)

# all of the t arrays are identical
ecg_t = df_1_sessions["t"].iloc[0]
fs = 300
assert np.allclose(ecg_t, np.arange(0, 30, 1 / fs))  # t can be easily reconstructed
assert df_1_sessions["t"].apply(lambda arr: np.array_equal(arr, ecg_t)).all()
assert (
    df_1_sessions["n_chunks"].eq(30).all()
)  # all sessions should now be concatenated into one chunk
assert (
    df_1_sessions["duration_s"].eq(30).all()
)  # all sessions should now be 30 seconds long
assert (
    df_1_sessions["n_samples"].eq(9000).all()
)  # all sessions should now have 9000 samples (30s * 300Hz)

# remove these columns as they are no longer needed and just take up memory
df_1_sessions = df_1_sessions.drop(columns=["n_chunks", "duration_s", "n_samples", "t"])

df_1_sessions

# %% vscode={"languageId": "python"}
# renamed df_1_ecg to df_1_exploded
# take each list in the specified columns (ecg_relative_t, ecg_v) and expand them so that each item in the list becomes its own row
# df_1_exploded = df_1_filtered.explode(["t", "v"]).reset_index(drop=True)
df_1_sessions["t"] = [np.arange(0, 30, 1 / fs)] * len(
    df_1_sessions
)  # add t back as a column of identical arrays for explode
df_1_exploded = df_1_sessions.explode(["t", "v"]).reset_index(drop=True)
df_1_sessions.drop(
    columns=["t"], inplace=True
)  # drop t again as it's no longer needed after explode

# sanity check: 300 samples per second, i.e. 301 should be the next sec in timestamp_start -> YES
# -> changed that, so the 9001 sample is the next session (different timestamp_start) -> YES
df_1_exploded.iloc[8995:9005]

# %% vscode={"languageId": "python"}
# sanity check (all IDs included)
df_1_exploded["id"].nunique()

# %% vscode={"languageId": "python"}
# general plausibility check before preprocessing (values in V)
print("Max:", df_1_exploded["v"].max())
print("Min:", df_1_exploded["v"].min())

# typical QRS complex (= one heartbeat) for a scanwatch is: 0.3 - 1.5 mV
# Max/Min values are quite high/low (possible noise/artefacts)

# %% vscode={"languageId": "python"}
# general plausibility check: row comparison before stringValue extraction vs. after
print("df_1_filtered shape:", df_1_filtered.shape)
print("df_1_exploded shape:", df_1_exploded.shape)

# %% vscode={"languageId": "python"}
# remove unnecessary columns
# -> already not included
# df_1_exploded = df_1_exploded.drop(columns=["stringValue", "booleanValue","doubleValue","longValue"])
df_1_exploded

# %% vscode={"languageId": "python"}
# final data frame
# -> use df_1_sessions, as it's easier to work with
# df_1_exploded.head()
print(len(df_1_sessions))
df_1_sessions.head()

# %% [markdown]
# ### 1.2 Import raw data as single .csv files [t, v] from 17.11.2023 -- 30.09.2025

# %% [markdown]
# > **Period:** 17 November 2023 until TODAY   
# > **Description:** ECG data are stored in `ID_yyyy-mm-dd_hh:mm:ss_Nokia_300.csv`files including a t and a v column (one .csv file per ecg session) 
# > 
# > <u>regex pattern:</u>
# > 
# > * **Group 1:** Customer ID -> `0AlyaEtHo8K3Cyo8`
# > * **Group 2:** Date -> `2024-04-08`
# > * **Group 3:** Time -> `12:04:46`
# > 
# > path: `/sc-projects/sc-proj-cc15-preact/SP6/raw/tiki_backup_files/export_tiki_daymonthyear ...` 

# %% [markdown]
# - the same files in multiple exports are read multiple times
#     if we only analyze a subset, the it skews the analysis

# %% vscode={"languageId": "python"}
# build file_list efficiently: keep one file per sample ID from the latest export
export_root = Path(raw_path) / "tiki_backup_files"
export_dirs = [p for p in export_root.glob("export_tiki_*") if p.is_dir()]

if not export_dirs:
    raise FileNotFoundError(f"No export_tiki_* folders found in {export_root}")

# parse export date from folder name in ddmmyyyy format
date_pattern = re.compile(r"(?:export_tiki_)?(\d{2})(\d{2})(\d{4})$")


def parse_export_date(folder_name):
    m = date_pattern.search(folder_name)
    if not m:
        return pd.NaT
    day, month, year = m.groups()
    return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")


# newest export first; NaT values are pushed to the end
export_dirs_sorted = sorted(
    export_dirs,
    key=lambda d: (pd.notna(parse_export_date(d.name)), parse_export_date(d.name)),
    reverse=True,
)

# first time we see a sample ID wins (because we iterate newest -> oldest)
latest_file_by_sample = {}
scanned_files = 0
skipped_hidden = 0

for export_dir in export_dirs_sorted:
    for entry in os.scandir(export_dir):
        if not entry.is_file() or not entry.name.endswith("_Nokia_300.csv"):
            continue
        if entry.name.startswith("._"):
            skipped_hidden += 1
            continue

        scanned_files += 1
        sample_id = entry.name[:-4]  # drop .csv
        if sample_id not in latest_file_by_sample:
            latest_file_by_sample[sample_id] = entry.path

# final file list: unique sample IDs, each from latest available export for that sample
file_list = sorted(latest_file_by_sample.values())

# sanity checks
print(len(export_dirs_sorted), "export folders scanned")
print(scanned_files, "Nokia_300.csv files scanned (excluding hidden macOS files)")
print(skipped_hidden, "hidden macOS files skipped")
print(len(file_list), "unique sample IDs kept (latest export per sample)")
print("Newest export folder (by parsed date):", export_dirs_sorted[0].name)

# %% [markdown]
# so the problem here is that we reread the same sample multiple times (~60)

# %% vscode={"languageId": "python"}
sample_ids = [f.split("/")[-1] for f in file_list]
len(set(sample_ids))


# %% vscode={"languageId": "python"}
# check whether the newest export folder contains all sample IDs seen across exports
import pandas as pd

# reuse results from the import cell when available
if "export_dirs_sorted" not in globals() or "latest_file_by_sample" not in globals():
    export_root = Path(raw_path) / "tiki_backup_files"
    export_dirs = [p for p in export_root.glob("export_tiki_*") if p.is_dir()]

    if not export_dirs:
        raise FileNotFoundError(f"No export_tiki_* folders found in {export_root}")

    date_pattern = re.compile(r"(?:export_tiki_)?(\d{2})(\d{2})(\d{4})$")

    def parse_export_date(folder_name):
        m = date_pattern.search(folder_name)
        if not m:
            return pd.NaT
        day, month, year = m.groups()
        return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")

    export_dirs_sorted = sorted(
        export_dirs,
        key=lambda d: (pd.notna(parse_export_date(d.name)), parse_export_date(d.name)),
        reverse=True,
    )

    latest_file_by_sample = {}
    for export_dir in export_dirs_sorted:
        for entry in os.scandir(export_dir):
            if not entry.is_file() or not entry.name.endswith("_Nokia_300.csv"):
                continue
            sample_id = entry.name[:-4]
            if sample_id not in latest_file_by_sample:
                latest_file_by_sample[sample_id] = entry.path

latest_export_dir = export_dirs_sorted[0]
all_sample_ids = set(latest_file_by_sample.keys())

latest_sample_ids = {
    entry.name[:-4]
    for entry in os.scandir(latest_export_dir)
    if entry.is_file() and entry.name.endswith("_Nokia_300.csv")
}

missing_in_latest = sorted(all_sample_ids - latest_sample_ids)
present_in_latest = len(all_sample_ids) - len(missing_in_latest)

print("Latest export folder:", latest_export_dir)
print("Total unique sample IDs across all exports:", len(all_sample_ids))
print("Unique sample IDs in latest export:", len(latest_sample_ids))
print("Included in latest export:", present_in_latest)
print("Missing from latest export:", len(missing_in_latest))

if missing_in_latest:
    print("\nFirst 20 missing sample filenames:")
    print(pd.Series(missing_in_latest[:20]).to_string(index=False))
else:
    print("\nAll sample IDs are included in the latest export.")

# %% vscode={"languageId": "python"}
# sanity check: latest recorded ecg session (latest timestamp is 2025-09-30)
# where are the data between October 2025 and April 2026? again a change in data format by thrive?
# At least some 2026 folder seem to include no Nokia_300.csv files with a 2026 timestamp

from datetime import datetime


def extract_ts_fast(filename):
    parts = Path(filename).stem.split("_")
    return datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y-%m-%d_%H:%M:%S")


latest_file = max(file_list, key=extract_ts_fast)

print("Latest file:", latest_file)
print("Timestamp:", extract_ts_fast(latest_file))

# %% vscode={"languageId": "python"}
# build dataframe from file_list with: long_id, timestamp, export_date, filepath

if "file_list" not in globals() or len(file_list) == 0:
    raise RuntimeError(
        "file_list is missing or empty. Run the section 1.2 import cell first."
    )

# long_id can itself contain underscores -> parse from the right using date/time suffix
filename_pattern = re.compile(
    r"^(?P<long_id>.+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}:\d{2}:\d{2})_Nokia_300\.csv$"
)
# supports folders such as export_tiki_09042024, export_tiki_09042024_v2, etc.
export_pattern = re.compile(
    r"export_tiki_(?P<day>\d{2})(?P<month>\d{2})(?P<year>\d{4})"
)

rows = []
failed = []

for fp in file_list:
    p = Path(fp)

    # extract long_id + timestamp from file name
    m_file = filename_pattern.match(p.name)
    if not m_file:
        failed.append((fp, "Filename pattern mismatch"))
        continue

    long_id = m_file.group("long_id")
    timestamp = pd.to_datetime(
        f"{m_file.group('date')}_{m_file.group('time')}",
        format="%Y-%m-%d_%H:%M:%S",
        errors="coerce",
    )

    # extract export date from any path segment containing export_tiki_ddmmyyyy
    m_export = None
    for part in p.parts:
        m = export_pattern.search(part)
        if m:
            m_export = m
            break

    if m_export:
        export_date = pd.to_datetime(
            f"{m_export.group('year')}-{m_export.group('month')}-{m_export.group('day')}",
            format="%Y-%m-%d",
            errors="coerce",
        )
    else:
        export_date = pd.NaT
        failed.append((fp, "Export folder pattern mismatch"))

    rows.append(
        {
            "long_id": long_id,
            "timestamp": timestamp,
            "export_date": export_date,
            "filepath": str(p),
        }
    )

df_file_list = pd.DataFrame(
    rows, columns=["long_id", "timestamp", "export_date", "filepath"]
)
df_file_list = df_file_list.sort_values(["timestamp", "long_id"]).reset_index(drop=True)

print(f"Rows parsed: {len(df_file_list)}")
print(f"Rows with parsing warnings: {len(failed)}")

df_file_list.head()

# %% [markdown]
# ------------

# %% vscode={"languageId": "python"}
df_file_list.head()

# %% vscode={"languageId": "python"}
len(file_list)

# %% [markdown]
# #### Import files
#
# >Issue: loading all .csv files from 17.11.2023 - until today takes hours if you just concatenat them. Instead the idea was to create 3 month batches (due to the fact that each single file contains 9000 rows). however, these batches are also very slow because we have 676686 files with 9000 rows.
# >* Please find out if this is really the best way in terms of memory and adjust it if you find something more feasible and faster
# >
# >Procedure so far:
# >* split data into 3-month batches
# >* split each 3-month batch (e.g. quarter) into smaller chunks (sub-batches) to reduce too many files in memory at once
#
# - multiple sessions in a file resolved below
# - there are also not that many .csv files (*look above*), as they are contained in each week's export
# so it's not that slow and i made it faster with paralellization (make sure to secure multiple cpus)

# %% vscode={"languageId": "python"}
# build df_2_concat from file_list: one row per ECG session/file with parsed t and v arrays
if "file_list" not in globals() or len(file_list) == 0:
    raise RuntimeError(
        "file_list is missing or empty. Run the section 1.2 import cell first."
    )

# long_id can contain underscores, so parse from the right
filename_pattern = re.compile(
    r"^(?P<long_id>.+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}:\d{2}:\d{2})_Nokia_300\.csv$"
)
export_pattern = re.compile(
    r"export_tiki_(?P<day>\d{2})(?P<month>\d{2})(?P<year>\d{4})"
)

# set an integer for quick testing (e.g., 500); keep None to process all files
max_files = None
input_files = sorted(file_list)
if max_files is not None:
    input_files = input_files[:max_files]

rows = []
failed_files = []

for i, fp in enumerate(input_files, start=1):
    p = Path(fp)

    try:
        # parse metadata from filename
        m_file = filename_pattern.match(p.name)
        if not m_file:
            raise ValueError(f"Filename pattern mismatch: {p.name}")

        long_id = m_file.group("long_id")
        timestamp = pd.to_datetime(
            f"{m_file.group('date')}_{m_file.group('time')}",
            format="%Y-%m-%d_%H:%M:%S",
            errors="coerce",
        )

        # parse export date from folder name
        m_export = None
        for part in p.parts:
            m = export_pattern.search(part)
            if m:
                m_export = m
                break

        if m_export:
            export_date = pd.to_datetime(
                f"{m_export.group('year')}-{m_export.group('month')}-{m_export.group('day')}",
                format="%Y-%m-%d",
                errors="coerce",
            )
        else:
            export_date = pd.NaT

        # read and parse signal columns only
        df_tmp = pd.read_csv(
            fp,
            usecols=lambda c: c in {"t", "v"},
            dtype={"t": "float32", "v": "float32"},
            low_memory=False,
        )

        if not {"t", "v"}.issubset(df_tmp.columns):
            raise ValueError("Missing required columns t and/or v")

        t_arr = df_tmp["t"].to_numpy(dtype=np.float32, copy=True)
        v_arr = df_tmp["v"].to_numpy(dtype=np.float32, copy=True)

        rows.append(
            {
                "id": long_id[:4],  # extract first 4 chars for id
                "timestamp_start": timestamp,
                "export_date": export_date,
                "filepath": str(p),
                "t": t_arr,
                "v": v_arr,
                "n_samples": int(len(v_arr)),
            }
        )

    except Exception as e:
        failed_files.append((fp, str(e)))

    if i % 5000 == 0:
        print(f"Processed {i}/{len(input_files)} files")
        gc.collect()

# one row per file/session
df_2_concat = pd.DataFrame(
    rows,
    columns=["id", "timestamp_start", "export_date", "filepath", "t", "v", "n_samples"],
)

if not df_2_concat.empty:
    df_2_concat = df_2_concat.sort_values(["timestamp_start", "id"]).reset_index(
        drop=True
    )

print(f"Rows in df_2_concat: {len(df_2_concat)}")
print(f"Failed files: {len(failed_files)}")

df_2_concat.head()

# %% vscode={"languageId": "python"}
df_2_concat["n_samples"].value_counts()

# %% [markdown]
# ----------

# %% vscode={"languageId": "python"}
# expand df_2_concat to one row per ECG session inside each file
# split points are inferred from t resets (non-increasing t)
if "df_2_concat" not in globals() or df_2_concat.empty:
    raise RuntimeError(
        "df_2_concat is missing or empty. Run the df_2_concat build cell first."
    )

required_cols = {
    "id",
    "timestamp_start",
    "export_date",
    "filepath",
    "t",
    "v",
    "n_samples",
}
missing_cols = required_cols.difference(df_2_concat.columns)
if missing_cols:
    raise ValueError(f"df_2_concat is missing required columns: {sorted(missing_cols)}")


def split_file_into_sessions(row):
    t = np.asarray(row["t"], dtype=np.float32)
    v = np.asarray(row["v"], dtype=np.float32)

    if t.size != v.size:
        raise ValueError(f"t/v length mismatch for file: {row['filepath']}")
    if t.size == 0:
        return []

    # session boundary when t restarts (or is not strictly increasing)
    reset_idx = np.where(np.diff(t) <= 0)[0] + 1
    starts = np.concatenate(([0], reset_idx))
    ends = np.concatenate((reset_idx, [t.size]))

    # counter rule:
    # - files with <=9000 points -> counter starts at 0
    # - files with >9000 points  -> counter starts at 1
    counter_start = 0 if t.size <= 9000 else 1

    out = []
    for i_session, (s, e) in enumerate(zip(starts, ends)):
        t_seg = t[s:e]
        v_seg = v[s:e]
        if t_seg.size == 0:
            continue

        # keep only first 30s (=9000 points) for long sessions
        if t_seg.size > 9000:
            t_seg = t_seg[:9000]
            v_seg = v_seg[:9000]

        # each split session should start at t=0
        t_seg = t_seg - t_seg[0]

        out.append(
            {
                "id": row["id"],
                "timestamp_start": row["timestamp_start"],
                "export_date": row["export_date"],
                "filepath": row["filepath"],
                "session_counter": int(counter_start + i_session),
                "t": t_seg,
                "v": v_seg,
                "n_samples": int(v_seg.size),
            }
        )
    return out


session_rows = []
failed_splits = []

for i, row in df_2_concat.iterrows():
    try:
        session_rows.extend(split_file_into_sessions(row))
    except Exception as e:
        failed_splits.append((row.get("filepath", f"row_{i}"), str(e)))

    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1}/{len(df_2_concat)} file-rows")


df_2_sessions_before_dedup = pd.DataFrame(
    session_rows,
    columns=[
        "id",
        "timestamp_start",
        "export_date",
        "filepath",
        "session_counter",
        "t",
        "v",
        "n_samples",
    ],
)

if not df_2_sessions_before_dedup.empty:
    df_2_sessions_before_dedup = df_2_sessions_before_dedup.sort_values(
        ["id", "timestamp_start", "session_counter"]
    ).reset_index(drop=True)

# final cleanup: drop incomplete sessions (< 9000 samples)
short_sessions_dropped = int((df_2_sessions_before_dedup["n_samples"] < 9000).sum())
df_2_sessions_before_dedup = df_2_sessions_before_dedup.loc[
    df_2_sessions_before_dedup["n_samples"] >= 9000
].reset_index(drop=True)

files_with_multiple_sessions = (
    df_2_sessions_before_dedup.groupby("filepath")["session_counter"].max().gt(0).sum()
    if not df_2_sessions_before_dedup.empty
    else 0
)

assert df_2_sessions_before_dedup["n_samples"].eq(9000).all(), (
    "Not all sessions have 9000 samples after cleanup"
)
assert (
    df_2_sessions_before_dedup["t"].apply(lambda arr: np.array_equal(arr, ecg_t)).all()
), "Not all t arrays are identical after cleanup"

df_2_sessions_before_dedup.drop(
    columns=["n_samples", "t"], inplace=True
)  # drop n_samples as it's no longer needed after cleanup

print(f"Rows in df_2_concat (file-level): {len(df_2_concat)}")
print(
    f"Rows in df_2_sessions_before_dedup (session-level): {len(df_2_sessions_before_dedup)}"
)
print(f"Dropped short sessions (<9000 samples): {short_sessions_dropped}")
print(f"Files with >1 session: {files_with_multiple_sessions}")
print(f"Failed file-rows during split: {len(failed_splits)}")

df_2_sessions_before_dedup.head()


# %% [markdown]
# use hash(v) to deduplicate sessions

# %% vscode={"languageId": "python"}
def hash_v_array(v):
    arr = np.asarray(v, dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


if "df_2_sessions_before_dedup" not in globals() or df_2_sessions_before_dedup.empty:
    raise RuntimeError(
        "df_2_sessions_before_dedup is missing or empty. Run the session-expansion cell first."
    )




df_2_sessions_before_dedup["v_hash"] = df_2_sessions_before_dedup["v"].apply(
    hash_v_array
)

print(f"Rows hashed: {len(df_2_sessions_before_dedup)}")
print(f"Unique hashes: {df_2_sessions_before_dedup['v_hash'].nunique()}")

df_2_sessions_before_dedup.head()

# %% vscode={"languageId": "python"}
# check cross-id hash collisions and deduplicate by v_hash
# keep earliest timestamp_start when duplicates/collisions occur
if "df_2_sessions_before_dedup" not in globals() or df_2_sessions_before_dedup.empty:
    raise RuntimeError(
        "df_2_sessions_before_dedup is missing or empty. Run the session-expansion cell first."
    )

if "v_hash" not in df_2_sessions_before_dedup.columns:
    raise RuntimeError("v_hash is missing. Run the hash cell first.")

# collision audit before dedup
pre_hash_stats = df_2_sessions_before_dedup.groupby("v_hash", as_index=False).agg(
    n_rows=("id", "size"), n_ids=("id", "nunique")
)
pre_cross_id = pre_hash_stats[pre_hash_stats["n_ids"] > 1].copy()

# deduplicate by hash keeping earliest timestamp
# tie-breakers ensure deterministic behavior
df_2_sessions = (
    df_2_sessions_before_dedup.sort_values(
        ["timestamp_start", "filepath", "session_counter"]
    )
    .drop_duplicates(subset=["v_hash"], keep="first")
    .reset_index(drop=True)
)

# collision audit after dedup
post_hash_stats = df_2_sessions.groupby("v_hash", as_index=False).agg(
    n_rows=("id", "size"), n_ids=("id", "nunique")
)
post_cross_id = post_hash_stats[post_hash_stats["n_ids"] > 1].copy()

print("Before dedup")
print("Rows:", len(df_2_sessions_before_dedup))
print("Unique hashes:", pre_hash_stats.shape[0])
print("Cross-ID collision hashes:", pre_cross_id.shape[0])

print("\nAfter dedup")
print("Rows:", len(df_2_sessions))
print("Dropped rows:", len(df_2_sessions_before_dedup) - len(df_2_sessions))
print("Unique hashes:", post_hash_stats.shape[0])
print("Cross-ID collision hashes:", post_cross_id.shape[0])

# df_2_sessions[["id", "timestamp_start", "session_counter", "n_samples", "v_hash"]].head()
df_2_sessions.head()

# %% vscode={"languageId": "python"}
# quick check of session_counter rule against ORIGINAL file size from df_2_concat
# <=9000 in original file -> only counter 0
# >9000 in original file -> counters start at 1
orig_size = df_2_concat[["filepath", "n_samples"]].rename(
    columns={"n_samples": "orig_n_samples"}
)

counter_check = (
    df_2_sessions_before_dedup.groupby("filepath", as_index=False)
    .agg(
        min_counter=("session_counter", "min"),
        max_counter=("session_counter", "max"),
        n_sessions=("session_counter", "size"),
    )
    .merge(orig_size, on="filepath", how="left")
)

violations_small = counter_check[
    (counter_check["orig_n_samples"] <= 9000) & (counter_check["min_counter"] != 0)
]
violations_large = counter_check[
    (counter_check["orig_n_samples"] > 9000) & (counter_check["min_counter"] != 1)
]

print("Small-file rule violations:", len(violations_small))
print("Large-file rule violations:", len(violations_large))

violations_large

# %% vscode={"languageId": "python"}
df_2_sessions.head()

# %% [markdown]
# #### Inspecting data coverage after 30/09/2025: plot sanity checks

# %% vscode={"languageId": "python"}
# import ema data to extract information about active assessment phases
ema_content = pd.read_pickle(ema_path)

# %% vscode={"languageId": "python"}
# sanity check plots: inspecting data coverage to give feedback to thrive

import pandas as pd
import plotly.express as px

# --- Step 1: PREPARATION --- 
milestones = [("2025-09-30", "30 Sep 2025")]

df_2_sessions = df_2_sessions.copy()
#df_2_sessions["timestamp_start"] = pd.to_datetime(df_2_sessions["timestamp_start"])
#df_2_sessions["date"] = df_2_sessions["timestamp_start"].dt.date

# active phases
ema_schedule = ema_content.copy()

#ema_schedule["ema_relative_start"] = pd.to_datetime(ema_schedule["ema_relative_start"], errors="coerce")
#ema_schedule["ema_relative_end"] = pd.to_datetime(ema_schedule["ema_relative_end"], errors="coerce")

ema_schedule = (ema_schedule[["id", "measurement_burst", "ema_relative_start", "ema_relative_end"]]
                .dropna(subset=["id", "measurement_burst", "ema_relative_start", "ema_relative_end"])
                .drop_duplicates(subset=["id", "measurement_burst"])
                )

# date time conversion
df_2_sessions["timestamp_start"] = pd.to_datetime(df_2_sessions["timestamp_start"], errors="coerce", utc=True).dt.tz_localize(None)

df_2_sessions["date"] = df_2_sessions["timestamp_start"].dt.date

ema_schedule["ema_relative_start"] = pd.to_datetime(ema_schedule["ema_relative_start"], errors="coerce", utc=True).dt.tz_localize(None)

ema_schedule["ema_relative_end"] = pd.to_datetime(ema_schedule["ema_relative_end"], errors="coerce", utc=True).dt.tz_localize(None)

# merge ecg sessions with all three burst windows per participant
df_ecg_bursts = df_2_sessions.merge(ema_schedule, on="id", how="left")

# filter ecg rows that fall into the correct active burst window
df_ecg_active = df_ecg_bursts[
    (df_ecg_bursts["timestamp_start"] >= df_ecg_bursts["ema_relative_start"]) &
    (df_ecg_bursts["timestamp_start"] <= df_ecg_bursts["ema_relative_end"])
].copy()

# create a relative day within each burst
df_ecg_active["burst_day"] = (
    df_ecg_active["timestamp_start"].dt.normalize()
    - df_ecg_active["ema_relative_start"].dt.normalize()
).dt.days + 1


# --- Step 2: PLOTTING --- 

# Plot 1: total daily ECG coverage
daily = (
    df_2_sessions.groupby(["date"])
      .size()
      .reset_index(name="ecg_count")
)

fig1 = px.line(
    daily,
    x="date",
    y="ecg_count",
    #color="id",
    markers=True,
    title="Daily ECG data coverage",
    labels={"date": "Date", "ecg_count": "Number of ECG records"}
)

for date, label in milestones:
    date = pd.Timestamp(date)

    # vertical dashed line
    fig1.add_vline(
        x=date,
        line_dash="dash",
        line_color="black",
        line_width=2,
    )

    # vertical label ("flag") at the top
    fig1.add_annotation(
        x=date,
        y=1.02,
        xref="x",
        yref="paper",      
        text=label,
        showarrow=False,
        textangle=-90,
        font=dict(size=10, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=2,
    )

fig1.show()



# Plot 2: daily ECG coverage per participant
# Plot 2: daily ECG coverage per participant

daily_by_id = (
    df_2_sessions
    .groupby(["id", "date"])
    .size()
    .reset_index(name="ecg_count")
)

n_ids = daily_by_id["id"].nunique()

fig2 = px.line(
    daily_by_id,
    x="date",
    y="ecg_count",
    color="id",
    markers=True,
    title=f"Daily ECG data coverage by participant (n = {n_ids})",
    labels={
        "date": "Date",
        "ecg_count": "Number of ECG records",
        "id": "Participant"
    }
)

# horizontal dashed reference lines with labels on the left
for y in [1, 2, 3]:
    fig2.add_hline(
        y=y,
        line_dash="dot",
        line_color="gray",
        line_width=1,
        annotation_text=f"{y} session",
        annotation_position="left",
        annotation_font_size=7,   # smaller text
    )

# vertical milestone line + label
for date, label in milestones:
    date = pd.Timestamp(date).date()

    fig2.add_vline(
        x=date,
        line_dash="dash",
        line_color="black",
        line_width=2,
    )

    fig2.add_annotation(
        x=date,
        y=1.02,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        textangle=-90,
        font=dict(size=10, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=2,
    )

fig2.show()




# %% [markdown]
# #### DELETE LATER [START]

# %% vscode={"languageId": "python"}
# ECG sessions per participant per burst
ecg_sessions_per_burst = (
    df_ecg_active
    .groupby(["id", "measurement_burst"])
    .size()
    .unstack(fill_value=0)
    .rename(columns={
        0: "Burst 0",
        1: "Burst 1",
        2: "Burst 2"
    })
)

# Total across bursts
ecg_sessions_per_burst["Total"] = ecg_sessions_per_burst.sum(axis=1)

display(ecg_sessions_per_burst)

# %% vscode={"languageId": "python"}
# Keep only one row per participant per day per burst
participant_day = (
    df_ecg_active
    .drop_duplicates(subset=["id", "measurement_burst", "burst_day"])
)

# Count participants with >=1 ECG session
coverage = (
    participant_day
    .groupby(["measurement_burst", "burst_day"])
    .size()
    .reset_index(name="n_participants")
)

coverage.head()

# %% vscode={"languageId": "python"}
fig = px.line(
    coverage,
    x="burst_day",
    y="n_participants",
    color="measurement_burst",
    markers=True,
    title="Participants with ≥1 ECG session per assessment day",
    labels={
        "burst_day": "Day within burst",
        "n_participants": "Participants with ≥1 ECG session",
        "measurement_burst": "Measurement burst"
    }
)

fig.show()

# %% [markdown]
# DELETE LATER [KEEP GOING, END IS FURTHER]

# %% vscode={"languageId": "python"}
# Plot the number of ECG sessions and active EMA phases per day
import os
import numpy as np
import pandas as pd
import plotly.express as px

# 1. Load the metadata containing the EMA start and end dates
meta_path = "/sc-projects/sc-proj-cc15-preact/SP6/preprocessed/meta/SP6_meta_data.csv"
if os.path.exists(meta_path):
    df_meta = pd.read_csv(meta_path)
    
    # Parse all EMA start/end date columns
    date_cols = [
        "ema_base_start", "ema_base_end",
        "ema_t20_start", "ema_t20_end",
        "ema_post_start", "ema_post_end"
    ]
    for col in date_cols:
        df_meta[col] = pd.to_datetime(df_meta[col], errors="coerce", dayfirst=True, utc=True).dt.tz_localize(None).dt.normalize()
    
    # 2. Get date range from df_2_sessions
    df_2_sessions_dates = pd.to_datetime(df_2_sessions["date"]).dt.normalize()
    min_date = df_2_sessions_dates.min()
    max_date = df_2_sessions_dates.max()
    
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        # Calculate daily ECG sessions count
        ecg_counts = df_2_sessions_dates.value_counts().reindex(date_range, fill_value=0)
        
        # Calculate number of active EMA participants/phases per day
        active_ema_counts = []
        for d in date_range:
            is_active = (
                ((df_meta["ema_base_start"] <= d) & (d <= df_meta["ema_base_end"])) |
                ((df_meta["ema_t20_start"] <= d) & (d <= df_meta["ema_t20_end"])) |
                ((df_meta["ema_post_start"] <= d) & (d <= df_meta["ema_post_end"]))
            )
            active_ema_counts.append(is_active.sum())
            
        # Combine into a single DataFrame for plotting
        df_plot = pd.DataFrame({
            "Date": date_range,
            "ECG Sessions": ecg_counts.values,
            "Active EMA Participants": active_ema_counts
        })
        
        # Melt to plot both metrics on the same graph
        df_plot_melted = df_plot.melt(
            id_vars=["Date"],
            value_vars=["ECG Sessions", "Active EMA Participants"],
            var_name="Metric",
            value_name="Count"
        )
        
        fig_compare = px.line(
            df_plot_melted,
            x="Date",
            y="Count",
            color="Metric",
            title="Daily ECG Sessions and Active EMA Participants",
            labels={"Date": "Date", "Count": "Count"}
        )
        
        # Add milestone vertical line
        fig_compare.add_vline(
            x=pd.Timestamp("2025-09-30"),
            line_dash="dash",
            line_color="black",
            line_width=2,
        )
        fig_compare.add_annotation(
            x=pd.Timestamp("2025-09-30"),
            y=1.02,
            xref="x",
            yref="paper",
            text="30 Sep 2025",
            showarrow=False,
            textangle=-90,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=2,
        )
        
        fig_compare.show()

        # 3. Calculate ratio (ECG sessions / active participants)
        df_plot["Sessions per Active Participant"] = (
            df_plot["ECG Sessions"] / df_plot["Active EMA Participants"]
        ).replace([np.inf, -np.inf], np.nan)
        
        fig_ratio = px.line(
            df_plot,
            x="Date",
            y="Sessions per Active Participant",
            title="Daily ECG Sessions / Active EMA Participants Ratio",
            labels={"Date": "Date", "Sessions per Active Participant": "Ratio (ECG Sessions / Active Participants)"}
        )
        
        # Add milestone vertical line
        fig_ratio.add_vline(
            x=pd.Timestamp("2025-09-30"),
            line_dash="dash",
            line_color="black",
            line_width=2,
        )
        fig_ratio.add_annotation(
            x=pd.Timestamp("2025-09-30"),
            y=1.02,
            xref="x",
            yref="paper",
            text="30 Sep 2025",
            showarrow=False,
            textangle=-90,
            font=dict(size=10, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=2,
        )
        
        fig_ratio.show()

        # 4. Save those figs to pyprojroot / tmp folder
        tmp_dir = here() / "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            fig_compare.write_html(str(tmp_dir / "ecg_sessions_vs_active_ema.html"))
            fig_compare.write_image(str(tmp_dir / "ecg_sessions_vs_active_ema.png"), scale=2)
        except Exception as e:
            print(f"Error saving comparison plot: {e}")
            
        try:
            fig_ratio.write_html(str(tmp_dir / "ecg_sessions_ratio.html"))
            fig_ratio.write_image(str(tmp_dir / "ecg_sessions_ratio.png"), scale=2)
        except Exception as e:
            print(f"Error saving ratio plot: {e}")
    else:
        print("Warning: ECG session dates are empty or invalid.")
else:
    print(f"Warning: Metadata file not found at {meta_path}")

# %% [markdown]
# #### DELETE LATER [END]

# %% [markdown]
# ## 2. Concatenate data from format 1.1 and 1.2

# %% [markdown]
# `string_session_id` is a number of session per ID, only for data from stringValue
#
# `csv_session_counter` is a variable used to split files with multiple sessions into separate sessions
# session_counter = 0 means there's only one session in the file,
# session_counter > 0, it means there are multiple sessions in the file and the counter indicates the session number (starting from 1) for that file
# as there's a deduplication step, not all sessions counters are retained,
# for example there might be sessions with counters 2-7, cause the counter 1 was removed
#
# `csv_filepath` is only deefined for csv data (from which file the session was extracted)

# %% vscode={"languageId": "python"}
df_1_sessions.rename(columns={"session_id": "string_session_id"}, inplace=True)
df_1_sessions

# %% vscode={"languageId": "python"}
df_2_sessions.rename(columns={"session_counter": "csv_session_counter", "filepath": "csv_filepath"}, inplace=True)
df_2_sessions

# %% vscode={"languageId": "python"}
df_2_sessions
# TODO timezones in df_2, assume utc for now
df_2_sessions["timestamp_start"] = df_2_sessions["timestamp_start"].dt.tz_localize(
    "UTC"
)

# %% vscode={"languageId": "python"}
df_all_ecg = (
    pd.concat([df_1_sessions, df_2_sessions], ignore_index=True)
    .sort_values(["id", "timestamp_start"])
    .reset_index(drop=True)
)
print("Total ECG sessions:", len(df_all_ecg))  # weirdly round number - 19k
df_all_ecg.head()

# %% [markdown]
# ### check if there are duplications from 1 to 2 

# %% vscode={"languageId": "python"}
# DONE check if there are duplications from 1 to 2

# DONE map ECGs to the unique beep identifier
# beep_per_person_id (from ema_content)
# and to measurement_burst (from ema_content as well)
# might not be a matching id
# match them based on the timestamp_start & id

# %% vscode={"languageId": "python"}
df_all_ecg["v_hash"] = df_all_ecg["v"].apply(hash_v_array)
assert df_all_ecg["v_hash"].nunique() == len(df_all_ecg)
print("All ECG sessions have unique v arrays based on hash check.")


# %% vscode={"languageId": "python"}
df_all_ecg.drop(columns=["v_hash", "modality"], inplace=True) 

# %% [markdown]
# ### map ecgs to unique beep identifier
# beep_per_person_id (from ema_content)\
# and to measurement_burst (from ema_content as well)\
# might not be a matching id\
# match them based on the timestamp_start & id

# %% vscode={"languageId": "python"}
# df_ema_content = pd.read_parquet(preprocessed_path / "ema_content.pkl")
with open(preprocessed_path / "ema_content.pkl", "rb") as f:
    df_ema_content = pickle.load(f)

print(f"EMA content rows: {len(df_ema_content):_}")
df_ema_content.head()

# %% vscode={"languageId": "python"}
print(len(df_all_ecg))
df_all_ecg.head()

# %% vscode={"languageId": "python"}
df_ema_content.head()

# %% vscode={"languageId": "python"}
# Prepare EMA data for merge_asof
# Select only necessary columns
ema_subset = df_ema_content[
    ["id", "timestamp_item_completion", "timestamp_beep_completion", "beep_per_person_id", "measurement_burst"]
].copy()

# Ensure id columns are the same type (string) for proper grouping
ema_subset["id"] = ema_subset["id"].astype(str)
df_all_ecg["id"] = df_all_ecg["id"].astype(str)

# Sort exclusively by the timestamp keys as strictly required by df_merge_asof.
# When using by='id', merge_asof splits by 'id' internally but expects the overall dataframe to be sorted by the join key.
df_all_ecg = df_all_ecg.sort_values("timestamp_start").reset_index(drop=True)
ema_subset = ema_subset.sort_values("timestamp_item_completion").reset_index(drop=True)

print(f"EMA subset shape: {ema_subset.shape}")
print(f"Original ECG DataFrame shape: {df_all_ecg.shape}")

# Perform merge_asof to find the closest EMA assessment for each ECG
# direction="nearest" finds the closest timestamp for each ECG within each subject
df_all_ecg = pd.merge_asof(
    df_all_ecg,
    ema_subset,
    left_on="timestamp_start",
    right_on="timestamp_item_completion",
    by="id",  # Match within the same subject
    direction="nearest",
    tolerance=pd.Timedelta("2h"),
)
# TODO decide on a reasonable tolerance -> 2h

# Calculate time difference in seconds
df_all_ecg["time_diff_to_ema_seconds"] = (
    (
        df_all_ecg["timestamp_start"] - df_all_ecg["timestamp_item_completion"]
    ).dt.total_seconds()
).abs()

# Verification
print(f"\nResult DataFrame shape: {df_all_ecg.shape}")
assert len(df_all_ecg) == len(df_all_ecg), "Row count mismatch!"

print(
    f"\nRows with EMA match: {df_all_ecg['beep_per_person_id'].notna().sum()} / {len(df_all_ecg)}"
)
print(f"Rows without EMA match: {df_all_ecg['beep_per_person_id'].isna().sum()}")
print(f"\nTime difference statistics (seconds):")
print(df_all_ecg["time_diff_to_ema_seconds"].describe())

# %% vscode={"languageId": "python"}
# Plot histogram and mark the peak
vals = df_all_ecg["time_diff_to_ema_seconds"].dropna().values
bins = np.arange(0, 300, 1)
counts, bin_edges = np.histogram(vals, bins=bins)
peak_idx = np.argmax(counts)
peak_bin_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

plt.figure(figsize=(8, 4))
plt.hist(vals, bins=bins, color="C0", alpha=0.7)
plt.axvline(
    peak_bin_center,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"Peak: {peak_bin_center:.0f}s",
)
plt.xlabel("Absolute Time Difference to EMA (seconds)")
plt.title(
    "Distribution of Time Differences Between ECG Sessions and Closest EMA Assessments"
)
plt.legend()
peak_height = counts[peak_idx]
plt.annotate(
    f"{int(peak_bin_center)}s",
    xy=(peak_bin_center, peak_height),
    xytext=(peak_bin_center + 8, peak_height * 0.9),
    arrowprops=dict(arrowstyle="->", color="red"),
)
plt.tight_layout()

# %% vscode={"languageId": "python"}
df_monitoring = pd.read_csv(
    "https://docs.google.com/spreadsheets/d/1z8LZJBBMzzAmiXIS47X8SLk-zSMwDIXSKPit4IlmfuE/export?format=csv"
)
df_monitoring.rename(
    columns={
        "Pseudonym": "id",
        "FOR_ID": "for_id",
        "EMA_ID": "ema_id",
        "Status": "study_status",
        "Studienversion": "study_version",
        "Start EMA Baseline": "ema_base_start",
        "Ende EMA Baseline": "ema_base_end",
        "Freischaltung/ Start EMA T20": "ema_t20_start",
        "Ende EMA T20": "ema_t20_end",
        "Freischaltung/ Start EMA Post": "ema_post_start",
        "Ende EMA Post": "ema_post_end",
        "T20=Post": "t20_post",
    },
    inplace=True,
)
monitoring_subset = df_monitoring[["id", "for_id", "ema_id"]].copy()
monitoring_subset["id"] = monitoring_subset["id"].str[:4]

# %% vscode={"languageId": "python"}
monitoring_subset["id"].value_counts()

# %% vscode={"languageId": "python"}
# TODO decide about hte OmAV duplication
monitoring_subset
df_all_ecg = df_all_ecg.merge(
    monitoring_subset,
    on="id",
    how="left",
    # validate="many_to_one", # OmAV appears twice
)

# %% vscode={"languageId": "python"}
df_all_ecg.head()

# %% vscode={"languageId": "python"}
# save to parquet
df_all_ecg.to_parquet(raw_path / "df_all_ecg_start--20250930.parquet", index=False)

# %% [markdown]
# ## 3. PREPROCESSING & HRV computation
#
# Use 2500 .csv files as starting point to write the preprocessing pipeline (adjust this part as soon you have all ecg sessions concatenated in one large data frame): loop over all files and preprocess every session separately

# %% [markdown]
# PREPOC LIBRARY: NeuroKit2
#
# * There are several different established libraries for ecg preprocessing. after trying out different libraries I decided to stick on [NeuroKit2](https://neuropsychology.github.io/NeuroKit/functions/ecg.html) 
#
# * Reason: 
#     - In comparison to general signal processing libraries such as [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.datasets.electrocardiogram.html) NeuroKit2 is specifically designed for physiological signals like ECG or PPG. It incorporates built-in knowledge of ECG morphology (e.g., P waves, QRS complexes, and T waves). As a result, many functions already include domain-specific processing steps, so users do not need to implement everything from scratch. Furthermore, NeuroKit2 is well established in the literature, making it a reliable choice for standardized ECG preprocessing.
#
#
#    - If, however, you want to develop custom signal processing pipelines or implement algorithms from first principles, SciPy is more suitable. This approach requires a deeper understanding of signal processing methods and how different algorithms affect the data
#
# * Memo: I removed all visual plots to keep the preproc as clean and short as possible. we can re-add them later after fixing the import section

# %% vscode={"languageId": "python"}
# Parallel execution

# for testing, use only the first 2500 ECG sessions
# later try out parallel processing with e.g. joblib to apply preprocessing to the whole data set



fs = 300  # sampling frequency in Hz
method = "neurokit"  # R-peak detection method


def process_single_ecg(index, ecg_arr, fs=300, method="neurokit"):
    """Process one ECG session and return one result row."""
    start = time.perf_counter()

    try:
        ecg_arr = np.asarray(ecg_arr, dtype=np.float32)
        assert ecg_arr.ndim == 1, "ECG signal must be 1-dimensional"
        assert ecg_arr.size == 9000, f"Expected 9000 samples, got {ecg_arr.size}"

        # convert ECG from volts (V) to millivolts (mV)
        ecg_mv = ecg_arr * 1000.0

        # clip extreme values to reduce noise/artifacts
        ecg_clipped = np.clip(ecg_mv, -2, 2)

        # preprocess signal + detect R-peaks
        _, info = nk.ecg_process(ecg_clipped, sampling_rate=fs, method=method)
        rpeaks = info["ECG_R_Peaks"]

        # alternatively
        # ecg_cleaned = nk.ecg_clean(ecg_clipped, sampling_rate=fs, method="neurokit")
        # _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs, correct_artifacts=True)
        # rpeaks = info["ECG_R_Peaks"]

        if len(rpeaks) < 3:
            raise ValueError("Too few R-peaks for RR/HRV computation")

        rr_ms = np.diff(rpeaks) / fs * 1000
        if len(rr_ms) == 0:
            raise ValueError("Empty RR intervals")

        # HRV time-domain feature (plots disabled)
        hrv = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
        hrv_rmssd = hrv["HRV_RMSSD"].iloc[0]

        row = {
            "index": index,
            "Number of Rows": int(ecg_arr.size),
            "low (mV)": float(ecg_mv.min()),
            "max (mV)": float(ecg_mv.max()),
            "RR min": float(rr_ms.min()),
            "RR max": float(rr_ms.max()),
            "RR mean": float(rr_ms.mean()),
            "HRV_RMSSD": float(hrv_rmssd),
            "error": None,
        }
    except Exception as e:
        # Keep failed sessions in output so the pipeline is robust and auditable.
        row = {
            "index": index,
            "Number of Rows": np.nan,
            "low (mV)": np.nan,
            "max (mV)": np.nan,
            "RR min": np.nan,
            "RR max": np.nan,
            "RR mean": np.nan,
            "HRV_RMSSD": np.nan,
            "error": str(e),
        }

    row["runtime_s"] = time.perf_counter() - start
    return row


if "df_all_ecg" not in globals() or df_all_ecg.empty:
    raise RuntimeError(
        "df_all_ecg is missing or empty. Run the concatenation step first."
    )

# No race conditions: workers do not mutate shared dataframes; each returns one dict.
results = Parallel(n_jobs=8, backend="loky")(
    delayed(process_single_ecg)(
        i,
        np.asarray(v, dtype=np.float32),
        fs=fs,
        method=method,
    )
    for i, v in tqdm(
        # df_all_ecg[:1000][["v"]].itertuples(index=True, name=None),
        df_all_ecg[["v"]].itertuples(index=True, name=None),
        total=len(df_all_ecg),
        desc="Processing ECG sessions",
    )
)

df = pd.DataFrame(results)
df.set_index("index", inplace=True)
times = df["runtime_s"].dropna().tolist()

if times:
    print("Avg runtime:", sum(times) / len(times))
    print("Max runtime:", max(times))
print("Failed sessions:", df["error"].notna().sum())
df.head()

# %% vscode={"languageId": "python"}
# Merge preprocessing output into full ECG session table using existing row index.
if "df" not in globals() or df.empty:
    raise RuntimeError("df is missing or empty. Run preprocessing cell first.")

df_all_ecg_processed = df_all_ecg.join(df, how="left")

df_all_ecg_processed.head()

# %% vscode={"languageId": "python"}
# if there should be more ecg files, change to the ecg folder, otherwise keep the current
# df_all_ecg_processed.to_parquet(preprocessed_path / "ecg" / "ecg_processed.parquet", index=False)
df_all_ecg_processed.to_parquet(preprocessed_path / "ecg_processed.parquet", index=False)

# %% vscode={"languageId": "python"}
df_all_ecg_processed[df_all_ecg_processed["error"].notna()]

# %% [markdown]
# Explore if RMSSD (time-domain HRV metric) is in a plausible range
# * In general, what is considered low or high depends heavily on the individual's baseline (higher: better, lower: worse)
# * but a very broad orientation can be this:
#
# | RMSSD (ms)     | Interpretation | 
# |------------------|-------------------|
# | < 20 ms    | low (stress, fatigue, illness) |  
# | 20 - 40 ms          | below average     | 
# | 40 - 80 ms     | normal / healthy      | 
# | 80 - 120 ms         | high (good recovery)     | 
# | > 120 ms         | very high (often athletes)     | 
#

# %% vscode={"languageId": "python"}
import plotly.express as px

rmssd = df["HRV_RMSSD"].dropna()

# apply both lower and upper bounds
rmssd_clean = rmssd[(rmssd >= 5) & (rmssd <= 200)]

# count excluded values (outside this range)
excluded_count = ((rmssd < 5) | (rmssd > 200)).sum()
total_count = len(rmssd)
excluded_pct = excluded_count / total_count * 100

excluded_pct = excluded_count / total_count * 100

# filtering (cut RMSSD values >= 200)
rmssd_clean = rmssd[rmssd < 200]
excluded_count = (rmssd >= 200).sum()
total_count = len(rmssd)

# violin plot (horizontal)
fig = px.violin(
    x=rmssd_clean,
    box=True,
    points="all",
    title=f"RMSSD Violin Plot (5 - 200 ms) | n sessions ={len(rmssd)}  | Excluded={excluded_count}/{total_count} ({excluded_pct:.1f}%)",
)

fig.update_traces(
    orientation="h",  # horizontal
    meanline_visible=True,
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="HRV_RMSSD (ms)",
    yaxis_title="",
    width=800,
)

fig.show()

# %% [markdown]
# a typical clean data set should look like
# * peak around 30 - 60 ms
# * right-skewed tail
# * very few values > 150 ms
#
#
# * even if it strongly depends on age, health status, and preprocessing conditions in general it's arguable to define
#   - too low: RMSSD < 5 - 10 ms
#   - too large: RMSSD > 150 - 200 ms
#
# Thus: 5 - 200 ms as acceptable range is quite a liberal choice with still 20.5% data loss in 2500 ecg sessions

# %% vscode={"languageId": "python"}
low_outliers = (rmssd < 5).sum()
high_outliers = (rmssd > 200).sum()

print(f"Low (<5): {low_outliers}")
print(f"High (>=200): {high_outliers}")

# %% vscode={"languageId": "python"}
# show all metrics (we can infer more from NeuroKit but I reduced it for know)
df.head(50)

# still an issue: some rows in the .csv files have more than 9000 data points
# 72000 means 8x 9000, 54000 means 6x 9000, 45000 means 5x 9000, 27000 means 3x 9000, 18000 means 2x 9000
# after inspecting single files, it seems so that some data are stored more than once but it is not completely clear (needs more investigation),
# every single file should only contain 9000 data points
df

# %% vscode={"languageId": "python"}
# note:
# removed all single case visualizations of preprocessing steps (raw, clean, peak detection etc.) to make the notebook as clean as possible for know
# (can add them later again if you want)



# %% vscode={"languageId": "python"}
