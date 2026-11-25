# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: py39-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Aggregate GPS data
#
# Pipeline:
# 1. Load passive data, build one row per GPS fix (Latitude/Longitude).
# 2. Coverage metrics per participant-day (on raw timestamps; coordinates not needed).
# 3. Undo the app's coordinate anonymisation (additive lat/lon offset, latitude wraps at +-90)
#    per key period -> consistent geometry for all participants.
# 4. Clustering (weighted DBSCAN) + home detection per participant-key unit.
# 5. Daily metrics, merged back to one row per participant-day.

# %%
from pyprojroot import here
import sys
sys.path.insert(0, str(here()))

import os
import gc
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
from server_config import base_path

import argparse
from src.preprocessing import gps_features

# %% [markdown]
# ## Parameters

# %%
# Reference point for re-centring the anonymised coordinates (centre of the study region).
# Only latitude matters for the distance scale; use ~ (51.2, 10.4) if participants come from all over Germany.
REF_LAT, REF_LON = 52.52, 13.40

# Restrict to each participant's first year of GPS data
MAX_DAYS_PER_PARTICIPANT = 365

# %% [markdown]
# ## Load Data

# %%
base_path = Path(base_path)
backup_path = base_path / "raw" / "backup_passive_recent.feather"
df_backup = pd.read_feather(backup_path)
print(f"Loaded backup data shape: {df_backup.shape}")

# %%
parser = argparse.ArgumentParser(description="Aggregate GPS passive sensor data.")
parser.add_argument("--debug", action="store_true", help="Enable debug mode (subset participants).")
args, _ = parser.parse_known_args()

debug = args.debug or os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON!!!!!!!!!!!!!")
    # Take a small subset of participants for fast debugging
    selected_participants = ["4MLe", "5nNG"]
    df_backup = df_backup[df_backup["id"].isin(selected_participants)].copy()
    print(f"Selected participants for debugging: {selected_participants}")
    print(f"Data shape after filtering for debug: {df_backup.shape}")
else:
    # DBSCAN memory issue is fixed (grid de-duplication), so no participants need to be excluded.
    print("Processing all participants")

# %% [markdown]
# ## GPS

# %%
df_gps_raw = df_backup[df_backup["modality"].isin(["Latitude", "Longitude"])].copy()

lat = (
    df_gps_raw.loc[
        df_gps_raw["modality"].eq("Latitude"),
        ["id", "timestamp_start", "local_timestamp_start", "float_value"],
    ]
    .rename(columns={"float_value": "Latitude"})
    .copy()
)
lon = (
    df_gps_raw.loc[
        df_gps_raw["modality"].eq("Longitude"),
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

# free memory: the full passive backup is no longer needed
del df_backup, df_gps_raw, lat, lon
gc.collect()

# %%
# Restrict to each participant's first year of GPS data
first_obs_per_customer = df_loc_all.groupby("id", observed=True)[
    "local_timestamp_start"
].transform("min")
df_loc = df_loc_all[
    df_loc_all["local_timestamp_start"]
    < (first_obs_per_customer + pd.Timedelta(days=MAX_DAYS_PER_PARTICIPANT))
].copy()

del df_loc_all, first_obs_per_customer
gc.collect()

df_loc = df_loc.sort_values(by=["id", "timestamp_start"])
print(df_loc.shape)

# %% [markdown]
# ### Coverage metrics per participant-day

# %%
gps_raw_points = (
    df_loc.groupby(["id", "local_day"], observed=True)
    .size()
    .reset_index(name="GPS_raw_points")
)

gps_hour_counts = (
    df_loc.groupby(["id", "local_day", "local_hour"], observed=True)
    .size()
    .reset_index(name="points_in_hour")
)

gps_hourly_summary = (
    gps_hour_counts.groupby(["id", "local_day"], observed=True)
    .agg(
        GPS_hours_with_data=("local_hour", "nunique"),
        GPS_points_per_hour_mean=("points_in_hour", "mean"),
        GPS_points_per_hour_median=("points_in_hour", "median"),
        GPS_points_per_hour_std=("points_in_hour", "std"),
    )
    .reset_index()
)

# gap statistics per participant-day (vectorized)
_gap = df_loc[["id", "local_day", "timestamp_start"]].copy()
_gap["gap_s"] = (
    _gap.groupby(["id", "local_day"], observed=True)["timestamp_start"].diff().dt.total_seconds()
)
gps_gap_stats = (
    _gap.groupby(["id", "local_day"], observed=True)["gap_s"]
    .agg(GPS_max_gap_seconds="max", GPS_mean_gap_seconds="mean")
    .reset_index()
)
del _gap

df_gps_coverage = gps_raw_points.merge(
    gps_hourly_summary, on=["id", "local_day"], how="outer"
).merge(gps_gap_stats, on=["id", "local_day"], how="outer")

df_gps_coverage["GPS_coverage_pct"] = (
    df_gps_coverage["GPS_hours_with_data"] / 24.0 * 100.0
)

logging.info("GPS coverage metrics calculated and merged into df_gps_coverage")
print(df_gps_coverage.shape)
df_gps_coverage.head()

# %% [markdown]
# ### Undo coordinate anonymisation
#
# The app adds a random offset to latitude (wrapping at +-90) and longitude per key period.
# A latitude offset distorts east-west distances by cos(lat_obfuscated), so we split each
# participant into key units and re-centre each unit on (REF_LAT, REF_LON).
# After this, ids are '<id>_k<key>' and 'id_orig' holds the original id.

# %%
df_loc = gps_features.normalize_shifted_coords(df_loc, ref_lat=REF_LAT, ref_lon=REF_LON)

# remove GPS glitches: points reached AND left faster than an airliner (e.g. single garbage fixes)
df_loc = gps_features.remove_impossible_excursions(df_loc, max_speed_kmh=1000)

n_units, n_people = df_loc["id"].nunique(), df_loc["id_orig"].nunique()
print(f"{n_people} participants in {n_units} key units")

# %% [markdown]
# ### Clustering & home detection

# %%
home_extractor = gps_features.HomeClusterExtractor(
    df_loc,
    speed_limit=1.4,
    max_distance=150,
    epsilon=100,  # metres; Müller uses 30
    min_samples=600,  # weighted: >= 10 min cumulative dwell time within epsilon
    min_nights_obs=4,
    min_f_home=0.5,
    clustering_method="dbscan",
    normalize_min_samples=False,
    min_data_points=50,
    n_jobs=16,
    use_weights=True,
    home_v2=True,
    min_f_secondary_home=0.15,  # second home: >= 15 % of night time on >= min_nights_obs nights
    home_merge_m=500,  # split home: clusters within 500 m of the primary home
)

geodata_clusters = home_extractor.run()
logging.info("Home cluster extraction completed")

# %%
# sanity checks on the extractor output
assert np.all(
    (home_extractor.df["transition"] == 1) | (home_extractor.df["stationary"])
)
assert not np.any(
    (home_extractor.df["transition"] == 1) & (home_extractor.df["stationary"])
)

# %% [markdown]
# ### Daily metrics

# %%
df_gps_metrics = gps_features.calculate_metrics_fast(
    geodata_clusters, group_by=["id", "local_day"]
)
logging.info("GPS metrics calculated and stored in df_gps_metrics")

# Back from participant-key units to one row per participant-day.
# On days where the key switches, keep the unit with the longer active period.
m = df_gps_metrics.copy()
m["id_key"] = m["id"]
m["id"] = m["id_key"].str.rsplit("_k", n=1).str[0]
m["key_switch_day"] = m.groupby(["id", "local_day"])["id_key"].transform("size") > 1
df_gps_daily = (m.sort_values("active_period_hours", ascending=False)
                 .drop_duplicates(["id", "local_day"])
                 .sort_values(["id", "local_day"])
                 .reset_index(drop=True))
del m

print(f"GPS daily rows: {len(df_gps_daily)}")
print(f"GPS daily columns: {len(df_gps_daily.columns)}")
print(f"Unique participants: {df_gps_daily['id'].nunique()}")
print(
    f"Date range: {df_gps_daily['local_day'].min()} to {df_gps_daily['local_day'].max()}"
)
df_gps_daily.head()

# %% [markdown]
# ## Quality checks

# %%
# 1) Geometry: after normalisation the E-W / N-S step ratio should be ~1 for all units
#    The level is data-specific (here ~0.5-0.6 already before correction); what matters is that it
#    no longer depends on the anonymised latitude of the unit (Spearman rho ~ 0).
d = geodata_clusters[["id", "timestamp_start", "Latitude", "Longitude"]].sort_values(["id", "timestamp_start"])
g = d.groupby("id")
step = gps_features.haversine(d["Longitude"], d["Latitude"], g["Longitude"].shift(-1), g["Latitude"].shift(-1))
dns = (g["Latitude"].shift(-1) - d["Latitude"]).abs() * 111_320
dew = (g["Longitude"].shift(-1) - d["Longitude"]).abs() * 111_320 * np.cos(np.radians(d["Latitude"]))
s = pd.DataFrame({"id": d["id"], "dns": dns, "dew": dew})[step.between(20, 2000)]
iso = s.groupby("id").agg(ns=("dns", "median"), ew=("dew", "median"))
iso["ew_ns"] = iso["ew"] / iso["ns"]
iso["abs_lat_obf"] = df_loc.groupby("id")["lat_obf_center"].first().abs()
print("E-W / N-S step ratio per unit:")
print(iso["ew_ns"].describe())
print("Spearman rho with |anonymised latitude| (should be ~0):",
      round(iso[["ew_ns", "abs_lat_obf"]].corr(method="spearman").iloc[0, 1], 3))
plt.scatter(iso["abs_lat_obf"], iso["ew_ns"], s=8)
plt.yscale("log"); plt.xlabel("|anonymised latitude| of unit"); plt.ylabel("E-W / N-S (corrected)")
plt.title("Should be flat"); plt.show()
del d, g, step, dns, dew, s, iso

# %%
# 2) Distances: maximum should be a plausible travel day (metres)
print(df_gps_daily["total_distance"].describe(percentiles=[.5, .95, .99, .999]))
df_gps_daily.nlargest(10, "total_distance")[["id", "local_day", "total_distance", "key_switch_day"]]

# %%
# 3) Home coverage: share of each participant's days with a time-at-home value
cov = (df_gps_daily.groupby("id")["percentage_time_at_home"]
       .apply(lambda s: s.notna().mean())
       .sort_values())
print(cov.describe())
print(f"{(cov == 0).sum()} participants without any home days, "
      f"{(cov < 0.8).sum()} with <80% of days covered")
cov.head(30)

# %%
# 4) Units without a home: the two clusters with most night time
diag = home_extractor.home_diagnostics
no_home = geodata_clusters.groupby("id", observed=True)["home"].first()
fail = no_home[no_home.isna()].index

top2 = (diag[diag["id"].isin(fail)]
        .sort_values(["id", "night_sec"], ascending=[True, False])
        .groupby("id", observed=True).head(2))

cent = (geodata_clusters[geodata_clusters["cluster"] != -1]
        .groupby(["id", "cluster"], observed=True)
        .agg(lat=("Latitude", "mean"), lon=("Longitude", "mean"),
             first=("local_timestamp_start", "min"), last=("local_timestamp_start", "max"))
        .reset_index())
top2 = top2.drop(columns=["x", "y", "z", "px", "py", "pz"], errors="ignore").merge(
    cent, on=["id", "cluster"], how="left")

rows = []
for pid, grp in top2.groupby("id", observed=True):
    a = grp.iloc[0]
    b = grp.iloc[1] if len(grp) > 1 else None
    rows.append({
        "id": pid,
        "n_nights1": a["n_nights"],
        "f1": a["f_home"],
        "f2": b["f_home"] if b is not None else np.nan,
        "f1+f2": a["f_home"] + b["f_home"] if b is not None else np.nan,
        "dist_m": gps_features.haversine(a["lon"], a["lat"], b["lon"], b["lat"]) if b is not None else np.nan,
        "c1_period": f'{a["first"]:%Y-%m-%d} – {a["last"]:%Y-%m-%d}',
        "c2_period": f'{b["first"]:%Y-%m-%d} – {b["last"]:%Y-%m-%d}' if b is not None else "",
    })

if rows:
    summary = pd.DataFrame(rows).set_index("id").sort_values("f1", ascending=False)
    pd.set_option("display.width", 200)
    print(summary.round(2).to_string())
print(f"\n{len(rows)} units without home but with a primary night cluster (>= min_nights_obs); "
      f"{len(fail) - len(rows)} units with too few nights or no clustered night time")

# %% [markdown]
# ## Save

# %%
# save the aggregated daily data
if debug:
    print("!!!!!!!!!!!!!DEBUG MODE ON: Saving locally inside the repository!!!!!!!!!!!!!")
    output_dir = here() / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "temp_gps_daily_metrics_debug.feather"
else:
    passive_daily_dir = base_path / "preprocessed" / "passive" / "daily"
    output_path = passive_daily_dir / "daily_gps_metrics.feather"

df_gps_daily.to_feather(output_path)
print(f"GPS daily metrics saved to: {output_path}")
