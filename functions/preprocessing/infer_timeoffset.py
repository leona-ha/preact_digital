# %%
"""
Infer timezone offsets per (participant, day) from TONI sensor data.

Produces an `inferred_tzoffset` (minutes from UTC) and an
`inferred_tzoffset_source` string that records *how* the offset was determined.

Source-string construction
==========================
Most values follow the pattern:

    {data_source}_{resolution}[_dst_adjusted]

Resolution proceeds through five priority stages. Each stage only fills days
that are still missing after all previous stages.

Stage 1 - Direct single-value assignment
-----------------------------------------
- ``gps_single``
    GPS records for the (id, day) contain exactly one unique offset -> used directly.
- ``activitydetailcreatedat_single``
    ActivityTypeDetail createdAt records for the (id, day) contain exactly one
    unique offset -> used as fallback when GPS is absent or ambiguous.

Stage 2 - GPS with multiple offsets (``gps_multiple_`` prefix)
--------------------------------------------------------------
When GPS provides >1 distinct offset for a day, the nearest resolved neighbors
for the same participant are consulted. The neighbor's offset must appear in
the GPS candidate set. Suffix describes the outcome:

- ``both_neighbors_agree``                   - previous & next neighbor share the same offset.
- ``previous_is_fine_no_next``               - previous neighbor valid; no next neighbor exists.
- ``no_previous_next_is_fine``               - next neighbor valid; no previous neighbor exists.
- ``conflict_with_next_previous_is_fine``    - previous valid; next exists but not in candidates.
- ``conflict_with_previous_next_is_fine``    - next valid; previous exists but not in candidates.
- ``inferred_from_previous_{N}d``            - both valid but disagree; previous is closer (N = days).
- ``inferred_from_previous_{N}d_equal_dist`` - same, but equidistant; previous wins by convention.
- ``inferred_from_next_{N}d``                - both valid but disagree; next is strictly closer.
- ``no_previous_no_next``                    - no resolved neighbors at all (unresolved).
- ``no_previous_conflict_with_next``         - no previous; next not in candidates (unresolved).
- ``conflict_with_previous_no_next``         - previous not in candidates; no next (unresolved).
- ``conflict_with_both``                     - neither neighbor in candidates (unresolved).

Stage 3 - ActivityDetail with multiple offsets (``activitydetail_multiple_`` prefix)
------------------------------------------------------------------------------------
Same neighbor logic and suffixes as Stage 2, applied to ActivityTypeDetail
createdAt data for days still unresolved after Stage 2.

Stage 4 - Unconstrained neighbor interpolation (``interpolate_`` prefix)
------------------------------------------------------------------------
For remaining gaps, neighbor lookup runs with *no* candidate restriction and
``prefer_previous_than_distance=True`` (previous neighbor always wins over a
closer next neighbor). Same suffixes as Stages 2/3.

Stage 5 - Berlin fallback
--------------------------
- ``assumed_berlin``
    No sensor data or neighbors could determine the offset. The DST-aware
    Berlin offset for that calendar day (evaluated at noon UTC) is assumed.

DST adjustment suffix (``_dst_adjusted``)
-----------------------------------------
``adjust_dst_changes`` forces the offset on Berlin DST-transition days:
  - March transition -> offset forced to 120 (CEST / summer).
  - October transition -> offset forced to 60 (CET / winter).
It is called (without appending a suffix) after Stages 1, 2, and 3.
It is **not** called after Stage 4 (interpolation).
A final call with ``update_source=True`` runs after Stage 5 (Berlin fallback),
which appends ``_dst_adjusted`` to the source string for any corrected rows.

merge_fill_tz additional source
-------------------------------
- ``ffilled``
    When merging ``df_tz`` into another DataFrame, days absent from ``df_tz``
    are forward-filled from the most recent known day per participant, with a
    post-hoc DST correction if the filled value crosses a transition boundary.
"""
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def midnight_timestamp_to_berlin_tzoffset(ts):
    # it's dst aware
    if isinstance(ts, pd.Series):
        return (ts + pd.Timedelta(hours=12)).dt.tz_convert("Europe/Berlin").map(
            lambda x: x.utcoffset()
        ).dt.total_seconds() / 60
    elif isinstance(ts, pd.Timestamp):
        return (ts + pd.Timedelta(hours=12)).tz_convert("Europe/Berlin").utcoffset().total_seconds() / 60


def create_utcday_tzoffset_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with inferred timezone offsets for each id and day.
    Assumes TONI-style structure (long)."""

    df = df[["id", "timestamp_start", "timezone_offset", "modality", "createdAt"]].copy()

    # EXACT original logic: day-level
    df["startTimestamp_day"] = df["timestamp_start"].dt.floor("D")
    df["createdAt_day"] = df["createdAt"].dt.floor("D")

    df_tz = df[["id", "startTimestamp_day"]].drop_duplicates().reset_index(drop=True)
    df_tz = df_tz.sort_values(["id", "startTimestamp_day"]).reset_index(drop=True)

    # Berlin DST changes
    date_range = pd.date_range(
        f"{df['startTimestamp_day'].min().year}",
        f"{df['startTimestamp_day'].max().year + 1}",
        freq="h",
        tz="Europe/Berlin",
    )

    berlin_offsets = pd.Series(date_range.map(lambda x: x.utcoffset()), index=date_range)

    offset_changes = berlin_offsets != berlin_offsets.shift(1)
    dst_changes = date_range[offset_changes][1:]  # remove first (shift vs NaN)

    df_tz["dst_berlin_change_day"] = df_tz.startTimestamp_day.dt.date.isin(dst_changes.date)

    # --- df tz gps
    df_tz_gps = (
        df[df["modality"].isin(["Latitude", "Longitude"])]
        .groupby(["id", "startTimestamp_day"], observed=True)["timezone_offset"]
        .apply(lambda x: x.unique())
        .reset_index(name="tzs_gps")
    )

    df_tz = df_tz.merge(df_tz_gps, on=["id", "startTimestamp_day"], how="left")
    df_tz["gps_exists"] = ~df_tz["tzs_gps"].isna()

    # --- df tz activitydetail createdAt
    df_tz_activitydetailcreatedat = (
        df[df["modality"].isin(["ActivityTypeDetail1", "ActivityTypeDetail2"])]
        .groupby(["id", "createdAt_day"], observed=True)["timezone_offset"]
        .apply(lambda x: x.unique())
        .reset_index(name="tzs_activitydetailcreatedat")
    )

    df_tz = df_tz.merge(
        df_tz_activitydetailcreatedat.rename(columns={"createdAt_day": "startTimestamp_day"}),
        on=["id", "startTimestamp_day"],
        how="left",
    )

    # --- return_tz
    df_tz["return_tz"] = pd.NA
    df_tz["return_source"] = pd.NA

    mask_gps = df_tz.gps_exists & (df_tz.tzs_gps.dropna().apply(len) == 1)
    df_tz.loc[mask_gps, "return_tz"] = df_tz.tzs_gps[mask_gps].apply(lambda x: x[0])
    df_tz.loc[mask_gps, "return_source"] = "gps_single"

    mask_activitydetailcreatedat = (~mask_gps) & (
        df_tz.tzs_activitydetailcreatedat.dropna().apply(len) == 1
    )
    df_tz.loc[mask_activitydetailcreatedat, "return_tz"] = (
        df_tz.tzs_activitydetailcreatedat[mask_activitydetailcreatedat].apply(lambda x: x[0])
    )
    df_tz.loc[mask_activitydetailcreatedat, "return_source"] = "activitydetailcreatedat_single"

    def adjust_dst_changes(df_tz, update_source=False, inplace=False):
        """Adjust for DST changes in Berlin timezone."""
        if not inplace:
            df_tz = df_tz.copy()

        change2summer = df_tz.dst_berlin_change_day & (df_tz.startTimestamp_day.dt.month == 3)
        change2winter = df_tz.dst_berlin_change_day & (df_tz.startTimestamp_day.dt.month == 10)

        dst_summer_mask = change2summer & df_tz.return_tz.isin([60, 120])
        dst_winter_mask = change2winter & df_tz.return_tz.isin([60, 120])

        df_tz.loc[dst_summer_mask, "return_tz"] = 120
        df_tz.loc[dst_winter_mask, "return_tz"] = 60

        if update_source:
            df_tz.loc[dst_summer_mask, "return_source"] += "_dst_adjusted"
            df_tz.loc[dst_winter_mask, "return_source"] += "_dst_adjusted"

        return df_tz

    df_tz = adjust_dst_changes(df_tz, inplace=True)

    def infer_timezone_from_neighbors(row, df_tz, potential_tzs=None, prefer_previous_than_distance=False):
        prev_df = df_tz[
            (df_tz.id == row.id)
            & (df_tz.startTimestamp_day < row.startTimestamp_day)
            & (df_tz.return_tz.notna())
        ].sort_values("startTimestamp_day")
        prev_available = prev_df.iloc[-1] if not prev_df.empty else None

        next_df = df_tz[
            (df_tz.id == row.id)
            & (df_tz.startTimestamp_day > row.startTimestamp_day)
            & (df_tz.return_tz.notna())
        ].sort_values("startTimestamp_day")
        next_available = next_df.iloc[0] if not next_df.empty else None

        if potential_tzs is None:
            prev_valid = prev_available is not None
            next_valid = next_available is not None
        else:
            prev_valid = prev_available is not None and prev_available.return_tz in potential_tzs
            next_valid = next_available is not None and next_available.return_tz in potential_tzs

        if not prev_valid and not next_valid:
            if prev_available is None and next_available is None:
                return None, "no_previous_no_next"
            return (
                None,
                "no_previous_conflict_with_next"
                if prev_available is None
                else ("conflict_with_previous_no_next" if next_available is None else "conflict_with_both"),
            )

        if prev_valid and not next_valid:
            return (
                prev_available.return_tz,
                "previous_is_fine_no_next" if next_available is None else "conflict_with_next_previous_is_fine",
            )

        if next_valid and not prev_valid:
            return (
                next_available.return_tz,
                "no_previous_next_is_fine" if prev_available is None else "conflict_with_previous_next_is_fine",
            )

        if prev_available.return_tz == next_available.return_tz:
            return prev_available.return_tz, "both_neighbors_agree"

        days_to_prev = (row.startTimestamp_day - prev_available.startTimestamp_day).days
        days_to_next = (next_available.startTimestamp_day - row.startTimestamp_day).days

        if days_to_prev <= days_to_next or prefer_previous_than_distance:
            return (
                prev_available.return_tz,
                f"inferred_from_previous_{days_to_prev}d"
                + ("_equal_dist" if days_to_prev == days_to_next else ""),
            )
        return next_available.return_tz, f"inferred_from_next_{days_to_next}d"

    # GPS multiple
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[
        df_tz.tzs_gps.dropna().apply(lambda x: len(x) > 1) & df_tz.return_tz.isna()
    ].iterrows():
        inferred_tz, source = infer_timezone_from_neighbors(row, df_tz_before, row.tzs_gps)
        source = "gps_multiple_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(
                f"[GPS] Could not infer tz for id {row.id} at {row.startTimestamp_day} "
                f"with potential tzs {row.tzs_gps} - {source}"
            )

    df_tz = adjust_dst_changes(df_tz)

    # ActivityDetail multiple
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[
        df_tz.tzs_activitydetailcreatedat.dropna().apply(lambda x: len(x) > 1)
        & df_tz.return_tz.isna()
    ].iterrows():
        potential_tzs = row.tzs_activitydetailcreatedat
        inferred_tz, source = infer_timezone_from_neighbors(row, df_tz_before, potential_tzs)
        source = "activitydetail_multiple_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(
                f"[ActivityDetailCreatedAt] Could not infer tz for id {row.id} at {row.startTimestamp_day} "
                f"with potential tzs {potential_tzs} - {source}"
            )

    df_tz = adjust_dst_changes(df_tz)

    # final interpolation
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[df_tz.return_tz.isna()].iterrows():
        inferred_tz, source = infer_timezone_from_neighbors(
            row, df_tz_before, potential_tzs=None, prefer_previous_than_distance=True
        )
        source = "interpolate_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(f"[Interpolate] Could not infer tz for id {row.id} at {row.startTimestamp_day} - {source}")

    # fill remaining with berlin timezone
    still_missing_mask = df_tz.return_tz.isna()
    if still_missing_mask.any():
        df_tz.loc[still_missing_mask, "return_tz"] = (
            df_tz.loc[still_missing_mask, "startTimestamp_day"] + pd.Timedelta(hours=12)
        ).dt.tz_convert("Europe/Berlin").map(lambda x: x.utcoffset()).dt.total_seconds() / 60
        df_tz.loc[still_missing_mask, "return_source"] = "assumed_berlin"

    df_tz = adjust_dst_changes(df_tz, update_source=True)

    df_tz["return_source"] = df_tz["return_source"].astype("category")

    # finalize
    df_tz = df_tz.rename(
        columns={
            "startTimestamp_day": "day",
            "return_tz": "inferred_tzoffset",
            "return_source": "inferred_tzoffset_source",
        }
    )
    df_tz = df_tz[["id", "day", "inferred_tzoffset", "inferred_tzoffset_source"]]
    return df_tz


def merge_fill_tz(df, df_tz, day_col="date", customer_col="id"):
    """
    Same logic as your original, but adjusted to df_tz using ['id','day',...].
    """
    assert (
        set(["inferred_tzoffset", "inferred_tzoffset_source"]) & set(df.columns)
        == set()
    ), (
        "Input df already contains 'inferred_tzoffset' or 'inferred_tzoffset_source' columns! "
        "Drop them before calling this function"
    )

    df_merged = pd.merge(
        df,
        df_tz,
        left_on=[customer_col, day_col],
        right_on=[customer_col, "day"],
        how="left",
        suffixes=("", ""),
    ).drop(columns=["day"])

    missing_tz = df_merged["inferred_tzoffset"].isna()
    if missing_tz.sum() == 0:
        logging.info("No missing timezone offsets to fill ✅.")
        df_merged["inferred_tzoffset"] = df_merged["inferred_tzoffset"].astype("int")
        df_merged["inferred_tzoffset_source"] = df_merged["inferred_tzoffset_source"].astype("category")
        return df_merged

    df_merged["inferred_tzoffset"] = df_merged.groupby(customer_col)["inferred_tzoffset"].ffill()
    df_merged["inferred_tzoffset_source"] = df_merged["inferred_tzoffset_source"].astype("string")

    ffilled = missing_tz & df_merged["inferred_tzoffset"].notna()
    df_merged.loc[ffilled, "inferred_tzoffset_source"] = "ffilled"
    logging.info(f"Forward-filled timezone offsets: {ffilled.sum()}")

    if ffilled.sum() > 0:
        expected_berlin_offset = midnight_timestamp_to_berlin_tzoffset(df_merged.loc[ffilled, day_col])

        is_summer_expected = expected_berlin_offset == 120
        is_winter_expected = expected_berlin_offset == 60

        needs_summer_correction = is_summer_expected & (df_merged.loc[ffilled, "inferred_tzoffset"] == 60)
        needs_winter_correction = is_winter_expected & (df_merged.loc[ffilled, "inferred_tzoffset"] == 120)

        ffilled_idx = df_merged.index[ffilled]
        df_merged.loc[ffilled_idx[needs_summer_correction], "inferred_tzoffset"] = 120
        df_merged.loc[ffilled_idx[needs_winter_correction], "inferred_tzoffset"] = 60

        logging.info(f"Corrected to summer time (120 min): {needs_summer_correction.sum()}")
        logging.info(f"Corrected to winter time (60 min): {needs_winter_correction.sum()}")

    still_missing = df_merged["inferred_tzoffset"].isna()
    if still_missing.sum() > 0:
        logging.info(f"Assumed Berlin timezone offsets: {still_missing.sum()}")
        df_merged.loc[still_missing, "inferred_tzoffset"] = midnight_timestamp_to_berlin_tzoffset(
            df_merged.loc[still_missing, day_col]
        )
        df_merged.loc[still_missing, "inferred_tzoffset_source"] = "assumed_berlin"

    df_merged["inferred_tzoffset"] = df_merged["inferred_tzoffset"].astype("int")
    df_merged["inferred_tzoffset_source"] = df_merged["inferred_tzoffset_source"].astype("category")

    assert df_merged["inferred_tzoffset"].isna().sum() == 0, "Failed to fill all timezone offsets!"
    return df_merged