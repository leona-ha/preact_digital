import pandas as pd
import numpy as np

# Complete sleep session aggregation - starts from df, produces df_sleep_sessions
# FULLY VECTORIZED - no loops!
def compute_sleep_sessions(
    df,
    max_gap_seconds=90 * 60,
    awakening_threshold=5 * 60,
    insomnia_sleep_threshold=6 * 60 * 60,
    insomnia_awake_threshold=30 * 60,
    hypersomnia_threshold=10 * 60 * 60,
):
    df = df[
        df["type"].isin(
            [
                "SleepDeepBinary",
                "SleepLightBinary",
                # "SleepREMBinary", # only 10 people got these records
                # "SnoringBinary", # only 2 people got these records #? maybe relevant for this 2 people?
                "SleepStateBinary",
                # "SleepBinary", # is included in SleepStateBinary
                "SleepInBedBinary",
                "SleepAwakeBinary",
            ]
        )
    ]
    df = df.sort_values(by=["customer", "startTimestamp"]).reset_index(drop=True)
    df["duration"] = (df["endTimestamp"] - df["startTimestamp"]).dt.total_seconds()


    # MAX_GAP_SECOND = 90 * 60  # 90 minutes as the maximum gap to consider same sleep session
    # AWAKENING_THRESHOLD = 5 * 60  # 5 minutes
    # INSOMNIA_SLEEP_THRESHOLD = 6 * 60 * 60  # 6 hours
    # INSOMNIA_AWAKE_THRESHOLD = 30 * 60  # 30 mins - at least one awakening longer than this
    # HYPERSOMNIA_THRESHOLD = 10 * 60 * 60  # 10 hours

    # Step 1: Identify sleep sessions from all the records
    # TODO might as well use only SleepState & SleepAwake records only
    df_sleepall = (
        # df[df["type"] == "SleepInBedBinary"]
        df.sort_values(by=["customer", "startTimestamp"]).reset_index(drop=True)
    )
    df_sleepall["next_sleep_record_start"] = df_sleepall.groupby("customer")[
        "startTimestamp"
    ].shift(-1)
    df_sleepall["gap_to_next"] = (
        df_sleepall["next_sleep_record_start"] - df_sleepall["endTimestamp"]
    )
    df_sleepall["is_lastentryinsession"] = (
        df_sleepall["gap_to_next"].dt.total_seconds() > max_gap_seconds
    ) | (df_sleepall["gap_to_next"].isna())
    df_sleepall["is_firstentryinsession"] = (
        df_sleepall.groupby("customer")["is_lastentryinsession"].shift(1).fillna(True)
    )
    df_sleepall["sleep_session_id"] = df_sleepall["is_firstentryinsession"].cumsum() - 1

    # Step 2: Create df_sleep_sessions with session boundaries
    df_sleep_sessions = (
        df_sleepall.groupby(["customer", "sleep_session_id"])
        .agg(
            {
                "startTimestamp": "first",
                "endTimestamp": "last",
                "local_start_time": "first",
                "local_end_time": "last",
            }
        )
        .reset_index()
    )
    df_sleep_sessions["sleep_session_duration"] = (
        df_sleep_sessions["endTimestamp"] - df_sleep_sessions["startTimestamp"]
    ).dt.total_seconds()

    # Step 3: Assign session IDs to all relevant records via merge + filter
    # relevant_types = [
    #     "SleepInBedBinary",
    #     "SleepAwakeBinary",
    #     "SleepLightBinary",
    #     "SleepDeepBinary",
    #     "SleepStateBinary",
    # ]
    # df_relevant = df[df["type"].isin(relevant_types)].copy()

    # Merge on customer only, then filter by time bounds (more memory but no sorting issues)
    sessions_for_merge = df_sleep_sessions[
        ["customer", "sleep_session_id", "startTimestamp", "endTimestamp"]
    ].rename(columns={"startTimestamp": "session_start", "endTimestamp": "session_end"})


    # with merge_asof, both times must be sorted
    df_with_session = (
        pd.merge_asof(
            df.sort_values(by=["startTimestamp", "customer"]),
            sessions_for_merge.sort_values(by=["session_start", "customer"]),
            left_on="startTimestamp",
            right_on="session_start",
            by="customer",
            direction="backward",
        )
        .sort_values(by=["customer", "startTimestamp"])
        .reset_index(drop=True)
    )

    # Step 4: Aggregate durations by session and type (vectorized!)
    duration_agg = (
        df_with_session.groupby(["customer", "sleep_session_id", "type"], observed=True)[
            "duration"
        ]
        .sum()
        .unstack(fill_value=0)
    )

    # Rename columns
    col_mapping = {
        # "SleepInBedBinary": "time_in_bed",
        "SleepInBedBinary": "SleepInBed_duration",
        "SleepAwakeBinary": "SleepAwake_duration",
        "SleepLightBinary": "SleepLight_duration",
        "SleepDeepBinary": "SleepDeep_duration",
        "SleepStateBinary": "total_sleep_time",
    }
    duration_agg = duration_agg.rename(columns=col_mapping)
    # Ensure all columns exist
    for col in col_mapping.values():
        if col not in duration_agg.columns:
            duration_agg[col] = 0
    duration_agg = duration_agg.reset_index()

    # Step 5: Calculate awakenings (vectorized with groupby + shift)
    sleepstate = (
        df_with_session[df_with_session["type"] == "SleepStateBinary"]
        .sort_values(["customer", "sleep_session_id", "startTimestamp"])
        .copy()
    )
    sleepstate["next_start"] = sleepstate.groupby(["customer", "sleep_session_id"])[
        "startTimestamp"
    ].shift(-1)
    sleepstate["gap_to_next"] = (
        sleepstate["next_start"] - sleepstate["endTimestamp"]
    ).dt.total_seconds()
    sleepstate["is_awakening"] = sleepstate["gap_to_next"] >= awakening_threshold
    sleepstate["is_long_awakening"] = sleepstate["gap_to_next"] >= insomnia_awake_threshold

    awakening_agg = (
        sleepstate.groupby(["customer", "sleep_session_id"])
        .agg(
            awakenings=("is_awakening", "sum"),
            long_awakenings=("is_long_awakening", "sum"),
            sleep_onset=("local_start_time", "min"),
            sleep_offset=("local_end_time", "max"),
        )
        .reset_index()
    )

    # Step 6: Merge all aggregations back to df_sleep_sessions
    df_sleep_sessions = df_sleep_sessions.merge(
        duration_agg, on=["customer", "sleep_session_id"], how="left"
    )
    df_sleep_sessions = df_sleep_sessions.merge(
        awakening_agg, on=["customer", "sleep_session_id"], how="left"
    )

    df_sleep_sessions["sleep_onset_hour"] = (
        df_sleep_sessions["sleep_onset"].dt.hour
        + df_sleep_sessions["sleep_onset"].dt.minute / 60
        + df_sleep_sessions["sleep_onset"].dt.second / 3600
    )
    df_sleep_sessions["sleep_offset_hour"] = (
        df_sleep_sessions["sleep_offset"].dt.hour
        + df_sleep_sessions["sleep_offset"].dt.minute / 60
        + df_sleep_sessions["sleep_offset"].dt.second / 3600
    )

    # it's more for training thing, so leave it for now
    # # change the onset hour to be > 24 if before 12 noon
    # SLEEP_ONSET_PLUS_24_HOUR = 12
    # df_sleep_sessions["sleep_onset_hour"] = df_sleep_sessions["sleep_onset_hour"].apply(
    #     lambda x: x + 24 if x < SLEEP_ONSET_PLUS_24_HOUR else x
    # )
    # # offset hour is fine as is

    # Fill NaN with 0 for numeric columns
    for col in [
        # "time_in_bed",
        "SleepInBed_duration",
        "SleepAwake_duration",
        "SleepLight_duration",
        "SleepDeep_duration",
        "total_sleep_time",
        "awakenings",
        "long_awakenings",
    ]:
        df_sleep_sessions[col] = df_sleep_sessions[col].fillna(0)

    # Step 7: Compute derived metrics
    #!!!! SleepInBed is wrongly computed for samples from 2023 (==0) as it wasn't recorded
    # df_sleep_sessions["time_in_bed"] = df_sleep_sessions["SleepInBed_duration"]
    df_sleep_sessions["time_in_bed"] = (
        df_sleep_sessions["total_sleep_time"] + df_sleep_sessions["SleepAwake_duration"]
    )

    df_sleep_sessions["time_out_of_bed"] = (
        df_sleep_sessions["sleep_session_duration"] - df_sleep_sessions["time_in_bed"]
    )
    df_sleep_sessions["sleep_efficiency"] = df_sleep_sessions[
        "total_sleep_time"
    ] / df_sleep_sessions["time_in_bed"].replace(0, np.nan)
    df_sleep_sessions["hypersomnia"] = (
        df_sleep_sessions["total_sleep_time"] >= hypersomnia_threshold
    )
    df_sleep_sessions["insomnia"] = (
        df_sleep_sessions["total_sleep_time"] <= insomnia_sleep_threshold
    ) & (df_sleep_sessions["long_awakenings"] >= 1)

    # !!
    df_sleep_sessions["awake_pct"] = df_sleep_sessions[
        "SleepAwake_duration"
    ] / df_sleep_sessions["time_in_bed"].replace(0, np.nan)
    df_sleep_sessions["light_sleep_pct"] = df_sleep_sessions[
        "SleepLight_duration"
    ] / df_sleep_sessions["time_in_bed"].replace(0, np.nan)
    df_sleep_sessions["deep_sleep_pct"] = df_sleep_sessions[
        "SleepDeep_duration"
    ] / df_sleep_sessions["time_in_bed"].replace(0, np.nan)

    # Step 8: Additional session-level features
    # assign the day of the sleep session based on the wake up day (local)
    df_sleep_sessions["day"] = df_sleep_sessions["local_end_time"].dt.date
    df_sleep_sessions["num_sessions_in_day"] = df_sleep_sessions.groupby(
        ["customer", "day"]
    )["sleep_session_id"].transform("count")

    return df_sleep_sessions
