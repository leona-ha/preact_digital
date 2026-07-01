import pandas as pd
import numpy as np

import logging


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
        df["modality"].isin(
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
    df = df.sort_values(by=["id", "timestamp_start"]).reset_index(drop=True)
    df["duration"] = (df["timestamp_end"] - df["timestamp_start"]).dt.total_seconds()

    # MAX_GAP_SECOND = 90 * 60  # 90 minutes as the maximum gap to consider same sleep session
    # AWAKENING_THRESHOLD = 5 * 60  # 5 minutes
    # INSOMNIA_SLEEP_THRESHOLD = 6 * 60 * 60  # 6 hours
    # INSOMNIA_AWAKE_THRESHOLD = 30 * 60  # 30 mins - at least one awakening longer than this
    # HYPERSOMNIA_THRESHOLD = 10 * 60 * 60  # 10 hours

    # Step 1: Identify sleep sessions from all the records
    # TODO might as well use only SleepState & SleepAwake records only
    df_sleepall = (
        # df[df["modality"] == "SleepInBedBinary"]
        df.sort_values(by=["id", "timestamp_start"]).reset_index(drop=True)
    )
    df_sleepall["next_sleep_record_start"] = df_sleepall.groupby("id", observed=True)[
        "timestamp_start"
    ].shift(-1)
    df_sleepall["gap_to_next"] = (
        df_sleepall["next_sleep_record_start"] - df_sleepall["timestamp_end"]
    )
    df_sleepall["is_lastentryinsession"] = (
        df_sleepall["gap_to_next"].dt.total_seconds() > max_gap_seconds
    ) | (df_sleepall["gap_to_next"].isna())
    df_sleepall["is_firstentryinsession"] = (
        df_sleepall.groupby("id", observed=True)["is_lastentryinsession"]
        .shift(1)
        .fillna(True)
    )
    df_sleepall["sleep_session_id"] = df_sleepall["is_firstentryinsession"].cumsum() - 1

    # Step 2: Create df_sleep_sessions with session boundaries
    df_sleep_sessions = (
        df_sleepall.groupby(["id", "sleep_session_id"], observed=True)
        .agg(
            {
                "timestamp_start": "first",
                "timestamp_end": "last",  # TODO change to max ?
                "local_timestamp_start": "first",
                "local_timestamp_end": "last",  #! and here
            }
        )
        .reset_index()
    )
    df_sleep_sessions["sleep_session_duration"] = (
        df_sleep_sessions["timestamp_end"] - df_sleep_sessions["timestamp_start"]
    ).dt.total_seconds()

    # Step 3: Assign session IDs to all relevant records via merge + filter
    # relevant_types = [
    #     "SleepInBedBinary",
    #     "SleepAwakeBinary",
    #     "SleepLightBinary",
    #     "SleepDeepBinary",
    #     "SleepStateBinary",
    # ]
    # df_relevant = df[df["modality"].isin(relevant_types)].copy()

    # Merge on id only, then filter by time bounds (more memory but no sorting issues)
    sessions_for_merge = df_sleep_sessions[
        ["id", "sleep_session_id", "timestamp_start", "timestamp_end"]
    ].rename(
        columns={"timestamp_start": "session_start", "timestamp_end": "session_end"}
    )

    # with merge_asof, both times must be sorted
    df_with_session = (
        pd.merge_asof(
            df.sort_values(by=["timestamp_start", "id"]),
            sessions_for_merge.sort_values(by=["session_start", "id"]),
            left_on="timestamp_start",
            right_on="session_start",
            by="id",
            direction="backward",
        )
        .sort_values(by=["id", "timestamp_start"])
        .reset_index(drop=True)
    )

    # Step 4: Aggregate durations by session and type (vectorized!)
    duration_agg = (
        df_with_session.groupby(["id", "sleep_session_id", "modality"], observed=True)[
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
        df_with_session[df_with_session["modality"] == "SleepStateBinary"]
        .sort_values(["id", "sleep_session_id", "timestamp_start"])
        .copy()
    )
    sleepstate["next_start"] = sleepstate.groupby(
        ["id", "sleep_session_id"], observed=True
    )["timestamp_start"].shift(-1)
    sleepstate["gap_to_next"] = (
        sleepstate["next_start"] - sleepstate["timestamp_end"]
    ).dt.total_seconds()
    sleepstate["is_awakening"] = sleepstate["gap_to_next"] >= awakening_threshold
    sleepstate["is_long_awakening"] = (
        sleepstate["gap_to_next"] >= insomnia_awake_threshold
    )

    awakening_agg = (
        sleepstate.groupby(["id", "sleep_session_id"], observed=True)
        .agg(
            awakenings=("is_awakening", "sum"),
            long_awakenings=("is_long_awakening", "sum"),
            sleep_onset=("local_timestamp_start", "min"),
            sleep_offset=("local_timestamp_end", "max"),
        )
        .reset_index()
    )

    # Step 6: Merge all aggregations back to df_sleep_sessions
    df_sleep_sessions = df_sleep_sessions.merge(
        duration_agg, on=["id", "sleep_session_id"], how="left"
    )
    df_sleep_sessions = df_sleep_sessions.merge(
        awakening_agg, on=["id", "sleep_session_id"], how="left"
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
    df_sleep_sessions["day"] = df_sleep_sessions["local_timestamp_end"].dt.date
    df_sleep_sessions["num_sessions_in_day"] = df_sleep_sessions.groupby(
        ["id", "day"], observed=True
    )["sleep_session_id"].transform("count")

    return df_sleep_sessions


MAP_ACTIVITYTYPE = {
    102: "SLEEP",
    103: "REST",
    104: "ACTIVE",
    105: "WALK",
    106: "RUN",
    107: "BIKE",
    108: "TRANSPORT",
    110: "LEISURE",
}

MAP_ACTIVITYTYPEDETAIL1 = {
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
    220: "AIR_SPORTS",
}

MAP_ACTIVITYTYPEDETAIL2 = {
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


def aggregate_sleep_daily(df_backup, max_gap_seconds=5400):
    """
    Aggregate sleep data to daily level.

    Returns a DataFrame with both longest session metrics and summed daily metrics.
    Uses local wake-up day as the day reference.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with sleep records
    max_gap_seconds : int
        Maximum gap in seconds to consider same sleep session (default 90 minutes)

    Returns
    -------
    pd.DataFrame
        Daily sleep aggregates with columns: id, local_day, and sleep metrics

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | Local wake-up day used for daily aggregation | Grouping key |
    | **`longest_sleep_session_duration`** | Duration (seconds) of the longest sleep session in day | Longest session by `sleep_session_duration` |
    | **`longest_total_sleep_time`** | Total sleep time (seconds) in the longest sleep session | `SleepStateBinary` duration in longest session |
    | **`longest_time_in_bed`** | Time in bed (seconds) in the longest sleep session | `total_sleep_time + SleepAwake_duration` in longest session |
    | **`longest_time_out_of_bed`** | Time out of bed (seconds) during longest sleep session window | `sleep_session_duration - time_in_bed` in longest session |
    | **`longest_SleepAwake_duration`** | Awake duration (seconds) in longest sleep session | Sum of `SleepAwakeBinary` duration in longest session |
    | **`longest_SleepLight_duration`** | Light sleep duration (seconds) in longest sleep session | Sum of `SleepLightBinary` duration in longest session |
    | **`longest_SleepDeep_duration`** | Deep sleep duration (seconds) in longest sleep session | Sum of `SleepDeepBinary` duration in longest session |
    | **`longest_awakenings`** | Number of awakenings in longest sleep session | Count of gaps >= awakening threshold |
    | **`longest_long_awakenings`** | Number of long awakenings in longest sleep session | Count of gaps >= insomnia-awake threshold |
    | **`longest_sleep_onset_hour`** | Local sleep onset clock-time (decimal hour) for longest session | Derived from local onset timestamp |
    | **`longest_sleep_offset_hour`** | Local sleep offset clock-time (decimal hour) for longest session | Derived from local offset timestamp |
    | **`longest_sleep_efficiency`** | Sleep efficiency for longest session | `total_sleep_time / time_in_bed` |
    | **`longest_hypersomnia`** | Hypersomnia flag for longest session | `total_sleep_time >= hypersomnia_threshold` |
    | **`longest_insomnia`** | Insomnia flag for longest session | `total_sleep_time <= insomnia_sleep_threshold` and >=1 long awakening |
    | **`longest_awake_pct`** | Awake proportion in bed for longest session | `SleepAwake_duration / time_in_bed` |
    | **`longest_light_sleep_pct`** | Light-sleep proportion in bed for longest session | `SleepLight_duration / time_in_bed` |
    | **`longest_deep_sleep_pct`** | Deep-sleep proportion in bed for longest session | `SleepDeep_duration / time_in_bed` |
    | **`sum_sleep_session_duration`** | Total sleep-session duration (seconds) across all sessions in day | Sum across sessions |
    | **`sum_total_sleep_time`** | Total sleep time (seconds) across all sessions in day | Sum across sessions |
    | **`sum_time_in_bed`** | Total time in bed (seconds) across all sessions in day | Sum across sessions |
    | **`sum_time_out_of_bed`** | Total time out of bed (seconds) across all sessions in day | Sum across sessions |
    | **`sum_SleepAwake_duration`** | Total awake duration (seconds) across all sessions in day | Sum across sessions |
    | **`sum_SleepLight_duration`** | Total light sleep duration (seconds) across all sessions in day | Sum across sessions |
    | **`sum_SleepDeep_duration`** | Total deep sleep duration (seconds) across all sessions in day | Sum across sessions |
    | **`sum_awakenings`** | Total awakening count across all sessions in day | Sum across sessions |
    | **`sum_long_awakenings`** | Total long-awakening count across all sessions in day | Sum across sessions |
    | **`num_sessions_in_day`** | Number of sleep sessions assigned to that day | Session count per id-day |
    """
    # Compute sleep sessions using existing function
    # ! the bottleneck is here, if to optimize, optimize compute_sleep_sessions
    df_sessions = compute_sleep_sessions(df_backup, max_gap_seconds=max_gap_seconds)

    # Convert day to datetime for consistency
    df_sessions["local_day"] = pd.to_datetime(df_sessions["day"])

    # Get longest session per id-day
    df_longest = df_sessions.loc[
        df_sessions.groupby(["id", "local_day"], observed=True)[
            "sleep_session_duration"
        ].idxmax()
    ].reset_index(drop=True)

    # Select columns for longest session (prefix with "longest_")
    longest_cols = [
        "sleep_session_duration",
        "total_sleep_time",
        "time_in_bed",
        "time_out_of_bed",
        "SleepAwake_duration",
        "SleepLight_duration",
        "SleepDeep_duration",
        "awakenings",
        "long_awakenings",
        "sleep_onset_hour",
        "sleep_offset_hour",
        "sleep_efficiency",
        "hypersomnia",
        "insomnia",
        "awake_pct",
        "light_sleep_pct",
        "deep_sleep_pct",
    ]

    df_longest_subset = df_longest[["id", "local_day"] + longest_cols].copy()
    df_longest_subset = df_longest_subset.rename(
        columns={col: f"longest_{col}" for col in longest_cols}
    )

    # Sum across all sessions per day
    df_summed = (
        df_sessions.groupby(["id", "local_day"], observed=True)
        .agg(
            sum_sleep_session_duration=("sleep_session_duration", "sum"),
            sum_total_sleep_time=("total_sleep_time", "sum"),
            sum_time_in_bed=("time_in_bed", "sum"),
            sum_time_out_of_bed=("time_out_of_bed", "sum"),
            sum_SleepAwake_duration=("SleepAwake_duration", "sum"),
            sum_SleepLight_duration=("SleepLight_duration", "sum"),
            sum_SleepDeep_duration=("SleepDeep_duration", "sum"),
            sum_awakenings=("awakenings", "sum"),
            sum_long_awakenings=("long_awakenings", "sum"),
            num_sessions_in_day=("num_sessions_in_day", "first"),
        )
        .reset_index()
    )

    # Merge longest and summed
    df_sleep_daily = df_longest_subset.merge(
        df_summed, on=["id", "local_day"], how="outer"
    )

    return df_sleep_daily


# TODO HR zones - change the algorithm
def aggregate_hr_daily(df_backup, zone_thresholds=(60, 100), include_zero_hours=False):
    """
    Aggregate heart rate data to daily level.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with HeartRate records
    zone_thresholds : tuple
        (resting_upper, vigorous_lower) thresholds for HR zones
    include_zero_hours : bool
        Whether to include hours with zero records when computing per-hour stats

    Returns
    -------
    pd.DataFrame
        Daily HR aggregates with columns: id, local_day, and HR metrics

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | The local calendar day of the records (floored to midnight) | Grouping key |
    | **`HR_count`** | Total number of 1-second HR records in the day | Calculated on 1-second expanded data |
    | **`HR_mean`** | Average heart rate over the day | Calculated on 1-second expanded data |
    | **`HR_std`** | Standard deviation of heart rate | Calculated on 1-second expanded data |
    | **`HR_min`** | Minimum heart rate recorded in the day | Calculated on 1-second expanded data |
    | **`HR_max`** | Maximum heart rate recorded in the day | Calculated on 1-second expanded data |
    | **`HR_skew`** | Skewness of the heart rate distribution | Calculated on 1-second expanded data |
    | **`HR_kurtosis`** | Kurtosis of the heart rate distribution | Calculated on 1-second expanded data |
    | **`HR_zone_resting`** | Total seconds spent with HR below resting threshold (default < 60) | Calculated on 1-second expanded data |
    | **`HR_zone_moderate`** | Total seconds spent with HR between resting and vigorous thresholds (default 60 ≤ HR < 100) | Calculated on 1-second expanded data |
    | **`HR_zone_vigorous`** | Total seconds spent with HR above vigorous threshold (default ≥ 100) | Calculated on 1-second expanded data |
    | **`HR_raw_records`** | Total count of raw HR records logged per day | Calculated on raw unexpanded records |
    | **`HR_raw_hours_with_records`** | Number of unique hours in the day that contain at least one raw HR record | Calculated on raw unexpanded records |
    | **`HR_raw_records_per_hour_mean`** | Mean of raw records per hour (averaged **only over hours with data**) | Calculated on raw unexpanded records |
    | **`HR_raw_records_per_hour_median`**| Median of raw records per hour (calculated **only over hours with data**) | Calculated on raw unexpanded records |
    | **`HR_raw_records_per_hour_std`** | Standard deviation of raw records per hour (calculated **only over hours with data**) | Calculated on raw unexpanded records |
    | **`HR_seconds_hours_with_data`** | Number of unique hours in the day containing 1-second expanded HR data | Calculated on 1-second expanded data |
    | **`HR_seconds_per_hour_mean`** | Mean of 1-second HR records per hour (averaged **only over hours with data**) | Calculated on 1-second expanded data |
    | **`HR_seconds_per_hour_median`** | Median of 1-second HR records per hour (calculated **only over hours with data**) | Calculated on 1-second expanded data |
    | **`HR_seconds_per_hour_std`** | Standard deviation of 1-second HR records per hour (calculated **only over hours with data**) | Calculated on 1-second expanded data |
    | **`HR_coverage`** | Percentage of the day covered by HR recordings (`HR_raw_hours_with_records / 24`) | Derived from raw coverage stats |
    """
    resting_threshold, vigorous_threshold = zone_thresholds

    # Filter HR data
    df = df_backup[df_backup["modality"] == "HeartRate"].copy()
    df = df[
        [
            "id",
            "timestamp_start",
            "timestamp_end",
            "float_value",
            "local_timestamp_start",
        ]
    ].copy()
    df = df.rename(columns={"float_value": "HeartRate"})

    # Compute duration and expand to 1-second resolution
    df["duration"] = (
        (df["timestamp_end"] - df["timestamp_start"])
        .dt.total_seconds()
        .fillna(0)
        .astype(int)
    )
    df = df[df["duration"] > 0]  # Remove zero-duration records

    # Record-level daily coverage (pre-expansion)
    df["local_day"] = df["local_timestamp_start"].dt.floor("D")
    df["local_hour"] = df["local_timestamp_start"].dt.floor("h")

    df_hr_raw_daily = (
        df.groupby(["id", "local_day"], observed=True)
        .size()
        .reset_index(name="HR_raw_records")
    )

    df_hr_raw_hourly = (
        df.groupby(["id", "local_day", "local_hour"], observed=True)
        .size()
        .reset_index(name="HR_raw_records_in_hour")
    )

    df_hr_raw_hours_with_records = (
        df_hr_raw_hourly.groupby(["id", "local_day"], observed=True)["local_hour"]
        .nunique()
        .reset_index(name="HR_raw_hours_with_records")
    )

    # Expand rows
    df_expanded = df.loc[df.index.repeat(df["duration"])].copy()
    df_expanded["time_offset"] = df_expanded.groupby(level=0).cumcount()
    df_expanded["timestamp"] = df_expanded["timestamp_start"] + pd.to_timedelta(
        df_expanded["time_offset"], unit="s"
    )
    df_expanded["local_timestamp"] = df_expanded[
        "local_timestamp_start"
    ] + pd.to_timedelta(df_expanded["time_offset"], unit="s")

    # Deduplicate overlapping entries via mean
    df_avg = (
        df_expanded.groupby(["id", "timestamp"], observed=True)
        .agg(
            HeartRate=("HeartRate", "mean"),
            local_timestamp=("local_timestamp", "first"),
        )
        .reset_index()
    )

    # Add HR zones
    df_avg["HR_zone_resting"] = df_avg["HeartRate"] < resting_threshold
    df_avg["HR_zone_moderate"] = (df_avg["HeartRate"] >= resting_threshold) & (
        df_avg["HeartRate"] < vigorous_threshold
    )
    df_avg["HR_zone_vigorous"] = df_avg["HeartRate"] >= vigorous_threshold

    # Add local day
    df_avg["local_day"] = df_avg["local_timestamp"].dt.floor("D")
    df_avg["local_hour"] = df_avg["local_timestamp"].dt.floor("h")

    # Second-level hourly coverage (post-expansion)
    df_hr_seconds_hourly = (
        df_avg.groupby(["id", "local_day", "local_hour"], observed=True)["HeartRate"]
        .count()
        .reset_index(name="HR_seconds_in_hour")
    )

    df_hr_seconds_hours_with_data = (
        df_hr_seconds_hourly.groupby(["id", "local_day"], observed=True)["local_hour"]
        .nunique()
        .reset_index(name="HR_seconds_hours_with_data")
    )

    # Daily aggregation
    df_hr_daily = (
        df_avg.groupby(["id", "local_day"], observed=True)
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

    # Per-hour stats (record-level)
    if include_zero_hours:
        df_hr_raw_days = df_hr_raw_daily[["id", "local_day"]].drop_duplicates()
        df_hr_raw_days = df_hr_raw_days.assign(_join_key=1)
        df_hr_raw_hours = pd.DataFrame({"hour": range(24), "_join_key": 1})
        df_hr_raw_full = df_hr_raw_days.merge(df_hr_raw_hours, on="_join_key").drop(
            columns=["_join_key"]
        )
        df_hr_raw_full["local_hour"] = df_hr_raw_full["local_day"] + pd.to_timedelta(
            df_hr_raw_full["hour"], unit="h"
        )
        df_hr_raw_full = df_hr_raw_full.merge(
            df_hr_raw_hourly,
            on=["id", "local_day", "local_hour"],
            how="left",
        )
        df_hr_raw_full["HR_raw_records_in_hour"] = df_hr_raw_full[
            "HR_raw_records_in_hour"
        ].fillna(0)
        df_hr_raw_hourly_stats = (
            df_hr_raw_full.groupby(["id", "local_day"], observed=True)
            .agg(
                HR_raw_records_per_hour_mean=(
                    "HR_raw_records_in_hour",
                    "mean",
                ),
                HR_raw_records_per_hour_median=(
                    "HR_raw_records_in_hour",
                    "median",
                ),
                HR_raw_records_per_hour_std=("HR_raw_records_in_hour", "std"),
            )
            .reset_index()
        )
    else:
        df_hr_raw_hourly_stats = (
            df_hr_raw_hourly.groupby(["id", "local_day"], observed=True)
            .agg(
                HR_raw_records_per_hour_mean=(
                    "HR_raw_records_in_hour",
                    "mean",
                ),
                HR_raw_records_per_hour_median=(
                    "HR_raw_records_in_hour",
                    "median",
                ),
                HR_raw_records_per_hour_std=("HR_raw_records_in_hour", "std"),
            )
            .reset_index()
        )

    # Per-hour stats (second-level)
    if include_zero_hours:
        df_hr_seconds_days = df_avg[["id", "local_day"]].drop_duplicates()
        df_hr_seconds_days = df_hr_seconds_days.assign(_join_key=1)
        df_hr_seconds_hours = pd.DataFrame({"hour": range(24), "_join_key": 1})
        df_hr_seconds_full = df_hr_seconds_days.merge(
            df_hr_seconds_hours, on="_join_key"
        ).drop(columns=["_join_key"])
        df_hr_seconds_full["local_hour"] = df_hr_seconds_full[
            "local_day"
        ] + pd.to_timedelta(df_hr_seconds_full["hour"], unit="h")
        df_hr_seconds_full = df_hr_seconds_full.merge(
            df_hr_seconds_hourly,
            on=["id", "local_day", "local_hour"],
            how="left",
        )
        df_hr_seconds_full["HR_seconds_in_hour"] = df_hr_seconds_full[
            "HR_seconds_in_hour"
        ].fillna(0)
        df_hr_seconds_hourly_stats = (
            df_hr_seconds_full.groupby(["id", "local_day"], observed=True)
            .agg(
                HR_seconds_per_hour_mean=("HR_seconds_in_hour", "mean"),
                HR_seconds_per_hour_median=("HR_seconds_in_hour", "median"),
                HR_seconds_per_hour_std=("HR_seconds_in_hour", "std"),
            )
            .reset_index()
        )
    else:
        df_hr_seconds_hourly_stats = (
            df_hr_seconds_hourly.groupby(["id", "local_day"], observed=True)
            .agg(
                HR_seconds_per_hour_mean=("HR_seconds_in_hour", "mean"),
                HR_seconds_per_hour_median=("HR_seconds_in_hour", "median"),
                HR_seconds_per_hour_std=("HR_seconds_in_hour", "std"),
            )
            .reset_index()
        )

    # Merge coverage stats into daily
    df_hr_daily = df_hr_daily.merge(df_hr_raw_daily, on=["id", "local_day"], how="left")
    df_hr_daily = df_hr_daily.merge(
        df_hr_raw_hours_with_records, on=["id", "local_day"], how="left"
    )
    df_hr_daily = df_hr_daily.merge(
        df_hr_raw_hourly_stats, on=["id", "local_day"], how="left"
    )
    df_hr_daily = df_hr_daily.merge(
        df_hr_seconds_hours_with_data, on=["id", "local_day"], how="left"
    )
    df_hr_daily = df_hr_daily.merge(
        df_hr_seconds_hourly_stats, on=["id", "local_day"], how="left"
    )

    # coverage as the number of hours with records divided by 24
    df_hr_daily["HR_coverage"] = df_hr_daily["HR_raw_hours_with_records"] / 24

    return df_hr_daily


def aggregate_steps_daily(df_backup, cutoff_seconds=600, nighttime_hour=6):
    """
    Aggregate steps data to daily level.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with steps records
    cutoff_seconds : int
        Maximum session duration to include (default 600 = 10 minutes)
    nighttime_hour : int
        Hour boundary for nighttime (0 to threshold-1), default 6 means 00:00-05:59

    Returns
    -------
    pd.DataFrame
        Daily steps aggregates with columns: id, local_day, and step metrics

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | The local calendar day of the records (floored to midnight) | Grouping key |
    | **`for_id`** | External participant identifier | First available value in day |
    | **`steps_in_day`** | Total daily steps proxy from expanded minute-level series | Sum of steps across day |
    | **`SPM_max`** | Maximum minute-level step intensity in day | Max of daily steps per minute (SPM) |
    | **`SPM_count`** | Number of expanded minute records in day | Count of daily `SPM` |
    | **`SPM_mean`** | Mean minute-level step intensity in day | Mean of daily `SPM` |
    | **`SPM_std`** | Standard deviation of minute-level step intensity in day | Std of daily `SPM` |
    | **`SPM_skew`** | Skewness of minute-level step intensity in day | Skew of daily `SPM` |
    | **`SPM_kurtosis`** | Kurtosis of minute-level step intensity in day | Kurtosis of daily `SPM` |
    | **`SPM_25pct`** | 25th percentile of minute-level step intensity | Quantile of daily `SPM` |
    | **`SPM_50pct`** | 50th percentile (median) of minute-level step intensity | Quantile of daily `SPM` |
    | **`SPM_75pct`** | 75th percentile of minute-level step intensity | Quantile of daily `SPM` |
    | **`steps_night_sum`** | Total nighttime steps | Sum of `SPM` where local hour `< nighttime_hour` |
    | **`steps_night_mean`** | Mean nighttime steps intensity | Mean of `SPM` where local hour `< nighttime_hour` |
    | **`steps_hours_with_records`** | Number of unique local hours with any steps records | Count of hourly bins after minute expansion |
    | **`steps_per_hour`** | Mean steps per recorded hour | Mean of hourly `steps_inhour` |
    | **`SPM_max_avgbyhour`** | Average hourly max `SPM` across the day | Mean of hourly max `SPM` |
    | **`SPM_mean_avgbyhour`** | Average hourly mean `SPM` across the day | Mean of hourly mean `SPM` |
    | **`SPM_std_avgbyhour`** | Average hourly std `SPM` across the day | Mean of hourly std `SPM` |
    | **`SPM_skew_avgbyhour`** | Average hourly skew `SPM` across the day | Mean of hourly skew `SPM` |
    | **`SPM_kurtosis_avgbyhour`** | Average hourly kurtosis `SPM` across the day | Mean of hourly kurtosis `SPM` |
    | **`steps_coverage`** | Percentage of the day covered by steps recordings (`steps_hours_with_records / 24`) | Derived from hourly coverage |
    | **`steps_in_most_active_hour`** | Steps volume in the most active hour | `steps_inhour` at daily max hour |
    | **`most_active_hour`** | Clock hour (0-23) with highest steps volume | Hour of daily max `steps_inhour` |
    | **`max_spm_in_most_active_hour`** | Max minute-level intensity in most active hour | Hour-level `SPM_max_inhour` at daily max hour |
    | **`avg_spm_in_most_active_hour`** | Mean minute-level intensity in most active hour | Hour-level `SPM_mean_inhour` at daily max hour |
    | **`hour_reach_25pct_steps_cumsum`** | Clock hour where cumulative steps first reach 25% of day total | From hourly cumulative sum |
    | **`hour_reach_50pct_steps_cumsum`** | Clock hour where cumulative steps first reach 50% of day total | From hourly cumulative sum |
    | **`hour_reach_75pct_steps_cumsum`** | Clock hour where cumulative steps first reach 75% of day total | From hourly cumulative sum |
    """
    # Filter steps data
    df = df_backup[df_backup["modality"] == "Steps"].copy()
    df = df[
        [
            "id",
            "for_id",
            "timestamp_start",
            "timestamp_end",
            "float_value",
            "local_timestamp_start",
        ]
    ].copy()
    df = df.rename(columns={"float_value": "steps"})

    # Compute duration and apply cutoff
    df["start_end"] = (df["timestamp_end"] - df["timestamp_start"]).dt.total_seconds()
    df = df[df["start_end"] <= cutoff_seconds].copy()

    # Compute SPM (steps Per Minute)
    df["duration"] = (
        (df["timestamp_end"] - df["timestamp_start"])
        .dt.total_seconds()
        .fillna(0)
        .astype(int)
    )
    df["SPM"] = df["steps"] / (df["duration"] / 60)
    df["SPM"] = df["SPM"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Expand to 1-minute resolution
    df["timestamp_start_minute"] = df["timestamp_start"].dt.round("min")
    df["duration_minutes"] = np.maximum(1, (df["duration"] / 60).round(0).astype(int))

    df_expanded = df.loc[df.index.repeat(df["duration_minutes"])].copy()
    df_expanded["time_offset"] = df_expanded.groupby(level=0).cumcount()
    df_expanded["timestamp"] = df_expanded["timestamp_start_minute"] + pd.to_timedelta(
        df_expanded["time_offset"], unit="min"
    )
    df_expanded["local_timestamp"] = df_expanded["local_timestamp_start"].dt.round(
        "min"
    ) + pd.to_timedelta(df_expanded["time_offset"], unit="min")

    # Deduplicate overlapping entries via mean
    df_avg = (
        df_expanded.groupby(["id", "timestamp"], observed=True)
        .agg(
            for_id=("for_id", "first"),
            SPM=("SPM", "mean"),
            local_timestamp=("local_timestamp", "first"),
        )
        .reset_index()
    )

    # Add time features
    df_avg["local_day"] = df_avg["local_timestamp"].dt.floor("D")
    df_avg["local_hour"] = df_avg["local_timestamp"].dt.floor("h")
    df_avg["isnighttime"] = df_avg["local_timestamp"].dt.hour < nighttime_hour

    # Daily aggregation
    df_steps_daily = (
        df_avg.groupby(["id", "local_day"], observed=True)
        .agg(
            for_id=("for_id", "first"),
            steps_in_day=("SPM", "sum"),
            SPM_max=("SPM", "max"),
            SPM_count=("SPM", "count"),
            SPM_mean=("SPM", "mean"),
            SPM_std=("SPM", "std"),
            SPM_skew=("SPM", "skew"),
            SPM_kurtosis=("SPM", lambda x: x.kurtosis()),
            SPM_25pct=("SPM", lambda x: x.quantile(0.25)),
            SPM_50pct=("SPM", lambda x: x.quantile(0.50)),
            SPM_75pct=("SPM", lambda x: x.quantile(0.75)),
            steps_night_sum=(
                "SPM",
                lambda x: x[df_avg.loc[x.index, "isnighttime"]].sum(),
            ),
            steps_night_mean=(
                "SPM",
                lambda x: x[df_avg.loc[x.index, "isnighttime"]].mean(),
            ),
        )
        .reset_index()
    )
    df_steps_daily["steps_night_mean"] = df_steps_daily["steps_night_mean"].fillna(0)

    # Hourly aggregation (intermediate step)
    df_hourly = (
        df_avg.groupby(["id", "local_hour"], observed=True)
        .agg(
            local_day=("local_day", "first"),
            steps_inhour=("SPM", "sum"),
            SPM_max_inhour=("SPM", "max"),
            SPM_mean_inhour=("SPM", "mean"),
            SPM_std_inhour=("SPM", "std"),
            SPM_skew_inhour=("SPM", "skew"),
            SPM_kurtosis_inhour=("SPM", lambda x: x.kurtosis()),
        )
        .reset_index()
    )
    df_hourly["clock_hour"] = df_hourly["local_hour"].dt.hour

    # Merge hourly stats into daily
    hourly_agg = (
        df_hourly.groupby(["id", "local_day"], observed=True)
        .agg(
            steps_hours_with_records=("local_hour", "count"),
            steps_per_hour=("steps_inhour", "mean"),
            SPM_max_avgbyhour=("SPM_max_inhour", "mean"),
            SPM_mean_avgbyhour=("SPM_mean_inhour", "mean"),
            SPM_std_avgbyhour=("SPM_std_inhour", "mean"),
            SPM_skew_avgbyhour=("SPM_skew_inhour", "mean"),
            SPM_kurtosis_avgbyhour=("SPM_kurtosis_inhour", "mean"),
        )
        .reset_index()
    )

    df_steps_daily = df_steps_daily.merge(
        hourly_agg, on=["id", "local_day"], how="left"
    )

    # Coverage is the share of day-hours with at least one steps record.
    df_steps_daily["steps_coverage"] = df_steps_daily["steps_hours_with_records"] / 24

    # Most active hour
    most_active = df_hourly.loc[
        df_hourly.groupby(["id", "local_day"], observed=True)["steps_inhour"].idxmax()
    ][
        [
            "id",
            "local_day",
            "steps_inhour",
            "clock_hour",
            "SPM_max_inhour",
            "SPM_mean_inhour",
        ]
    ].rename(
        columns={
            "steps_inhour": "steps_in_most_active_hour",
            "clock_hour": "most_active_hour",
            "SPM_max_inhour": "max_spm_in_most_active_hour",
            "SPM_mean_inhour": "avg_spm_in_most_active_hour",
        }
    )
    df_steps_daily = df_steps_daily.merge(
        most_active, on=["id", "local_day"], how="left"
    )

    # Percentile hours (when 25%, 50%, 75% of daily steps are reached)
    df_hourly["steps_inhour_cumsum"] = df_hourly.groupby(
        ["id", "local_day"], observed=True
    )["steps_inhour"].cumsum()

    def percentile_hours(group, percentiles=(0.25, 0.5, 0.75)):
        total_steps = group["steps_inhour_cumsum"].iloc[-1]
        result = {}
        for p in percentiles:
            if total_steps > 0:
                idx = group["steps_inhour_cumsum"].searchsorted(p * total_steps)
                idx = min(idx, len(group) - 1)
                result[f"hour_reach_{int(p * 100)}pct_steps_cumsum"] = group.iloc[idx][
                    "local_hour"
                ].hour
            else:
                result[f"hour_reach_{int(p * 100)}pct_steps_cumsum"] = np.nan
        return pd.Series(result)

    pct_hours = (
        df_hourly.groupby(["id", "local_day"], observed=True)
        .apply(percentile_hours, include_groups=False)
        .reset_index()
    )
    df_steps_daily = df_steps_daily.merge(pct_hours, on=["id", "local_day"], how="left")

    return df_steps_daily


def aggregate_activity_daily(df_backup, exclude=None):
    """
    Aggregate activity type data to daily level in wide format.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with ActivityType records
    exclude : list
        Activity types to exclude (default ["SLEEP", "REST"])

    Returns
    -------
    pd.DataFrame
        Daily activity aggregates in wide format with prefixed columns per activity type

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | The local calendar day of the records (floored to midnight) | Grouping key |
    | **`ACTIVE_avg_session_duration`** | Mean ACTIVE session duration (minutes) | Mean of ACTIVE session durations |
    | **`BIKE_avg_session_duration`** | Mean BIKE session duration (minutes) | Mean of BIKE session durations |
    | **`RUN_avg_session_duration`** | Mean RUN session duration (minutes) | Mean of RUN session durations |
    | **`WALK_avg_session_duration`** | Mean WALK session duration (minutes) | Mean of WALK session durations |
    | **`ACTIVE_max_session_duration`** | Longest ACTIVE session duration (minutes) | Max of ACTIVE session durations |
    | **`BIKE_max_session_duration`** | Longest BIKE session duration (minutes) | Max of BIKE session durations |
    | **`RUN_max_session_duration`** | Longest RUN session duration (minutes) | Max of RUN session durations |
    | **`WALK_max_session_duration`** | Longest WALK session duration (minutes) | Max of WALK session durations |
    | **`ACTIVE_n_sessions`** | Number of ACTIVE sessions in day | Count of ACTIVE sessions |
    | **`BIKE_n_sessions`** | Number of BIKE sessions in day | Count of BIKE sessions |
    | **`RUN_n_sessions`** | Number of RUN sessions in day | Count of RUN sessions |
    | **`WALK_n_sessions`** | Number of WALK sessions in day | Count of WALK sessions |
    | **`ACTIVE_total_duration`** | Total ACTIVE minutes in day | Sum of ACTIVE session durations |
    | **`BIKE_total_duration`** | Total BIKE minutes in day | Sum of BIKE session durations |
    | **`RUN_total_duration`** | Total RUN minutes in day | Sum of RUN session durations |
    | **`WALK_total_duration`** | Total WALK minutes in day | Sum of WALK session durations |

    Notes:
    - The table above reflects the column set observed in your notebook output.
    - Actual output columns still depend on which activity classes are present after filtering.
    - Output is pivoted to wide format with zero fill for missing activity/day combinations among included classes.
    """
    if exclude is None:
        exclude = ["SLEEP", "REST"]

    # Filter ActivityType data
    df = df_backup[df_backup["modality"] == "ActivityType"].copy()
    df["activity_type_str"] = df["float_value"].map(MAP_ACTIVITYTYPE)
    df = df[~df["activity_type_str"].isin(exclude)].copy()

    # Add local day and duration
    df["local_day"] = df["local_timestamp_start"].dt.floor("D")
    df["duration"] = (
        df["timestamp_end"] - df["timestamp_start"]
    ).dt.total_seconds() / 60  # minutes

    # Aggregate by id, day, activity type
    df_agg = (
        df.groupby(["id", "local_day", "activity_type_str"], observed=True)
        .agg(
            total_duration=("duration", "sum"),
            n_sessions=("duration", "count"),
            avg_session_duration=("duration", "mean"),
            max_session_duration=("duration", "max"),
        )
        .reset_index()
    )

    # Pivot to wide format
    df_wide = df_agg.pivot_table(
        index=["id", "local_day"],
        columns="activity_type_str",
        values=[
            "total_duration",
            "n_sessions",
            "avg_session_duration",
            "max_session_duration",
        ],
        fill_value=0,
        observed=True,
    )

    # Flatten column names
    df_wide.columns = [f"{activity}_{metric}" for metric, activity in df_wide.columns]
    df_wide = df_wide.reset_index()

    return df_wide


def aggregate_elevation_daily(df_backup, cutoff_hours=4):
    """
    Aggregate elevation gain data to daily level.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with ElevationGain records
    cutoff_hours : int
        Maximum session duration in hours to include (default 4)

    Returns
    -------
    pd.DataFrame
        Daily elevation aggregates with columns: id, local_day, total_elevation_gain

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | The local calendar day of the records (floored to midnight) | Grouping key |
    | **`total_elevation_gain`** | Total daily elevation gain | Sum of `float_value` for `ElevationGain` records after cutoff filter |

    Notes:
    - Records with session duration above `cutoff_hours` are excluded before aggregation.
    """
    df = df_backup[df_backup["modality"] == "ElevationGain"].copy()
    df["local_day"] = df["local_timestamp_start"].dt.floor("D")
    df["start_end"] = (df["timestamp_end"] - df["timestamp_start"]).dt.total_seconds()

    # Apply cutoff
    df = df[df["start_end"] <= cutoff_hours * 3600].copy()

    df_elevation_daily = (
        df.groupby(["id", "local_day"], observed=True)
        .agg(total_elevation_gain=("float_value", "sum"))
        .reset_index()
    )

    return df_elevation_daily


def aggregate_floors_daily(df_backup):
    """
    Aggregate floors climbed data to daily level.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with FloorsClimbed records

    Returns
    -------
    pd.DataFrame
        Daily floors aggregates with columns: id, local_day, total_floors_climbed

    Returned metrics include:
    | Column Name | Description | Source / Calculation |
    | :--- | :--- | :--- |
    | **`id`** | Participant identifier | Grouping key |
    | **`local_day`** | The local calendar day of the records (floored to midnight) | Grouping key |
    | **`total_floors_climbed`** | Total floors climbed in day | Sum of `float_value` for `FloorsClimbed` records |
    """
    df = df_backup[df_backup["modality"] == "FloorsClimbed"].copy()
    df["local_day"] = df["local_timestamp_start"].dt.floor("D")

    df_floors_daily = (
        df.groupby(["id", "local_day"], observed=True)
        .agg(total_floors_climbed=("float_value", "sum"))
        .reset_index()
    )

    return df_floors_daily


def aggregate_all_passive(df_backup, **kwargs):
    """
    Aggregate all passive data modalities and merge into a single DataFrame.

    Parameters
    ----------
    df_backup : pd.DataFrame
        Raw backup data with all passive records
    **kwargs : dict
        Additional keyword arguments passed to individual aggregation functions:
        - sleep_max_gap_seconds: for aggregate_sleep_daily
        - hr_zone_thresholds: for aggregate_hr_daily
        - steps_cutoff_seconds: for aggregate_steps_daily
        - steps_nighttime_hour: for aggregate_steps_daily
        - activity_exclude: for aggregate_activity_daily
        - elevation_cutoff_hours: for aggregate_elevation_daily

    Returns
    -------
    pd.DataFrame
        Merged daily aggregates from all modalities on ["id", "local_day"]
    """
    logging.info("Aggregating sleep data...")
    df_sleep = aggregate_sleep_daily(
        df_backup, max_gap_seconds=kwargs.get("sleep_max_gap_seconds", 5400)
    )

    logging.info("Aggregating heart rate data...")
    df_hr = aggregate_hr_daily(
        df_backup, zone_thresholds=kwargs.get("hr_zone_thresholds", (60, 100))
    )

    logging.info("Aggregating steps data...")
    df_steps = aggregate_steps_daily(
        df_backup,
        cutoff_seconds=kwargs.get("steps_cutoff_seconds", 600),
        nighttime_hour=kwargs.get("steps_nighttime_hour", 6),
    )

    logging.info("Aggregating activity data...")
    df_activity = aggregate_activity_daily(
        df_backup, exclude=kwargs.get("activity_exclude", ["SLEEP", "REST"])
    )

    logging.info("Aggregating elevation data...")
    df_elevation = aggregate_elevation_daily(
        df_backup, cutoff_hours=kwargs.get("elevation_cutoff_hours", 4)
    )

    logging.info("Aggregating floors data...")
    df_floors = aggregate_floors_daily(df_backup)

    # Merge all DataFrames on ["id", "local_day"]
    logging.info("Merging all modalities...")
    df_all = df_sleep.merge(df_hr, on=["id", "local_day"], how="outer")
    df_all = df_all.merge(df_steps, on=["id", "local_day"], how="outer")
    df_all = df_all.merge(df_activity, on=["id", "local_day"], how="outer")
    df_all = df_all.merge(df_elevation, on=["id", "local_day"], how="outer")
    df_all = df_all.merge(df_floors, on=["id", "local_day"], how="outer")

    logging.info(f"Final shape: {df_all.shape}")

    return df_all
