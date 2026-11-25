import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer


import statistics
import math
from scipy import stats
from heapq import nlargest
from datetime import datetime, timedelta, time
from math import radians, cos, sin, asin, sqrt, log
import string
import pickle
import pingouin as pg

#import hdbscan
from scipy.stats import mode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


# Adjusted from Mueller et al., (2021); https://doi.org/ 10.1038/s41598-021-93087-x

# Haversine formula to calculate distance between two lat/lon points in meters
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # Radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    meters = R * c  # Output distance in meters
    return meters

def calculate_sample_weights(
    df,
    groupby="id",
    max_gap=12 * 3600, # maximum 12 hours at one place
    far_threshold=1000, # 1km is considered far
    far_weight=60, # assume one minute for large distances
):
    """
    Calculate sample weights based on time and distance to the next sample.
    """
    df = df.sort_values(by=[groupby, "timestamp_start"])
    gby = df.groupby(groupby, observed=True)

    # Time to next sample
    time_to_next = (-gby["timestamp_start"].diff(-1)).dt.total_seconds()
    time_to_next_clipped = time_to_next.clip(upper=max_gap).fillna(0)

    # Distance to next sample (Haversine)
    lat_next = gby["Latitude"].shift(-1)
    lon_next = gby["Longitude"].shift(-1)

    dist_to_next = haversine(df["Longitude"], df["Latitude"], lon_next, lat_next)

    is_far = dist_to_next > far_threshold

    weights = np.where(
        is_far,
        time_to_next_clipped.clip(upper=far_weight),
        time_to_next_clipped
    )

    return pd.Series(weights, index=df.index)

def compute_row_night_time(row, start_night_hour=23, end_night_hour=6):
    """
    Calculate the duration of an interval that falls within a night window.
    (Row-wise reference implementation; find_home_cluster_v2 uses a vectorized version.)
    """
    # Ensure inputs are parsed as pandas Timestamps
    start = pd.to_datetime(row['local_timestamp_start'])
    # assumed_local_endtime should be pre-calculated
    if 'assumed_local_endtime' in row:
        end = pd.to_datetime(row['assumed_local_endtime'])
    else:
        # Fallback if assumed_local_endtime is not there
        return pd.Timedelta(0)

    # Safety check for missing or invalid intervals
    if pd.isna(start) or pd.isna(end) or start >= end:
        return pd.Timedelta(0)

    total_night_duration = pd.Timedelta(0)

    # Look back 1 day prior to start date to capture stays
    # that occur during the early morning window (00:00 - 06:00)
    current_date = start.date() - pd.Timedelta(days=1)
    end_date = end.date()

    while current_date <= end_date:
        # Define the night window for the current iteration
        night_start = pd.Timestamp.combine(current_date, time(start_night_hour, 0))
        night_end = pd.Timestamp.combine(current_date + pd.Timedelta(days=1), time(end_night_hour, 0))

        # Calculate the intersection of the stay interval and the night window
        overlap_start = max(start, night_start)
        overlap_end = min(end, night_end)

        if overlap_start < overlap_end:
            total_night_duration += (overlap_end - overlap_start)

        current_date += pd.Timedelta(days=1)

    return total_night_duration

def apply_clustering(df, epsilon, min_samples):
    def db2(x):
        clustering_model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
        cluster_labels = clustering_model.fit_predict(x[['Longitude', 'Latitude']].apply(np.radians))
        return pd.DataFrame({'cluster_100m': cluster_labels})

    # Group by 'id' and apply clustering function
    geodata_cluster_df = df.groupby('id').apply(lambda x: db2(x)).reset_index()
    return geodata_cluster_df

def adaptive_kmeans(x, Dkmeans=500, max_k=50):
    """
    Perform K-Means clustering with an adaptive number of clusters.

    Parameters:
    x (pd.DataFrame): The data to cluster, with 'Longitude' and 'Latitude' columns.
    Dkmeans (float): The maximum allowed distance from any point to its cluster center (in meters).
    max_k (int): The maximum number of clusters to test.

    Returns:
    pd.Series: Cluster labels for each point in the data.
    """
    X = x[['Longitude', 'Latitude']].apply(np.radians)

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Convert centers back to degrees for distance calculation
        centers_degrees = np.degrees(centers)

        # Calculate the maximum distance of any point to its cluster center
        max_distance = 0
        for i in range(k):
            cluster_points = x[labels == i]
            if not cluster_points.empty:
                distances = cluster_points.apply(
                    lambda row: haversine(row['Longitude'], row['Latitude'], centers_degrees[i][0], centers_degrees[i][1]), axis=1
                )
                if distances.max() > max_distance:
                    max_distance = distances.max()

        if max_distance <= Dkmeans:
            print(f"Number of clusters selected: {k}")
            return pd.Series(labels, index=x.index)

    print(f"Maximum number of clusters reached: {max_k}")
    return pd.Series(labels, index=x.index)


def identify_home(df):
    df['home'] = df.groupby('id')['clusterID'].transform(lambda x: statistics.mode(x))
    return df

def isclose(loc1, loc2, threshold=30):
    """
    Determine if two locations are within a given distance of each other.

    Parameters:
    - loc1, loc2: Tuples representing (latitude, longitude) of two locations.
    - threshold: Distance in meters below which locations are considered "close". Default is 30 meters.

    Returns:
    - True if the distance between loc1 and loc2 is less than the threshold; otherwise, False.
    """
    return haversine(loc1[1], loc1[0], loc2[1], loc2[0]) < threshold


def cal_entropy(sig_locs):
    """
    Calculate entropy and normalized entropy.

    NOTE: keys on raw (lat, lon), not on clusters. Used only by the legacy
    calculate_metrics(); use calculate_metrics_fast() for cluster-based entropy.

    :param sig_locs: tuple of tuple of lat, lon, start, end, dur
    :return: entropy, normalized_entropy
    """
    if len(sig_locs) <= 1:
        return np.nan, np.nan
    # Calculate the total duration for each cluster
    clusters = {}
    for lat, lon, _, _, dur, _ in sig_locs:
        if (lat, lon) in clusters:
            clusters[(lat, lon)] += dur
        else:
            clusters[(lat, lon)] = dur
    values = np.array(list(clusters.values())).astype(np.float64)
    probs = values / values.sum()
    probs[probs == 0] = 1e-10  # Avoid log(0) by setting zero probabilities to a very small positive number
    ent = -probs.dot(np.log(probs))

    # normalized entropy ranges from 0 to 1. If N clusters are defined, we normalize the max entropy
    # (when every case is distributed equally as 1/N)
    norm_ent = ent / (log(len(clusters)) + 1e-09)
    return ent, norm_ent

def calculate_metrics(df, group_by=['id']):
    """
    LEGACY, slow row-wise version. Kept for reference only; use calculate_metrics_fast().
    Known issue: entropy is computed over raw coordinates, not clusters.

    Calculate raw entropy, entropy, normalized entropy, total distance traveled, percentage of time at home,
    number of unique clusters visited, number of non-unique clusters visited for each group defined by `group_by`.
    """
    results = []

    for group_keys, group in df.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_results = {key: val for key, val in zip(group_by, group_keys)}

        group = group.sort_values(by='timestamp_start')
        group['time_spent'] = group['timestamp_start'].diff().shift(-1).fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

        valid_labels = group['clusterID'][group['clusterID'] != -1]
        if len(valid_labels) == 0:
            group_results.update({
                'raw_entropy': np.nan,
                'entropy': np.nan,
                'normalized_entropy': np.nan,
                'total_distance': 0,
                'percentage_time_at_home': np.nan,
                'num_unique_clusters': 0,
                'num_total_clusters': 0
            })
            results.append(group_results)
            continue

        # Calculate clusters entropy
        sig_locs = [(row['Latitude'], row['Longitude'], row['timestamp_start'], row['timestamp_start'] + pd.Timedelta(seconds=row['time_spent']), row['time_spent'], row['clusterID']) for index, row in group.iterrows()]
        entropy, normalized_entropy = cal_entropy(sig_locs)

        group_results['entropy'] = entropy
        group_results['normalized_entropy'] = normalized_entropy

        # Calculate the total distance using the Haversine formula
        latitudes = group['Latitude'].to_numpy()
        longitudes = group['Longitude'].to_numpy()
        total_distance = sum(haversine(lon1, lat1, lon2, lat2) for lon1, lat1, lon2, lat2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:]))

        # Calculate percentage of time spent at home if home is not NaN
        total_duration = group['time_spent'].sum()
        if pd.notna(group['home'].iloc[0]):
            time_at_home = group.loc[group['at_home'] == 1, 'time_spent'].sum()
            percentage_time_at_home = (time_at_home / total_duration) * 100 if total_duration > 0 else 0
        else:
            percentage_time_at_home = np.nan

        # Number of unique clusters
        num_unique_clusters = len(set(valid_labels))

        # Number of total clusters
        num_total_clusters = len(group['clusterID'].unique())

        # Raw location entropy based on non-clustered GPS coordinates
        lat_bins = pd.cut(group['Latitude'], bins=10)
        lon_bins = pd.cut(group['Longitude'], bins=10)
        raw_entropy = 0
        bin_counts = group.groupby([lat_bins, lon_bins], observed=True).size()
        total_points = bin_counts.sum()

        for count in bin_counts:
            p_bin = count / total_points
            if p_bin > 0:
                raw_entropy -= p_bin * np.log(p_bin)

        group_results.update({
            'total_distance': total_distance,
            'percentage_time_at_home': percentage_time_at_home,
            'num_unique_clusters': num_unique_clusters,
            'num_total_clusters': num_total_clusters,
            'raw_entropy': raw_entropy
        })

        results.append(group_results)

    return pd.DataFrame(results)


def calculate_metrics_fast(df, group_by=("id", "local_day"), max_speed_kmh=1000, min_jump_m=1000):
    """Vectorized daily mobility metrics.

    - time_spent per point = time to the next point WITHIN the same group (day);
      percentage_time_at_home therefore refers to the active period between the
      first and last GPS event of the day.
    - entropy / normalized_entropy: time-weighted over DBSCAN clusters (noise excluded).
    - at_home == 1 marks points in any home cluster (see HomeClusterExtractor).
    - total_distance in meters. Steps that would require > max_speed_kmh (and are longer than
      min_jump_m) are physically impossible and NOT counted; their number per day is reported
      in n_impossible_jumps.
    """
    keys = list(group_by)
    d = df[keys + ["timestamp_start", "Latitude", "Longitude", "cluster", "home", "at_home"]].copy()
    d = d.sort_values(keys + ["timestamp_start"])
    g = d.groupby(keys, observed=True, sort=False)

    # time to next point and distance to next point, within group
    d["time_spent"] = (-g["timestamp_start"].diff(-1)).dt.total_seconds().fillna(0.0)
    d["step_dist"] = haversine(
        d["Longitude"], d["Latitude"], g["Longitude"].shift(-1), g["Latitude"].shift(-1)
    ).fillna(0.0)
    # physically impossible steps (teleports) are not counted as distance
    step_speed_kmh = (d["step_dist"] / 1000) / (d["time_spent"].clip(lower=1) / 3600)
    d["impossible"] = (d["step_dist"] > min_jump_m) & (step_speed_kmh > max_speed_kmh)
    d.loc[d["impossible"], "step_dist"] = 0.0
    d["valid"] = d["cluster"].ne(-1)
    d["home_time"] = d["time_spent"].where(d["at_home"].eq(1), 0.0)

    # raw-coordinate entropy: 10x10 bins of lat/lon per group
    # (only meaningful after normalize_shifted_coords, which restores consistent geometry)
    g = d.groupby(keys, observed=True, sort=False)
    for col, b in (("Latitude", "latb"), ("Longitude", "lonb")):
        lo, hi = g[col].transform("min"), g[col].transform("max")
        rng = (hi - lo).where(hi > lo)
        d[b] = np.floor((d[col] - lo) / rng * 10).fillna(0).clip(0, 9).astype(np.int8)

    g = d.groupby(keys, observed=True, sort=False)
    out = g.agg(
        total_distance=("step_dist", "sum"),
        n_impossible_jumps=("impossible", "sum"),
        total_time=("time_spent", "sum"),
        home_time=("home_time", "sum"),
        n_valid=("valid", "sum"),
        num_total_clusters=("cluster", "nunique"),
        has_home=("home", lambda s: s.notna().iloc[0]),
    )

    # cluster-based entropy (time-weighted, noise excluded)
    v = d[d["valid"]]
    ct = v.groupby(keys + ["cluster"], observed=True)["time_spent"].sum()
    p = ct / ct.groupby(level=keys).transform("sum")
    ent = (-(p * np.log(p.where(p > 0)))).fillna(0).groupby(level=keys).sum()
    n_cl = ct.groupby(level=keys).size()
    out["num_unique_clusters"] = n_cl.reindex(out.index).fillna(0).astype(int)
    out["entropy"] = ent.reindex(out.index).where(out["num_unique_clusters"] > 1)
    out["normalized_entropy"] = out["entropy"] / np.log(out["num_unique_clusters"].where(lambda x: x > 1))

    # raw entropy
    bc = d.groupby(keys + ["latb", "lonb"], observed=True).size()
    pb = bc / bc.groupby(level=keys).transform("sum")
    out["raw_entropy"] = (-(pb * np.log(pb))).groupby(level=keys).sum().reindex(out.index)

    # % time at home (any home cluster), relative to the day's active period
    pct = np.where(out["total_time"] > 0, out["home_time"] / out["total_time"] * 100, 0.0)
    out["percentage_time_at_home"] = np.where(out["has_home"], pct, np.nan)

    # length of the day's active period (first to last GPS point), in hours
    out["active_period_hours"] = out["total_time"] / 3600

    # groups with no clustered points
    none = out["n_valid"] == 0
    out.loc[none, ["entropy", "normalized_entropy", "percentage_time_at_home", "raw_entropy"]] = np.nan
    out.loc[none, "total_distance"] = 0.0

    cols = ["raw_entropy", "entropy", "normalized_entropy", "total_distance",
            "percentage_time_at_home", "num_unique_clusters", "num_total_clusters",
            "active_period_hours", "n_impossible_jumps"]
    return out[cols].reset_index()

def calculate_transition_time(df, group_by=['id']):
    """
    Calculate the transition time percentage for the raw, unfiltered GPS data for each group defined by `group_by`.

    Parameters:
    df (pd.DataFrame): DataFrame with 'id', 'Latitude', 'Longitude', 'timestamp_start', and optional 'day' columns.
    group_by (list): List of column names to group by. Default is ['id'].

    Returns:
    pd.DataFrame: DataFrame with group levels and transition time percentage.
    """
    results = []

    for group_keys, group in df.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_results = {key: val for key, val in zip(group_by, group_keys)}

        # Compute the transition time
        if len(group) > 1:
            group = group.sort_values(by='timestamp_start')
            latitudes = group['Latitude'].to_numpy()
            longitudes = group['Longitude'].to_numpy()
            times = group['timestamp_start'].to_numpy()

            distances = np.array([
                haversine(lon1, lat1, lon2, lat2)
                for lon1, lat1, lon2, lat2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:])
            ])

            time_deltas = np.diff(times) / np.timedelta64(1, 's')  # Convert time deltas to seconds
            speeds = distances / time_deltas
            moving_time = np.sum(time_deltas[speeds >= 1.4])
            total_time = np.sum(time_deltas)

            transition_time_percentage = (moving_time / total_time) * 100 if total_time > 0 else 0
        else:
            transition_time_percentage = 0.0

        group_results['transition_time'] = transition_time_percentage
        results.append(group_results)

    return pd.DataFrame(results)


######################################################################

# Undo the app's coordinate anonymisation

######################################################################
# The app adds a random offset to latitude (wrapping at +-90, i.e. modulo 180)
# and to longitude (modulo 360). The offset is constant within a "key" period
# and changes e.g. on reinstall / device change. A latitude offset distorts
# east-west distances by cos(lat_obfuscated) / cos(lat_true), so all distance-based
# steps (DBSCAN eps, speeds, total_distance) are only meaningful after undoing it.

def _wrap(x, period):
    """Wrap values/differences into [-period/2, period/2)."""
    return (x + period / 2) % period - period / 2


def _shift_dist_km(lat1, lon1, lat2, lon2, lon_scale):
    """Approximate true distance (km) between two points in lat/lon-SHIFTED coordinates.
    Latitude differences wrap with period 180, longitude with period 360.
    lon_scale = cos(true latitude) of the study region."""
    dlat = _wrap(lat2 - lat1, 180)
    dlon = _wrap(lon2 - lon1, 360)
    return 111.2 * np.sqrt(dlat ** 2 + (dlon * lon_scale) ** 2)


def _robust_center(df, cell_deg=0.5, radius_deg=2.0):
    """Robust centre per unit 'id' in (wrapped) lat/lon-shifted coordinates.
    Finds the densest cell_deg x cell_deg cell (most points, typically home), then averages
    all points within radius_deg of it (wrap-aware). Returns (c_lat, c_lon) Series by id."""
    cell_lat = np.floor(df["Latitude"] / cell_deg) * cell_deg + cell_deg / 2
    cell_lon = np.floor(df["Longitude"] / cell_deg) * cell_deg + cell_deg / 2
    counts = pd.DataFrame({"id": df["id"], "la": cell_lat, "lo": cell_lon}).groupby(["id", "la", "lo"]).size()
    top = counts.loc[counts.groupby(level=0).idxmax()].reset_index().set_index("id")

    m_lat, m_lon = df["id"].map(top["la"]), df["id"].map(top["lo"])
    dlat = _wrap(df["Latitude"] - m_lat, 180)
    dlon = _wrap(df["Longitude"] - m_lon, 360)
    near = (dlat.abs() <= radius_deg) & (dlon.abs() <= radius_deg)
    off = pd.DataFrame({"id": df["id"], "dlat": dlat.where(near), "dlon": dlon.where(near)})
    mean_off = off.groupby("id")[["dlat", "dlon"]].mean()
    c_lat = _wrap(top["la"] + mean_off["dlat"], 180)
    c_lon = _wrap(top["lo"] + mean_off["dlon"], 360)
    return c_lat, c_lon


def normalize_shifted_coords(df, ref_lat=52.52, ref_lon=13.40, jump_km=1000, max_trip_days=30,
                             max_speed_kmh=1000):
    """Undo additive lat/lon offset anonymisation.

    1. Split each participant into key units:
       - a jump > jump_km (wrap-aware distance) between consecutive points starts a new segment;
       - segments whose centre is within jump_km of an earlier key's centre get that key
         (handles A -> B -> A alternation);
       - short excursions (<= max_trip_days) between two segments of the same key are real
         trips and get that key too, but ONLY if the jumps into and out of the excursion are
         physically feasible (<= max_speed_kmh). Otherwise the excursion is a stretch recorded
         under another key (e.g. two devices alternating) and stays a separate key unit.
    2. Per key unit, re-centre coordinates on (ref_lat, ref_lon). Relative geometry is
       restored (scale error only from the difference between true latitude and ref_lat);
       absolute positions remain unknown.

    Adds 'id_orig', 'rot_key'; replaces 'id' with '<id>_k<key>' and overwrites
    Latitude/Longitude with the re-centred coordinates.
    """
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df = df.sort_values(["id", "timestamp_start"]).reset_index(drop=True)
    lon_scale = np.cos(np.radians(ref_lat))

    # 1) segments split at large (wrap-aware) jumps
    g = df.groupby("id", sort=False)
    step = _shift_dist_km(g["Latitude"].shift(), g["Longitude"].shift(),
                          df["Latitude"], df["Longitude"], lon_scale)
    dt_h = (df["timestamp_start"] - g["timestamp_start"].shift()).dt.total_seconds() / 3600
    speed_kmh = step / dt_h.clip(lower=1 / 3600)
    df["seg"] = (step > jump_km).astype(int).groupby(df["id"]).cumsum()

    # circular segment centres (latitude has period 180 -> use doubled angle)
    a_lat, a_lon = np.radians(df["Latitude"] * 2), np.radians(df["Longitude"])
    comp = pd.DataFrame({"id": df["id"], "seg": df["seg"], "t": df["timestamp_start"],
                         "v": speed_kmh,
                         "slat": np.sin(a_lat), "clat": np.cos(a_lat),
                         "slon": np.sin(a_lon), "clon": np.cos(a_lon)})
    segs = comp.groupby(["id", "seg"]).agg(slat=("slat", "mean"), clat=("clat", "mean"),
                                           slon=("slon", "mean"), clon=("clon", "mean"),
                                           t0=("t", "min"), t1=("t", "max"),
                                           entry_kmh=("v", "first"))   # speed of the jump into the segment
    segs["lat"] = np.degrees(np.arctan2(segs["slat"], segs["clat"])) / 2
    segs["lon"] = np.degrees(np.arctan2(segs["slon"], segs["clon"]))
    del comp

    # 2) match segments to keys, merge back real trips
    max_gap = pd.Timedelta(days=max_trip_days)
    rows, n_trip_segs, n_rejected = [], 0, 0
    for pid, sc in segs.groupby(level=0):
        sc = sc.droplevel(0).sort_index()
        centers, keys = [], []
        for la, lo in zip(sc["lat"], sc["lon"]):
            k = next((i for i, (cla, clo) in enumerate(centers)
                      if _shift_dist_km(cla, clo, la, lo, lon_scale) < jump_km), None)
            if k is None:
                centers.append((la, lo))
                k = len(centers) - 1
            keys.append(k)
        keys = np.array(keys)

        t0, t1, entry = list(sc["t0"]), list(sc["t1"]), list(sc["entry_kmh"])
        n = len(keys)
        for i in range(n):
            for j in range(i + 2, n):
                if keys[j] == keys[i]:
                    if t0[j] - t1[i] <= max_gap and np.any(keys[i + 1:j] != keys[i]):
                        # real trip only if getting there AND back was physically possible
                        feasible = (entry[i + 1] <= max_speed_kmh) and (entry[j] <= max_speed_kmh)
                        if feasible:
                            n_trip_segs += int(np.sum(keys[i + 1:j] != keys[i]))
                            keys[i + 1:j] = keys[i]
                        else:
                            n_rejected += 1
                    break
        keys = pd.factorize(keys)[0]           # renumber 0, 1, 2, ...
        rows += [(pid, seg, k) for seg, k in zip(sc.index, keys)]

    keymap = pd.DataFrame(rows, columns=["id", "seg", "rot_key"])
    df = df.merge(keymap, on=["id", "seg"], how="left")
    df["id_orig"] = df["id"]
    df["id"] = df["id_orig"] + "_k" + df["rot_key"].astype(str)

    # 3) re-centre each key unit on the reference point, using a ROBUST centre
    #    (densest region, i.e. where the person spends most time) so that far-away
    #    points (trips, garbage, stray points of another key) cannot shift it.
    c_lat, c_lon = _robust_center(df)
    df["lat_obf_center"] = df["id"].map(c_lat)   # anonymised latitude of the unit (for QC only)
    df["Latitude"] = ref_lat + _wrap(df["Latitude"] - df["lat_obf_center"], 180)
    df["Longitude"] = ref_lon + _wrap(df["Longitude"] - df["id"].map(c_lon), 360)

    # points that still end up at an impossible latitude cannot be real
    invalid = df["Latitude"].abs() > 90
    if invalid.any():
        logging.info(f"Shift normalisation: dropped {int(invalid.sum())} points with invalid "
                     f"latitude after re-centring")
        df = df[~invalid].reset_index(drop=True)

    n_multi = (df.groupby("id_orig")["rot_key"].nunique() > 1).sum()
    logging.info(f"Shift normalisation: {n_trip_segs} trip segments merged back, "
                 f"{n_rejected} excursions kept separate (infeasible speed = other key); "
                 f"{n_multi} participants have >1 key; {df['id'].nunique()} key units; "
                 f"coordinates re-centred on ({ref_lat}, {ref_lon})")
    return df.drop(columns=["seg"])


def remove_impossible_excursions(df, max_speed_kmh=1000, min_jump_km=1.0, max_excursion_hours=24,
                                 max_iter=3):
    """Remove GPS glitches: stretches of points that are entered AND left with a
    physically impossible jump (> max_speed_kmh over > min_jump_km), e.g. single
    garbage fixes thousands of km away. Real trips (incl. flights) have feasible speeds;
    key changes are entered but never left, so they are not affected. Only excursions
    lasting <= max_excursion_hours are removed (glitches are short; real stays are not).
    Run AFTER normalize_shifted_coords (needs consistent geometry), per unit 'id'."""
    df = df.sort_values(["id", "timestamp_start"]).reset_index(drop=True)
    n_total = 0
    for _ in range(max_iter):
        g = df.groupby("id", sort=False)
        dist_km = haversine(g["Longitude"].shift(), g["Latitude"].shift(),
                            df["Longitude"], df["Latitude"]) / 1000
        dt_h = (df["timestamp_start"] - g["timestamp_start"].shift()).dt.total_seconds() / 3600
        speed = dist_km / dt_h.clip(lower=1 / 3600)
        imp = ((dist_km > min_jump_km) & (speed > max_speed_kmh)).fillna(False)

        seg = imp.astype(int).groupby(df["id"]).cumsum()
        info = pd.DataFrame({"id": df["id"], "seg": seg, "imp": imp, "t": df["timestamp_start"]})
        s = info.groupby(["id", "seg"], sort=False).agg(entry=("imp", "first"),
                                                         t0=("t", "min"), t1=("t", "max"))
        s["exit"] = s.groupby(level=0)["entry"].shift(-1).fillna(False).astype(bool)
        glitch = s["entry"] & s["exit"] & ((s["t1"] - s["t0"]) <= pd.Timedelta(hours=max_excursion_hours))
        bad = s.index[glitch]
        if len(bad) == 0:
            break
        drop = pd.MultiIndex.from_arrays([df["id"], seg]).isin(bad)
        n_total += int(drop.sum())
        df = df[~drop].reset_index(drop=True)

    logging.info(f"Removed {n_total} points in impossible excursions "
                 f"(> {max_speed_kmh} km/h in and out)")
    return df


def normalize_data(data):
    """
    Normalize skewed data using a log transformation.

    Parameters:
    data (pd.Series): The data to transform.

    Returns:
    pd.Series: The normalized data.
    """
    # Add a small constant to ensure all values are positive
    shift_value = abs(data.min()) + 1 if data.min() <= 0 else 0
    normalized_data = np.log(data + shift_value)

    return normalized_data

def calculate_retest_reliability(geodata_cluster_merged, features):
    # Initialize a dictionary to store the correlation results
    correlations = {}

    # Calculate the correlation for each feature
    for feature in features:
        feature_first = feature + '_first'
        feature_second = feature + '_second'

        if feature_first in geodata_cluster_merged.columns and feature_second in geodata_cluster_merged.columns:
            # Drop rows with NaN values in the relevant columns
            clean_data = geodata_cluster_merged[[feature_first, feature_second]].dropna()

            if not clean_data.empty:
                # Normalize the data if it is skewed
                clean_data[feature_first] = normalize_data(clean_data[feature_first])
                clean_data[feature_second] = normalize_data(clean_data[feature_second])

                correlation = clean_data[feature_first].corr(clean_data[feature_second])
                correlations[feature] = correlation
            else:
                correlations[feature] = np.nan
        else:
            print(f"Feature columns for {feature} are missing in the dataframe.")

    return correlations


def calculate_intraclass_coefficient(geodata_cluster_merged, features):
    # Initialize a dictionary to store the ICC results
    iccs = {}

    # Calculate the ICC for each feature
    for feature in features:
        feature_first = feature + '_first'
        feature_second = feature + '_second'

        if feature_first in geodata_cluster_merged.columns and feature_second in geodata_cluster_merged.columns:
            # Drop rows with NaN values in the relevant columns
            clean_data = geodata_cluster_merged[[feature_first, feature_second]].dropna()

            if len(clean_data) >= 5:
                # Print the clean data for debugging
                print(f"Clean data for feature {feature}:")
                print(clean_data.head())

                # Reshape data for ICC calculation
                long_data = pd.DataFrame({
                    'subject': np.repeat(clean_data.index, 2),
                    'measurement': pd.concat([clean_data[feature_first], clean_data[feature_second]]).values,
                    'time': ['first'] * len(clean_data) + ['second'] * len(clean_data)
                })

                # Print the reshaped data for debugging
                print(f"Long data for feature {feature}:")
                print(long_data.head())

                # Check if data is balanced and has at least 5 subjects with both time points
                count_per_subject = long_data.groupby(['subject', 'time']).size().unstack(fill_value=0)
                if (count_per_subject.min(axis=1) >= 1).sum() >= 5:
                    try:
                        # Calculate ICC(2,1) for single measurements
                        icc = pg.intraclass_corr(data=long_data, targets='subject', raters='time', ratings='measurement', nan_policy='omit')
                        icc_value = icc[icc['Type'] == 'ICC2,1']['ICC'].values[0]
                        iccs[feature] = icc_value
                    except Exception as e:
                        print(f"Error calculating ICC for feature {feature}: {e}")
                        iccs[feature] = np.nan
                else:
                    print(f"Data for feature {feature} is unbalanced or insufficient data points.")
                    iccs[feature] = np.nan
            else:
                print(f"Not enough non-missing values for feature {feature}.")
                iccs[feature] = np.nan
        else:
            print(f"Feature columns for {feature} are missing in the dataframe.")

    return iccs

######################################################################

# Home Cluster Extractor Class from Short-term Prediction Paper

######################################################################
class HomeClusterExtractor:
    def __init__(
        self, df, speed_limit, max_distance, epsilon, min_samples, min_nights_obs, min_f_home,
        clustering_method='dbscan', normalize_min_samples=False, min_data_points=10, n_jobs=1,
        use_weights=False, home_v2=False, start_night_hour=23, end_night_hour=6,
        min_f_secondary_home=0.15, home_merge_m=500, dedup_grid_m=10.0,
    ):
        # TODO move to one correct version and remove the use_weights and home_v2 parameters
        self.df = df.copy()
        # Parameter validation
        if speed_limit <= 0:
            raise ValueError("speed_limit must be positive.")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if min_nights_obs < 1:
            raise ValueError("min_nights_obs must be at least 1.")
        if not 0 <= min_f_home <= 1:
            raise ValueError("min_f_home must be between 0 and 1.")
        if clustering_method != 'dbscan':
            raise ValueError("clustering_method must be 'dbscan'.")
        if min_data_points < 1:
            raise ValueError("min_data_points must be at least 1.")

        self.speed_limit = speed_limit
        self.max_distance = max_distance
        self.epsilon = epsilon
        # NOTE: with use_weights=True, sample weights are seconds, so min_samples is a
        # minimum cumulative dwell time in seconds within epsilon (e.g. 600 = 10 min).
        self.min_samples = min_samples
        self.min_nights_obs = min_nights_obs
        self.min_f_home = min_f_home
        self.clustering_method = clustering_method
        self.normalize_min_samples = normalize_min_samples
        self.min_data_points = min_data_points  # Minimum data points threshold
        self.n_jobs = n_jobs  # For parallel neighbour queries in DBSCAN
        self.use_weights = use_weights
        self.home_v2 = home_v2
        self.start_night_hour = start_night_hour
        self.end_night_hour = end_night_hour
        self.min_f_secondary_home = min_f_secondary_home
        self.home_merge_m = home_merge_m
        self.dedup_grid_m = dedup_grid_m
        self.home_clusters = None
        self.home_diagnostics = None

        # Ensure 'timestamp_start' is datetime
        self.df['timestamp_start'] = pd.to_datetime(self.df['timestamp_start'], errors='coerce')
        # Ensure 'Latitude' and 'Longitude' are floats
        self.df['Latitude'] = self.df['Latitude'].astype(float)
        self.df['Longitude'] = self.df['Longitude'].astype(float)

        # Extract hour and day from 'timestamp_start'
        self.df['hour_gps'] = self.df['timestamp_start'].dt.hour
        self.df['day_gps'] = self.df['timestamp_start'].dt.date

    def data_quality_check(self):
        """Filter out ids with insufficient data points."""
        id_counts = self.df.groupby('id', observed=True).size().reset_index(name='point_count')
        valid_ids = id_counts[id_counts['point_count'] >= self.min_data_points]['id']
        self.df = self.df[self.df['id'].isin(valid_ids)]
        logging.info(f"Data quality check: {len(valid_ids)} ids with sufficient data retained.")

    def calculate_distances_and_speeds(self):
        """Calculate distances and speeds for each id (legacy, distance to PREVIOUS sample)."""
        self.df['distance'], self.df['time_diff'], self.df['speed'] = np.nan, np.nan, np.nan

        for id in self.df['id'].unique():
            mask = self.df['id'] == id
            id_data = self.df.loc[mask].sort_values('timestamp_start')

            distances = self._calculate_distances(id_data)
            time_diffs = id_data['timestamp_start'].diff().dt.total_seconds().fillna(0)
            speeds = distances / time_diffs.replace(0, np.nan)

            # Assign calculated values using id_data.index to ensure alignment
            self.df.loc[id_data.index, 'distance'] = distances
            self.df.loc[id_data.index, 'time_diff'] = time_diffs
            self.df.loc[id_data.index, 'speed'] = speeds

    def calculate_distances_and_speeds_v2(self):
        """Calculate distances and speeds to the NEXT sample."""
        self.df = self.df.sort_values(by=['id', 'timestamp_start'])
        gby = self.df.groupby('id', observed=True)

        self.df['time_diff'] = (-gby['timestamp_start'].diff(-1)).dt.total_seconds()

        lat_next = gby['Latitude'].shift(-1)
        lon_next = gby['Longitude'].shift(-1)

        self.df['distance'] = haversine(self.df['Longitude'], self.df['Latitude'], lon_next, lat_next)
        self.df['speed'] = self.df['distance'] / self.df['time_diff'].replace(0, np.nan)
        self.df['distance'] = self.df['distance'].fillna(0)
        self.df['time_diff'] = self.df['time_diff'].fillna(0)

    def calculate_stationary_and_transition(self):
        """Determine stationary points and transition status based on speed and distance."""
        # Filter out points with speed > 220 km/h (converted to m/s)
        self.df = self.df[self.df['speed'] <= 220 * 1000 / 3600]
        self.df['stationary'] = (self.df['speed'] < self.speed_limit) & (self.df['distance'] < self.max_distance)
        self.df['transition'] = np.where(self.df['stationary'], 0, 1)
        return self.df

    def _calculate_distances(self, df):
        """Helper method to calculate distances using haversine formula."""
        coords = df[['Latitude', 'Longitude']].values
        distances = np.array([
            self._haversine(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
            for i in range(1, len(coords))
        ])
        return np.append(distances, 0)

    def _haversine(self, lon1, lat1, lon2, lat2):
        """Haversine formula to calculate distance between two lat/lon points in meters."""
        R = 6371000  # Earth radius in meters
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _to_xyz(self, df):
        R = 6371000
        lat_rad = np.radians(df['Latitude'])
        lon_rad = np.radians(df['Longitude'])
        x = R * np.cos(lat_rad) * np.cos(lon_rad)
        y = R * np.cos(lat_rad) * np.sin(lon_rad)
        z = R * np.sin(lat_rad)
        return x, y, z

    def apply_clustering(self, df):
        """Apply clustering per id."""
        df_cleaned = df.dropna(subset=['Longitude', 'Latitude'])

        clusters = []
        for id_id, group_df in df_cleaned.groupby('id', observed=True):
            group_df = group_df.copy()
            group_df['id'] = id_id
            cluster_result = self._apply_clustering_method(group_df)
            clusters.append(cluster_result)

        cluster_results = pd.concat(clusters, ignore_index=True)
        return cluster_results

    def _apply_clustering_method(self, df):
        """Weighted DBSCAN on XYZ coordinates; near-duplicate points are merged on a grid
        (weights summed) to keep memory bounded."""
        id_point_count = len(df)
        id_id = df['id'].iloc[0]

        # Skip clustering for ids with too few points
        if id_point_count < self.min_data_points:
            logging.info(f"id {id_id} has too few data points ({id_point_count}). Skipping clustering.")
            df['cluster'] = -1
            return df

        if self.normalize_min_samples:
            min_samples = max(2, int(id_point_count * 0.03))  # 3% of points, with a minimum of 2
        else:
            min_samples = self.min_samples

        x, y, z = self._to_xyz(df)
        X = np.column_stack((x, y, z))

        if 'sample_weight' in df.columns:
            w = df['sample_weight'].to_numpy(dtype=float)
        else:
            w = np.ones(len(df))  # preserves "count of points" semantics

        # Collapse near-duplicate points onto a grid to bound neighborhood size
        grid_m = self.dedup_grid_m
        keys = np.round(X / grid_m).astype(np.int64)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        w_u = np.bincount(inverse, weights=w)
        X_u = uniq.astype(float) * grid_m

        logging.info(f"id {id_id}: {len(df)} points -> {len(X_u)} unique cells")

        clustering_model = DBSCAN(eps=self.epsilon, min_samples=min_samples,
                                  metric="euclidean", n_jobs=self.n_jobs)
        labels_u = clustering_model.fit_predict(X_u, sample_weight=w_u)
        df['cluster'] = labels_u[inverse].astype(int)
        return df

    def find_home_cluster(self, geodata_clusters):
        """Legacy: identify the home cluster based on nighttime point counts, with fallback to the largest cluster."""
        geodata_night = geodata_clusters.loc[
            (geodata_clusters['hour_gps'] >= 20) | (geodata_clusters['hour_gps'] <= 6)
        ].copy()

        geodata_clusters['home'] = np.nan

        if not geodata_night.empty:
            valid_clusters_night = geodata_night[geodata_night['cluster'] != -1].copy()

            if not valid_clusters_night.empty:
                def safe_mode(x):
                    modes = x.mode()
                    if len(modes) == 1:
                        return modes.iloc[0]
                    elif len(modes) > 1:
                        return modes.min()
                    else:
                        return np.nan

                valid_clusters_night['home'] = valid_clusters_night.groupby('id')['cluster'].transform(safe_mode)
                valid_clusters_night['nights_with_obs'] = valid_clusters_night.groupby('id')['day_gps'].transform('nunique')
                valid_clusters_night['n_home'] = valid_clusters_night.groupby(['id', 'home'])['day_gps'].transform('size')
                valid_clusters_night['night_obs'] = valid_clusters_night.groupby('id')['day_gps'].transform('size')
                valid_clusters_night['f_home'] = valid_clusters_night['n_home'] / valid_clusters_night['night_obs']
                valid_clusters_night['home'] = valid_clusters_night.apply(
                    lambda x: x['home'] if (x['nights_with_obs'] >= self.min_nights_obs) and (x['f_home'] >= self.min_f_home) else np.nan, axis=1
                )

                home_mapping = valid_clusters_night[['id', 'home']].drop_duplicates(subset=['id'])
                geodata_clusters = geodata_clusters.merge(home_mapping, on='id', how='left', suffixes=('', '_temp'))
                geodata_clusters['home'] = geodata_clusters['home'].fillna(geodata_clusters['home_temp'])
                geodata_clusters.drop(columns=['home_temp'], inplace=True)

        no_home_ids = geodata_clusters.loc[geodata_clusters['home'].isna(), 'id'].unique()
        logging.info(f"ids with no home after time-based method: {len(no_home_ids)}")

        if len(no_home_ids) > 0:
            fallback_home_clusters = (
                geodata_clusters[geodata_clusters['id'].isin(no_home_ids) & (geodata_clusters['cluster'] != -1)]
                .groupby(['id', 'cluster'])
                .size()
                .reset_index(name='cluster_size')
            )

            if not fallback_home_clusters.empty:
                fallback_home_clusters = fallback_home_clusters.loc[
                    fallback_home_clusters.groupby('id')['cluster_size'].idxmax()
                ]
                fallback_home_clusters['home'] = fallback_home_clusters['cluster']
                fallback_home_mapping = fallback_home_clusters[['id', 'home']].drop_duplicates()
                geodata_clusters = geodata_clusters.merge(fallback_home_mapping, on='id', how='left', suffixes=('', '_fallback'))
                geodata_clusters['home'] = geodata_clusters['home'].fillna(geodata_clusters['home_fallback'])
                geodata_clusters.drop(columns=['home_fallback'], inplace=True)
                logging.info(f"Fallback home clusters assigned: {len(fallback_home_clusters)}")

        final_no_home = geodata_clusters.loc[geodata_clusters['home'].isna(), 'id'].unique()
        logging.warning(f"{len(final_no_home)} ids still do not have a home cluster.")

        geodata_clusters['homeID'] = geodata_clusters.apply(
            lambda x: f"{x['id']}00{int(x['home'])}" if pd.notna(x['home']) else None, axis=1
        )

        return geodata_clusters

    def find_home_cluster_v2(self, geodata_clusters):
        """Identify home cluster(s) based on weighted night-time duration.

        Primary home = cluster with the most night time (start_night_hour-end_night_hour),
        observed on >= min_nights_obs distinct nights.
        Home clusters = primary + clusters within home_merge_m of it (split home)
                        + clusters with >= min_f_secondary_home of night time on
                          >= min_nights_obs nights (second home).
        A home is only assigned if all home clusters together reach >= min_f_home of
        all night time (noise included in the denominator).
        """
        if 'sample_weight' not in geodata_clusters.columns:
            logging.warning("sample_weight not found, using default home estimation.")
            return self.find_home_cluster(geodata_clusters)

        df = geodata_clusters[["id", "cluster", "local_timestamp_start", "sample_weight"]].copy()
        s = df["local_timestamp_start"]
        e = s + pd.to_timedelta(df["sample_weight"], unit="s")

        # Vectorized night overlap. Intervals are <= 12 h, so only the night windows
        # starting the day before and the same day as the interval start can overlap.
        d0 = s.dt.normalize()
        night_sec = pd.Series(0.0, index=df.index)
        night_date = pd.Series(pd.NaT, index=df.index, dtype=d0.dtype)
        for k in (-1, 0):
            ns = d0 + pd.Timedelta(days=k, hours=self.start_night_hour)
            ne = d0 + pd.Timedelta(days=k + 1, hours=self.end_night_hour)
            ov = (e.where(e < ne, ne) - s.where(s > ns, ns)).dt.total_seconds().clip(lower=0)
            night_sec += ov
            night_date = night_date.where(~(ov > 0), d0 + pd.Timedelta(days=k))  # which night
        df["night_sec"] = night_sec
        df["night_date"] = night_date

        # Total night time per id INCLUDING noise (denominator)
        night_total = df.groupby("id", observed=True)["night_sec"].sum()

        clustered = df[(df["cluster"] != -1) & (df["night_sec"] > 0)]
        if clustered.empty:
            logging.warning("No valid clusters found for home estimation v2. Falling back.")
            return self.find_home_cluster(geodata_clusters)

        cs = (clustered.groupby(["id", "cluster"], observed=True)
              .agg(night_sec=("night_sec", "sum"), n_nights=("night_date", "nunique"))
              .reset_index())
        cs["f_home"] = cs["night_sec"] / cs["id"].map(night_total)
        n_before = cs["id"].nunique()          # ids with any clustered night time

        # cluster centroids (XYZ) for merging split homes
        lat, lon = np.radians(geodata_clusters["Latitude"]), np.radians(geodata_clusters["Longitude"])
        xyz = pd.DataFrame({"id": geodata_clusters["id"], "cluster": geodata_clusters["cluster"],
                            "x": np.cos(lat) * np.cos(lon), "y": np.cos(lat) * np.sin(lon),
                            "z": np.sin(lat)})
        cent = (xyz[xyz["cluster"] != -1]
                .groupby(["id", "cluster"], observed=True)[["x", "y", "z"]].mean())
        cs = cs.join(cent, on=["id", "cluster"])

        # primary home = most night time, needs enough nights
        primary = cs.loc[cs.groupby("id", observed=True)["night_sec"].idxmax()]
        primary = primary[primary["n_nights"] >= self.min_nights_obs]
        p = primary[["id", "cluster", "x", "y", "z"]].rename(
            columns={"cluster": "primary", "x": "px", "y": "py", "z": "pz"})
        cs = cs.merge(p, on="id", how="inner")

        dot = ((cs["x"] * cs["px"] + cs["y"] * cs["py"] + cs["z"] * cs["pz"])
               / (np.sqrt(cs["x"]**2 + cs["y"]**2 + cs["z"]**2)
                  * np.sqrt(cs["px"]**2 + cs["py"]**2 + cs["pz"]**2)))
        cs["dist_primary_m"] = np.arccos(np.clip(dot, -1, 1)) * 6371000

        cs["is_home"] = (
            (cs["cluster"] == cs["primary"])
            | (cs["dist_primary_m"] <= self.home_merge_m)                      # split home
            | ((cs["f_home"] >= self.min_f_secondary_home)
               & (cs["n_nights"] >= self.min_nights_obs))                      # second home
        )
        home_f = cs[cs["is_home"]].groupby("id", observed=True)["f_home"].sum()
        ok_ids = home_f[home_f >= self.min_f_home].index

        self.home_diagnostics = cs.copy()
        self.home_clusters = cs.loc[cs["is_home"] & cs["id"].isin(ok_ids), ["id", "cluster"]]

        n_multi = (self.home_clusters.groupby("id")["cluster"].nunique() > 1).sum()
        logging.info(f"Home v2: {len(ok_ids)}/{n_before} ids have a home "
                     f"({n_multi} with multiple home clusters)")

        geodata_clusters["home"] = (geodata_clusters["id"].map(primary.set_index("id")["cluster"])
                                    .where(geodata_clusters["id"].isin(ok_ids)))
        return geodata_clusters

    def determine_if_at_home(self, df):
        """at_home: 1 = in a home cluster, 0 = in another cluster, -1 = noise."""
        df["cluster"] = df["cluster"].astype(int)
        df["home"] = df["home"].astype(float)
        ids = df["id"].astype(str)
        df["clusterID"] = (ids + "00" + df["cluster"].astype(str)).where(df["cluster"] != -1)
        df["homeID"] = (ids + "00" + df["home"].astype("Int64").astype(str)).where(df["home"].notna())

        if self.home_clusters is not None:
            hs = self.home_clusters.assign(_is_home=1)
            df = df.merge(hs, on=["id", "cluster"], how="left")
            df["at_home"] = np.where(df["cluster"] == -1, -1, df["_is_home"].fillna(0).astype(int))
            df = df.drop(columns="_is_home")
        else:
            df["at_home"] = np.where(df["cluster"] == -1, -1,
                                     (df["clusterID"] == df["homeID"]).astype(int))
        return df

    def run(self):
        """Run the full extraction process."""
        logging.info("Starting home cluster extraction process.")
        self.data_quality_check()
        logging.info("Data quality check completed.")

        if self.use_weights:
            self.calculate_distances_and_speeds_v2()
            self.df['sample_weight'] = calculate_sample_weights(self.df)
        else:
            self.calculate_distances_and_speeds()

        logging.info("Distance and speed calculations completed.")
        self.df = self.calculate_stationary_and_transition()
        logging.info("Stationary and transition points calculated.")

        # Apply clustering based on all data (not just stationary points)
        geodata_cluster_df = self.apply_clustering(self.df)
        logging.info("Clustering completed.")

        geodata_clusters = geodata_cluster_df.copy()
        geodata_clusters['cluster'] = geodata_clusters['cluster'].fillna(-1)

        # Find home cluster(s)
        if self.home_v2:
            geodata_clusters = self.find_home_cluster_v2(geodata_clusters)
        else:
            geodata_clusters = self.find_home_cluster(geodata_clusters)
        logging.info("Home cluster finding completed.")

        # Determine if the person is at home
        geodata_clusters = self.determine_if_at_home(geodata_clusters)
        logging.info("At-home determination completed.")

        return geodata_clusters