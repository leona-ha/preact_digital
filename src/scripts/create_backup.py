#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""Create a unified, timezone‑aware Parquet backup from mixed CSV exports.

Changes compared with the previous version
------------------------------------------
* All timestamp columns are now **tz‑aware (UTC)** from the moment they are
  parsed.  This fixes the FutureWarning about assigning tz‑aware values into
  tz‑naïve Series.
* Parquet schema updated to `pa.timestamp("ns", tz="UTC")`.
* Helper functions `epoch_to_utc` and `parse_iso_utc` simplified—no
  `.dt.tz_localize(None)` calls.
"""
# %%
import glob
import logging
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------
# Pfade wie im Notebook
# ---------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
for p in (BASE_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.append(p)

from server_config import backup_path, preprocessed_path, raw_path  # noqa: E402

PATTERN_BIG = os.path.join(backup_path, "first_backup", "tiki_backup_*.csv")
PATTERN_S1 = os.path.join(raw_path, "tiki_backup_files", "export_tiki_21052024", "epoch_part*.csv")
PATTERN_S2 = os.path.join(raw_path, "tiki_backup_files", "export_tiki_11112024", "epoch_part*.csv")
PATTERN_S3 = os.path.join(raw_path, "tiki_backup_files", "export_tiki_05052025", "epoch_part*.csv")
PATTERN_S4 = os.path.join(raw_path, "tiki_backup_files", "export_tiki_03112025", "epoch_part*.csv")


OUT_PARQUET = os.path.join(preprocessed_path, "backup_passive_05052025.parquet")

SCHEMA = [
    "customer",
    "startTimestamp",
    "endTimestamp",
    "timezoneOffset",
    "type",
    "stringValue",
    "booleanValue",
    "doubleValue",
    "longValue",
    "createdAt",
]

# Timestamps are tz‑aware UTC now
PA_SCHEMA = pa.schema(
    [
        pa.field("customer", pa.dictionary(pa.int64(), pa.utf8())),
        pa.field("startTimestamp", pa.timestamp("ns", tz="UTC")),
        pa.field("endTimestamp", pa.timestamp("ns", tz="UTC")),
        pa.field("timezoneOffset", pa.int32()),
        pa.field("type", pa.dictionary(pa.int64(), pa.utf8())),
        pa.field("stringValue", pa.utf8()),
        pa.field("booleanValue", pa.bool_()),
        pa.field("doubleValue", pa.float64()),
        pa.field("longValue", pa.float64()),
        pa.field("createdAt", pa.timestamp("ns", tz="UTC")), # !
        # pa.field("createdAt", pa.string()),
    ]
)

CHUNK = 1_000_000
MAX_MS = 4_102_444_800_000  # ~2100‑01‑01

# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------

def epoch_to_utc(series: pd.Series) -> pd.Series:
    """Convert epoch milliseconds to **tz‑aware UTC**; invalid values → NaT."""
    vals = pd.to_numeric(series, errors="coerce").astype("float64")
    mask = np.isfinite(vals) & (vals >= 0) & (vals <= MAX_MS)
    if (vals < 0).any():
        logging.warning(f"[EPOCH] negative timestamps → NaT: {series[vals < 0].unique()}")
    if (vals > MAX_MS).any():
        logging.warning(f"[EPOCH] too large timestamps → NaT: {series[vals > MAX_MS].unique()}")

    # Pre‑allocate a tz‑aware destination Series
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    if mask.any():
        out.loc[mask] = pd.to_datetime(
            vals[mask].astype("int64"), unit="ms", utc=True, errors="coerce"
        )
    return out


def parse_iso_utc(series: pd.Series) -> pd.Series:
    """Parse ISO‑8601 strings with offset to **tz‑aware UTC**; invalid → NaT."""
    ts = pd.to_datetime(series, format="%Y-%m-%dT%H:%M:%S%z", utc=True, errors="coerce")
    need_frac = ts.isna()
    if need_frac.any():
        ts_frac = pd.to_datetime(
            series[need_frac],
            format="%Y-%m-%dT%H:%M:%S.%f%z",
            utc=True,
            errors="coerce",
        )
        ts.loc[need_frac] = ts_frac
    return ts  # tz‑aware UTC


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing columns and enforce numeric dtypes."""
    for c in SCHEMA:
        if c not in df:
            df[c] = pd.NA

    # Hash/anonymise customer
    df["customer"] = (
        df["customer"].astype("string", copy=False).str.split("@").str[0].str[:4]
    )

    for c in ("timezoneOffset", "doubleValue", "longValue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[SCHEMA]

# %%
# ---------------------------------------------------------------------
# Writer / Zustand
# ---------------------------------------------------------------------

parquet_writer: Optional[pq.ParquetWriter] = None
last_ts: Optional[pd.Timestamp] = None
last_dataset: Optional[str] = None
os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)


def write_df(df: pd.DataFrame) -> None:
    """Normalise a chunk and write it as its own Row Group."""
    global parquet_writer

    df = ensure_schema(df)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Cast only if necessary (table.schema == PA_SCHEMA is cheap)
    if table.schema != PA_SCHEMA:
        table = table.cast(PA_SCHEMA, safe=False)

    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(
            OUT_PARQUET, table.schema, compression="snappy"
        )
        logging.info(f"[WRITE] opened {OUT_PARQUET}")

    parquet_writer.write_table(table)

# %%
# ---------------------------------------------------------------------
# BIG verarbeiten (ISO → UTC)
# ---------------------------------------------------------------------

big_files = sorted(
    glob.glob(PATTERN_BIG),
    key=lambda p: (
        m := re.match(r".*_(\d{4}-\d{2}-\d{2})_(\d+)\.csv", os.path.basename(p))
    )
    and (pd.to_datetime(m.group(1)), int(m.group(2)))
    or (pd.Timestamp.min, 0),
)

logging.info(f"[BIG] {len(big_files)} Dateien")
for f in big_files:
    for chunk in pd.read_csv(
        f,
        chunksize=CHUNK,
        low_memory=False,
        dtype={"startTimestamp": "str", "endTimestamp": "str"},
    ):
        chunk["startTimestamp"] = parse_iso_utc(chunk["startTimestamp"])
        chunk["endTimestamp"] = parse_iso_utc(chunk["endTimestamp"])
        # "createdAt" is not in these files

        for c in (
            "timezoneOffset",
            "type",
            "stringValue",
            "booleanValue",
            "doubleValue",
            "longValue",
            "customer",
        ):
            if c not in chunk:
                chunk[c] = pd.NA

        write_df(chunk)

        ts_valid = chunk["startTimestamp"].dropna()
        if not ts_valid.empty:
            m = ts_valid.max()
            if last_ts is None or m > last_ts:
                last_ts = m

last_dataset = "BIG"
logging.info(f"[BIG DONE] last_ts={last_ts}")

# %%
# ---------------------------------------------------------------------
# SMALL‑Datasets – Epoch ms → UTC
# ---------------------------------------------------------------------

for label, pattern in [("S1", PATTERN_S1), ("S2", PATTERN_S2), ("S3", PATTERN_S3), ("S4", PATTERN_S4)]:
    files = sorted(glob.glob(pattern))
    logging.info(f"[{label}] {len(files)} Dateien")

    prev_last_ts = last_ts
    first_raw: Optional[pd.Timestamp] = None
    first_written: Optional[pd.Timestamp] = None
    removed_overlap = 0

    for f in files:
        for chunk in pd.read_csv(
            f, encoding="latin-1", chunksize=CHUNK, low_memory=False
        ):
            chunk["startTimestamp"] = epoch_to_utc(chunk["startTimestamp"])
            chunk["endTimestamp"] = epoch_to_utc(chunk["endTimestamp"])
            chunk["createdAt"] = epoch_to_utc(chunk["createdAt"])

            if "timezoneOffset" not in chunk:
                chunk["timezoneOffset"] = pd.NA
            for c in (
                "type",
                "stringValue",
                "booleanValue",
                "doubleValue",
                "longValue",
                "customer",
            ):
                if c not in chunk:
                    chunk[c] = pd.NA

            if first_raw is None:
                tmp = chunk["startTimestamp"].dropna()
                if not tmp.empty:
                    first_raw = tmp.min()

            # Drop rows that overlap with the previous dataset’s max timestamp
            if (
                prev_last_ts is not None
                and last_dataset is not None
                and label != last_dataset
            ):
                before = len(chunk)
                chunk = chunk[chunk["startTimestamp"] >= prev_last_ts]
                removed_overlap += before - len(chunk)

            if chunk.empty:
                continue

            if first_written is None:
                tmp2 = chunk["startTimestamp"].dropna()
                if not tmp2.empty:
                    first_written = tmp2.min()

            write_df(chunk)

            ts_valid = chunk["startTimestamp"].dropna()
            if not ts_valid.empty:
                m = ts_valid.max()
                if last_ts is None or m > last_ts:
                    last_ts = m

    if last_dataset is not None and label != last_dataset:
        logging.info(
            f"[BOUNDARY] {last_dataset} -> {label}: "
            f"prev_last_ts={prev_last_ts}; first_raw_new_df={first_raw}; "
            f"first_written={first_written}; new_last_ts={last_ts}; "
            f"dropped_overlap_rows={removed_overlap}"
        )
    last_dataset = label

# ---------------------------------------------------------------------
# Abschluss
# ---------------------------------------------------------------------
# %%
if parquet_writer is not None:
    parquet_writer.close()
    logging.info(f"[DONE] last_ts={last_ts} -> {OUT_PARQUET}")
else:
    logging.info("[WARN] nothing written")


# %%
