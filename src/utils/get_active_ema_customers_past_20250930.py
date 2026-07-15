import sys
import pandas as pd
from pyprojroot import here

def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    val_str = str(val).strip()
    for fmt in ["%d.%m.%Y", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            return pd.to_datetime(val_str, format=fmt, exact=False).tz_localize(None)
        except Exception:
            pass
    try:
        return pd.to_datetime(val_str).tz_localize(None)
    except Exception:
        return pd.NaT

def main():
    # Dynamically import base_path from server_config
    sys.path.insert(0, str(here()))
    from server_config import base_path
    csv_path = base_path / "preprocessed" / "meta" / "SP6_meta_data.csv"

    # Load SP6 meta data CSV
    df = pd.read_csv(csv_path)

    # Filter criteria (from 30/09/2025 onwards)
    target_start = pd.to_datetime("2025-09-30").tz_localize(None)

    burst_columns = {
        "baseline": ("ema_base_start", "ema_base_end"),
        "t20": ("ema_t20_start", "ema_t20_end"),
        "tpost": ("ema_post_start", "ema_post_end")
    }

    results = []
    for idx, row in df.iterrows():
        cust_id = row["id"]
        for burst_name, (start_col, end_col) in burst_columns.items():
            start_dt = parse_date(row.get(start_col))
            end_dt = parse_date(row.get(end_col))
            if pd.isna(start_dt) or pd.isna(end_dt):
                continue
            if end_dt >= target_start:
                results.append({
                    "id": cust_id,
                    "burst": burst_name,
                    "start": start_dt.strftime("%Y-%m-%d"),
                    "end": end_dt.strftime("%Y-%m-%d")
                })

    # Remove duplicates (if any) and sort
    df_res = pd.DataFrame(results).drop_duplicates().sort_values(by=["id", "burst"]).reset_index(drop=True)

    # Resolve target directory (root/tmp/)
    tmp_dir = here() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_path = tmp_dir / "active_ema_customers_past_20250930.csv"

    # Save to tmp/
    df_res.to_csv(output_path, index=False)
    print(f"List of active customers successfully saved to {output_path}.")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
