import pandas as pd

def summarize_missing_data(
    df,
    feature_group_pa,
    feature_group_gps,
    feature_group_hr,
    feature_group_weather,
    feature_group_person_static,
    columns_to_check,
    customer_id_col='customer'
):
    """
    Summarize missing data for seven groups:
      1) PA (Physical Activity)
      2) GPS
      3) HR (Heart Rate)
      4) Weather
      5) Person_Static (sociodemographic and clinical information)
      6) n_steps (single column)
      7) calories_burned (single column)

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing all relevant data.
        
    feature_group_pa : list of str
        List of column names related to Physical Activity.
        
    feature_group_gps : list of str
        List of column names related to GPS data.
        
    feature_group_hr : list of str
        List of column names related to Heart Rate data.
        
    feature_group_weather : list of str
        List of column names related to Weather data.
        
    feature_group_person_static : list of str
        List of column names related to Person-Static features.
        
    columns_to_check : list of str
        List of all column names to consider for missingness.
        
    customer_id_col : str, optional (default='customer')
        The name of the column that uniquely identifies each customer.

    Returns
    -------
    group_missing_df : pd.DataFrame
        DataFrame with the same index as `df` and exactly these columns:
        ["PA", "GPS", "HR", "Weather", "Person_Static", "n_steps", "calories_burned"].
        Each cell is 1 if that group is missing for that row, else 0.

    Notes
    -----
    - Person-Static features are analyzed per customer.
    - Any columns not in `columns_to_check` or not present in the DataFrame are ignored.
    - The function prints group-wise missing data summaries.
    """
    # -------------------------------------------------------------------------
    # 1. Analyze Person-Static Features by Customer
    # -------------------------------------------------------------------------
    if customer_id_col not in df.columns:
        raise ValueError(f"Customer identifier column '{customer_id_col}' not found in DataFrame.")
    
    # Filter person-static features that are present in the DataFrame
    person_static_present = [col for col in feature_group_person_static if col in df.columns]
    
    # Count the number of unique customers
    total_customers = df[customer_id_col].nunique()
    
    if person_static_present:
        # Step 1: For each customer, check if any of their person-static features are missing (NaN)
        # This results in a Series indexed by customer_id_col with True/False
        missing_person_static_series = df.groupby(customer_id_col)[person_static_present].apply(
            lambda x: x.isna().any().any()
        )
        
        # Step 2: Convert the boolean Series to integers (1 for missing, 0 otherwise)
        missing_person_static_series = missing_person_static_series.astype(int)
        
        # Step 3: Map this Series back to the original DataFrame based on customer_id_col
        df['Person_Static'] = df[customer_id_col].map(missing_person_static_series)
        
        # Step 4: Handle any NaN values resulting from customers not present in person_static_present
        df['Person_Static'] = df['Person_Static'].fillna(0).astype(int)
    else:
        # If no person-static features are present, set Person_Static to 0
        df['Person_Static'] = 0

    # -------------------------------------------------------------------------
    # 2. Define Feature Groups for Missingness Analysis
    # -------------------------------------------------------------------------
    # Only consider columns actually in df AND in columns_to_check
    relevant_cols = set(df.columns).intersection(columns_to_check)
    
    # Define the groups, filtering out columns not in relevant_cols
    group_dict = {
        "PA": [c for c in feature_group_pa if c in relevant_cols],
        "GPS": [c for c in feature_group_gps if c in relevant_cols],
        "HR": [c for c in feature_group_hr if c in relevant_cols],
        "Weather": [c for c in feature_group_weather if c in relevant_cols],
        "Person_Static": ['Person_Static'] if 'Person_Static' in df.columns else [],
        "n_steps": ["n_steps"] if "n_steps" in relevant_cols else [],
        "calories_burned": ["calories_burned"] if "calories_burned" in relevant_cols else []
    }

    # -------------------------------------------------------------------------
    # 3. Print Missing Data Summaries
    # -------------------------------------------------------------------------
    print("=== Missing Data Analysis ===\n")
    
    # 3.1 Person-Static Features Missingness per Feature (based on unique customers)
    if person_static_present:
        print("---- Person-Static Features Missingness (Based on Unique Customers) ----\n")
        for col in person_static_present:
            missing_count = df.groupby(customer_id_col)[col].apply(lambda x: x.isna().any()).sum()
            pct_missing = (missing_count / total_customers) * 100
            print(f"  {col}: {missing_count} missing [ {pct_missing:.2f}% of unique customers ]")
        print()
    
    # 3.2 Group-wise summary of missing data (for other groups and Person_Static)
    print("---- Group-wise Missing Data Summary ----\n")
    total_rows = len(df)
    
    for group_name, cols in group_dict.items():
        if not cols:
            # If a group has zero columns, skip.
            continue

        print(f"Group: {group_name} (contains {len(cols)} column(s))")
        print("-" * 50)
        for col in cols:
            if group_name == "Person_Static" and col == "Person_Static":
                # For Person_Static, count how many customers have any missing person-static feature
                missing_count = df[col].sum()
                pct_missing = (missing_count / total_customers) * 100
                print(f"  {col}: {missing_count} missing (any person-static feature)  [{pct_missing:.2f}% of unique customers]")
            else:
                # For other groups, missingness is via -1
                missing_count = (df[col] == -1).sum()
                pct_missing = (missing_count / total_rows) * 100
                print(f"  {col}: {missing_count} missing (-1)  [{pct_missing:.2f}% of total rows]")
        print()
    
    print("=== End of Missing Data Analysis ===\n")

    # -------------------------------------------------------------------------
    # 4. Create and return group_missing_df
    # -------------------------------------------------------------------------
    # Initialize the DataFrame with zeros
    group_missing_df = pd.DataFrame(0, index=df.index, columns=["PA", "GPS", "HR", "Weather", "Person_Static", "n_steps", "calories_burned"])

    # Populate the DataFrame with 1s where any column in the group is missing (-1)
    for group_name, cols in group_dict.items():
        if not cols:
            continue
        if group_name == "Person_Static":
            group_missing_df[group_name] = df['Person_Static']
        else:
            group_missing_df[group_name] = (df[cols] == -1).any(axis=1).astype(int)

    return group_missing_df



def compute_availability_metrics(sub):
    """
    Given a subset of the dataframe for a single participant,
    return the daily availability metrics (including days with no data),
    and retain the ema_base_start column.
    """
    def row_has_data(row):
        t = row['type']
        if t in double_value_types:
            return not pd.isnull(row['doubleValue'])
        elif t in long_value_types:
            return not pd.isnull(row['longValue'])
        elif t in string_value_types:
            return not pd.isnull(row['stringValue'])
        elif t in boolean_value_types:
            return not pd.isnull(row['booleanValue'])
        return False
    
    # 1) Mark rows that truly have data
    sub['has_data'] = sub.apply(row_has_data, axis=1)

    # 2) Aggregate at hourly level
    hourly = (
        sub.groupby(['type', 'startTimestamp_day', 'startTimestamp_hour'], observed=True)['has_data']
           .any()  # yields boolean: True if at least one row has data in that hour
           .reset_index(name='has_data_in_hour')
    )

    # 3) Aggregate at daily level
    daily_agg = (
        hourly.groupby(['type', 'startTimestamp_day'], observed=True)
              .agg(
                  available_binary=('has_data_in_hour', 'any'),  # True if any hour had data
                  available_hours=('has_data_in_hour', 'sum')     # count of hours with data
              )
              .reset_index()
    )
    # Convert boolean to int for available_binary; available_hours remains numeric.
    daily_agg['available_binary'] = daily_agg['available_binary'].astype(int)
    daily_agg['available_hours'] = daily_agg['available_hours'].astype(int)

    # ----------- GENERATE FULL DATE RANGE FOR THIS PARTICIPANT -----------
    # Use ema_base_start from sub (it exists in the original dataframe)
    baseline_date = sub['ema_base_start'].min()
    max_date = sub['startTimestamp_day'].max()

    if pd.isnull(baseline_date) or pd.isnull(max_date):
        daily_agg['customer'] = sub['customer'].iloc[0]
        daily_agg['ema_base_start'] = sub['ema_base_start'].iloc[0]
        return daily_agg

    # Create a full daily date range from baseline_date to max_date
    all_dates = pd.date_range(start=baseline_date, end=max_date, freq='D')

    # Unique types present for this participant
    unique_types = sub['type'].unique()

    # Cartesian product of [all types] x [all dates]
    all_combos = pd.MultiIndex.from_product(
        [unique_types, all_dates],
        names=['type', 'startTimestamp_day']
    )
    all_days_df = all_combos.to_frame(index=False)

    # Merge daily_agg onto all possible (type, day) combos
    daily_agg_full = pd.merge(
        all_days_df,
        daily_agg,
        on=['type', 'startTimestamp_day'],
        how='left'
    )

    # Fill missing availability with 0
    daily_agg_full['available_binary'] = daily_agg_full['available_binary'].fillna(0)
    daily_agg_full['available_hours'] = daily_agg_full['available_hours'].fillna(0)

    # Convert them to int
    daily_agg_full['available_binary'] = daily_agg_full['available_binary'].astype(int)
    daily_agg_full['available_hours'] = daily_agg_full['available_hours'].astype(int)

    # Assign the participant ID and ema_base_start from the original data
    daily_agg_full['customer'] = sub['customer'].iloc[0]
    daily_agg_full['ema_base_start'] = sub['ema_base_start'].iloc[0]

    return daily_agg_full
