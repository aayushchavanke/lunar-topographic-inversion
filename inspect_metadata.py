import os
import sys
import pandas as pd

def inspect_metadata(train_path="train_metadata.csv", test_path="test_metadata.csv"):
    print("=" * 60)
    print("METADATA INSPECTION REPORT: The Pareidolia Paradox")
    print("=" * 60)

    # 1. Check file existence
    for path, name in [(train_path, "Train metadata"), (test_path, "Test metadata")]:
        if not os.path.exists(path):
            print(f"Error: {name} file not found at '{path}'.", file=sys.stderr)
            return

    try:
        train_df = pd.read_csv(train_path)
        print(f"\n[+] Loaded '{train_path}' successfully.")
    except Exception as e:
        print(f"Error reading '{train_path}': {e}", file=sys.stderr)
        return

    try:
        test_df = pd.read_csv(test_path)
        print(f"[+] Loaded '{test_path}' successfully.")
    except Exception as e:
        print(f"Error reading '{test_path}': {e}", file=sys.stderr)
        return

    # Check required columns
    required_train_cols = ["image_id", "sun_azimuth_angle", "label"]
    required_test_cols = ["image_id", "sun_azimuth_angle"]

    train_missing_cols = [c for c in required_train_cols if c not in train_df.columns]
    test_missing_cols = [c for c in required_test_cols if c not in test_df.columns]

    if train_missing_cols:
        print(f"Error: Missing columns in '{train_path}': {train_missing_cols}", file=sys.stderr)
        return
    if test_missing_cols:
        print(f"Error: Missing columns in '{test_path}': {test_missing_cols}", file=sys.stderr)
        return

    # (4) Row counts check
    print("\n" + "-" * 50)
    print("(4) TOTAL ROW COUNTS")
    print("-" * 50)
    train_rows = len(train_df)
    test_rows = len(test_df)
    print(f"Train row count: {train_rows:,} (Expected: 7,854) -> {'MATCH' if train_rows == 7854 else 'MISMATCH'}")
    print(f"Test row count:  {test_rows:,} (Expected: 2,000) -> {'MATCH' if test_rows == 2000 else 'MISMATCH'}")

    # (1) Column names and dtypes
    print("\n" + "-" * 50)
    print("(1) COLUMNS & DATA TYPES")
    print("-" * 50)
    print("Train Columns & Dtypes:")
    for col, dtype in train_df.dtypes.items():
        print(f"  - {col}: {dtype}")

    print("\nTest Columns & Dtypes:")
    for col, dtype in test_df.dtypes.items():
        print(f"  - {col}: {dtype}")

    # (2) Summary statistics for sun_azimuth_angle
    print("\n" + "-" * 50)
    print("(2) SUMMARY STATISTICS: sun_azimuth_angle")
    print("-" * 50)
    for name, df in [("Train", train_df), ("Test", test_df)]:
        angle_series = df["sun_azimuth_angle"]
        print(f"\n{name} 'sun_azimuth_angle':")
        print(f"  - Min:  {angle_series.min():.6f}")
        print(f"  - Max:  {angle_series.max():.6f}")
        print(f"  - Mean: {angle_series.mean():.6f}")
        print(f"  - Std:  {angle_series.std():.6f}")
        print(f"  - Median: {angle_series.median():.6f}")

    # (3) Value counts and proportions of label in train
    print("\n" + "-" * 50)
    print("(3) LABEL VALUE COUNTS & PROPORTIONS (TRAIN)")
    print("-" * 50)
    counts = train_df["label"].value_counts(dropna=False)
    proportions = train_df["label"].value_counts(normalize=True, dropna=False) * 100
    label_summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": proportions
    })
    print(label_summary.to_string())

    # (5) Missing / NaN checks
    print("\n" + "-" * 50)
    print("(5) MISSING / NaN VALUE CHECK")
    print("-" * 50)
    train_azimuth_nan = train_df["sun_azimuth_angle"].isna().sum()
    train_label_nan = train_df["label"].isna().sum()
    test_azimuth_nan = test_df["sun_azimuth_angle"].isna().sum()

    print(f"Train 'sun_azimuth_angle' missing values: {train_azimuth_nan} ({train_azimuth_nan / train_rows * 100:.2f}%)")
    print(f"Train 'label' missing values:              {train_label_nan} ({train_label_nan / train_rows * 100:.2f}%)")
    print(f"Test 'sun_azimuth_angle' missing values:  {test_azimuth_nan} ({test_azimuth_nan / test_rows * 100:.2f}%)")

    total_missing_train = train_df.isna().sum().sum()
    total_missing_test = test_df.isna().sum().sum()
    print(f"\nTotal missing values across all columns in train: {total_missing_train}")
    print(f"Total missing values across all columns in test:  {total_missing_test}")
    print("=" * 60)

if __name__ == "__main__":
    inspect_metadata()
