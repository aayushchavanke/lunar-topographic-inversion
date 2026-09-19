import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(input_csv="train_metadata.csv", 
                  train_out="train_split.csv", 
                  val_out="val_split.csv", 
                  test_size=0.15, 
                  random_state=42):
    print("=" * 60)
    print("CREATING STRATIFIED TRAIN / VALIDATION SPLIT (85 / 15)")
    print("=" * 60)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found.")

    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f"[+] Loaded '{input_csv}' with {total_samples:,} total rows.")

    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
        shuffle=True
    )

    # Save splits
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"[+] Saved '{train_out}'")
    print(f"[+] Saved '{val_out}'")

    # Read back from saved CSVs to confirm directly from disk
    print("\n" + "-" * 50)
    print("VERIFYING CONFIRMED DATA FROM SAVED CSV FILES")
    print("-" * 50)
    train_loaded = pd.read_csv(train_out)
    val_loaded = pd.read_csv(val_out)

    # Overlap check
    overlap = set(train_loaded["image_id"]).intersection(set(val_loaded["image_id"]))
    print(f"Data leakage check (overlap count): {len(overlap)} (Expected: 0)")

    print(f"\nOriginal count:    {total_samples:,}")
    print(f"Train split count: {len(train_loaded):,} ({len(train_loaded)/total_samples*100:.2f}%)")
    print(f"Val split count:   {len(val_loaded):,} ({len(val_loaded)/total_samples*100:.2f}%)")
    print(f"Sum check:         {len(train_loaded) + len(val_loaded):,} == {total_samples:,} -> {'PASS' if len(train_loaded) + len(val_loaded) == total_samples else 'FAIL'}")

    print("\n" + "-" * 50)
    print("CLASS BALANCE DISTRIBUTION")
    print("-" * 50)

    def print_distribution(name, split_df):
        counts = split_df["label"].value_counts().sort_index()
        proportions = split_df["label"].value_counts(normalize=True).sort_index() * 100
        summary = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": proportions.map("{:.2f}%".format)
        })
        print(f"\n{name} (N={len(split_df):,}):")
        print(summary.to_string())

    print_distribution("Original Full Dataset", df)
    print_distribution("Train Split (85%)", train_loaded)
    print_distribution("Validation Split (15%)", val_loaded)
    print("=" * 60)

if __name__ == "__main__":
    create_splits()
