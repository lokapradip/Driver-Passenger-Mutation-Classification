import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_combined_features.csv"

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "brca_train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "brca_blind_test.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    X = df.drop(columns=["Label"])
    y = df["Label"]
    groups = df["UniProt_ID"]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    train_proteins = set(train_df["UniProt_ID"])
    test_proteins = set(test_df["UniProt_ID"])

    overlap = train_proteins.intersection(test_proteins)

    print("Leakage-safe split complete!")
    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Blind test rows: {len(test_df)}")
    print(f"Training proteins: {len(train_proteins)}")
    print(f"Test proteins: {len(test_proteins)}")
    print(f"Protein overlap: {len(overlap)}")

    print("\nTraining labels:")
    print(train_df["Label"].value_counts())

    print("\nBlind test labels:")
    print(test_df["Label"].value_counts())

    print(f"\nSaved training data: {TRAIN_FILE}")
    print(f"Saved blind test data: {TEST_FILE}")


if __name__ == "__main__":
    main()