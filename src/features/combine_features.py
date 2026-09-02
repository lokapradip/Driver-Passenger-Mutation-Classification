import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_FILE = PROJECT_ROOT / "data" / "processed" / "brca_sequence_features.csv"
STRUCTURE_FILE = PROJECT_ROOT / "data" / "processed" / "brca_structure_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_combined_features.csv"

KEYS = [
    "Gene",
    "Mutation",
    "Protein_Position",
    "UniProt_ID"
]


def main():
    sequence_df = pd.read_csv(SEQUENCE_FILE)
    structure_df = pd.read_csv(STRUCTURE_FILE)

    print(f"Sequence rows: {len(sequence_df)}")
    print(f"Structure rows: {len(structure_df)}")

    combined = pd.merge(
        sequence_df,
        structure_df,
        on=KEYS,
        how="inner",
        suffixes=("", "_structure")
    )

    # Keep only one label column
    if "Label_structure" in combined.columns:
        if not (combined["Label"] == combined["Label_structure"]).all():
            raise ValueError("Label mismatch detected!")

        combined = combined.drop(columns=["Label_structure"])

    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nFeature combination complete!")
    print(f"Combined rows: {len(combined)}")
    print(f"Combined columns: {len(combined.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()