import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "external" / "BRCA.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_labeled_mutations.csv"


def main():
    print("Loading BRCA labelled dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows: {len(df):,}")

    # Keep the core information needed for our project
    clean = df[
        [
            "Wild",
            "Mut",
            "Pos",
            "Class",
            "Uniprot ID",
            "Gene Name",
            "Mutation",
        ]
    ].copy()

    # Rename columns to clearer project names
    clean = clean.rename(
        columns={
            "Wild": "Wild_AA",
            "Mut": "Mutant_AA",
            "Pos": "Protein_Position",
            "Class": "Label",
            "Uniprot ID": "UniProt_ID",
            "Gene Name": "Gene",
            "Mutation": "Mutation",
        }
    )

    # Convert label to integer
    clean["Label"] = clean["Label"].astype(int)

    # Add cancer type
    clean["Cancer_Type"] = "BRCA"

    # Remove exact duplicate mutations
    clean = clean.drop_duplicates(
        subset=["UniProt_ID", "Gene", "Mutation", "Label"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(OUTPUT_FILE, index=False)

    print(f"Final rows: {len(clean):,}")
    print("\nLabel distribution:")
    print(clean["Label"].value_counts().sort_index())

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()