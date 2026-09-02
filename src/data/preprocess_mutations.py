import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brca_tcga_pan_can_atlas_2018"
    / "data_mutations.txt"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "brca_missense_mutations.csv"


def main():
    print("Loading BRCA mutation data...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        comment="#",
        low_memory=False
    )

    print(f"Total mutation records: {len(df):,}")

    missense = df[
        df["Variant_Classification"].astype(str).str.lower()
        == "missense_mutation"
    ].copy()

    print(f"Missense mutations: {len(missense):,}")

    missense = missense.drop_duplicates()

    print(f"After removing duplicates: {len(missense):,}")

    missense["Cancer_Type"] = "BRCA"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missense.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessing complete!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()