import pandas as pd
import requests
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_labeled_mutations.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "structures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_structure(uniprot_id):
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    try:
        response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            return False

        predictions = response.json()

        if not predictions:
            return False

        pdb_url = predictions[0].get("pdbUrl")

        if not pdb_url:
            return False

        pdb_response = requests.get(pdb_url, timeout=60)

        if pdb_response.status_code != 200:
            return False

        output_file = OUTPUT_DIR / f"{uniprot_id}.pdb"
        output_file.write_bytes(pdb_response.content)

        return True

    except requests.RequestException:
        return False


def main():
    df = pd.read_csv(INPUT_FILE, dtype={"UniProt_ID": str})

    uniprot_ids = (
        df["UniProt_ID"]
        .dropna()
        .astype(str)
        .unique()
    )

    print(f"Structures required: {len(uniprot_ids)}")

    downloaded = 0
    failed = 0

    for i, uniprot_id in enumerate(uniprot_ids, start=1):

        print(f"[{i}/{len(uniprot_ids)}] Downloading {uniprot_id}...")

        if download_structure(uniprot_id):
            downloaded += 1
        else:
            failed += 1

        time.sleep(0.2)

    print("\nStructure download complete!")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {failed}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()