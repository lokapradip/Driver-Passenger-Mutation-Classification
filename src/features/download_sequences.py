import time
import requests
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brca_labeled_mutations.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

BASE_URL = "https://rest.uniprot.org/uniprotkb"


def download_sequence(uniprot_id):
    url = f"{BASE_URL}/{uniprot_id}.fasta"

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=60
            )

            if response.status_code == 200:
                return response.text

            print(
                f"{uniprot_id}: HTTP {response.status_code}"
            )

        except requests.RequestException as error:
            print(
                f"{uniprot_id}: attempt {attempt + 1}/3 failed"
            )

        time.sleep(3)

    return None


def main():
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"UniProt_ID": str}
    )

    ids = (
        df["UniProt_ID"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = {
        file.stem
        for file in OUTPUT_DIR.glob("*.fasta")
    }

    print(f"Total proteins: {len(ids)}")
    print(f"Already downloaded: {len(existing)}")
    print(f"Remaining: {len(ids) - len(existing)}")

    for number, uniprot_id in enumerate(ids, start=1):

        output_file = OUTPUT_DIR / f"{uniprot_id}.fasta"

        if uniprot_id in existing:
            continue

        print(
            f"[{number}/{len(ids)}] "
            f"Downloading {uniprot_id}..."
        )

        sequence = download_sequence(uniprot_id)

        if sequence:
            output_file.write_text(
                sequence,
                encoding="utf-8"
            )

            print(f"Saved: {uniprot_id}")
        else:
            print(f"Skipped: {uniprot_id}")

        time.sleep(1)

    final_count = len(
        list(OUTPUT_DIR.glob("*.fasta"))
    )

    print("\nDownload process finished.")
    print(f"Sequences available: {final_count}/{len(ids)}")


if __name__ == "__main__":
    main()