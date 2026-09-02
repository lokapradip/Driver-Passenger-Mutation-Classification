import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_labeled_mutations.csv"
STRUCTURE_DIR = PROJECT_ROOT / "data" / "processed" / "structures"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "brca_structure_features.csv"


def parse_pdb(pdb_file):
    residues = {}

    with open(pdb_file, "r") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue

            atom = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21].strip()
            res_num_text = line[22:26].strip()

            if not res_num_text:
                continue

            try:
                res_num = int(res_num_text)
            except ValueError:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                b_factor = float(line[60:66])
            except ValueError:
                continue

            key = (chain, res_num)

            if key not in residues:
                residues[key] = {
                    "plddt": [],
                    "ca": None
                }

            residues[key]["plddt"].append(b_factor)

            if atom == "CA":
                residues[key]["ca"] = np.array([x, y, z])

    for key in residues:
        residues[key]["plddt"] = np.mean(residues[key]["plddt"])

    return residues


def get_structure_features(uniprot_id, position):
    pdb_file = STRUCTURE_DIR / f"{uniprot_id}.pdb"

    if not pdb_file.exists():
        return None

    residues = parse_pdb(pdb_file)

    if not residues:
        return None

    target_candidates = [
        key for key in residues
        if key[1] == int(position)
    ]

    if not target_candidates:
        return None

    target_key = target_candidates[0]
    target = residues[target_key]

    target_plddt = target["plddt"]
    target_ca = target["ca"]

    if target_ca is None:
        return None

    ca_residues = []

    for key, data in residues.items():
        if data["ca"] is not None:
            ca_residues.append(
                (key, data["ca"], data["plddt"])
            )

    distances = []

    for key, ca, plddt in ca_residues:
        if key == target_key:
            continue

        distance = np.linalg.norm(target_ca - ca)

        if distance <= 8.0:
            distances.append((distance, plddt))

    if distances:
        neighbor_distances = [x[0] for x in distances]
        neighbor_plddts = [x[1] for x in distances]

        neighbor_count = len(distances)
        mean_neighbor_distance = np.mean(neighbor_distances)
        mean_neighbor_plddt = np.mean(neighbor_plddts)
    else:
        neighbor_count = 0
        mean_neighbor_distance = np.nan
        mean_neighbor_plddt = np.nan

    local_plddts = []

    for key, data in residues.items():
        if abs(key[1] - int(position)) <= 5:
            local_plddts.append(data["plddt"])

    local_mean_plddt = (
        np.mean(local_plddts)
        if local_plddts
        else np.nan
    )

    return {
        "Structure_pLDDT": target_plddt,
        "Local_Mean_pLDDT": local_mean_plddt,
        "Neighbor_Count_8A": neighbor_count,
        "Mean_Neighbor_Distance_8A": mean_neighbor_distance,
        "Mean_Neighbor_pLDDT": mean_neighbor_plddt
    }


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"UniProt_ID": str}
    )

    results = []

    total = len(df)
    processed = 0
    skipped = 0

    for i, row in df.iterrows():

        if i % 250 == 0:
            print(f"Processing {i}/{total}...")

        features = get_structure_features(
            row["UniProt_ID"],
            row["Protein_Position"]
        )

        if features is None:
            skipped += 1
            continue

        record = {
            "Gene": row["Gene"],
            "Mutation": row["Mutation"],
            "Protein_Position": row["Protein_Position"],
            "UniProt_ID": row["UniProt_ID"],
            "Label": row["Label"]
        }

        record.update(features)

        results.append(record)
        processed += 1

    output_df = pd.DataFrame(results)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nStructural feature extraction complete!")
    print(f"Total mutations: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Structural features: 5")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()