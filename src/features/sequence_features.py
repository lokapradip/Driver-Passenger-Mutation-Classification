import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brca_labeled_mutations.csv"
)

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "brca_sequence_features.csv"
)


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# Basic physicochemical properties
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1,
    "F": 0, "G": 0, "H": 0.1, "I": 0,
    "K": 1, "L": 0, "M": 0, "N": 0,
    "P": 0, "Q": 0, "R": 1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0
}

POLARITY = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3,
    "F": 5.2, "G": 9.0, "H": 10.4, "I": 5.2,
    "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6,
    "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2,
    "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2
}

MOLECULAR_WEIGHT = {
    "A": 89.09, "C": 121.15, "D": 133.10, "E": 147.13,
    "F": 165.19, "G": 75.07, "H": 155.16, "I": 131.17,
    "K": 146.19, "L": 131.17, "M": 149.21, "N": 132.12,
    "P": 115.13, "Q": 146.14, "R": 174.20, "S": 105.09,
    "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19
}


def read_fasta(path):
    """Read a FASTA file and return its amino-acid sequence."""
    lines = path.read_text(encoding="utf-8").splitlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if not line.startswith(">")
    )

    return sequence.upper()


def amino_acid_composition(sequence, prefix):
    """Calculate amino-acid frequencies."""
    length = len(sequence)

    if length == 0:
        return {
            f"{prefix}_{aa}": 0.0
            for aa in AMINO_ACIDS
        }

    return {
        f"{prefix}_{aa}": sequence.count(aa) / length
        for aa in AMINO_ACIDS
    }


def mutation_properties(wild, mutant):
    """Calculate physicochemical changes caused by mutation."""

    return {
        "Wild_Hydrophobicity": HYDROPHOBICITY[wild],
        "Mutant_Hydrophobicity": HYDROPHOBICITY[mutant],
        "Delta_Hydrophobicity": (
            HYDROPHOBICITY[mutant]
            - HYDROPHOBICITY[wild]
        ),

        "Wild_Charge": CHARGE[wild],
        "Mutant_Charge": CHARGE[mutant],
        "Delta_Charge": (
            CHARGE[mutant]
            - CHARGE[wild]
        ),

        "Wild_Polarity": POLARITY[wild],
        "Mutant_Polarity": POLARITY[mutant],
        "Delta_Polarity": (
            POLARITY[mutant]
            - POLARITY[wild]
        ),

        "Wild_Molecular_Weight": MOLECULAR_WEIGHT[wild],
        "Mutant_Molecular_Weight": MOLECULAR_WEIGHT[mutant],
        "Delta_Molecular_Weight": (
            MOLECULAR_WEIGHT[mutant]
            - MOLECULAR_WEIGHT[wild]
        ),
    }


def extract_mutation(mutation):
    """
    Extract wild-type amino acid, position and mutant amino acid.

    Example:
        R175H -> R, 175, H
    """

    match = re.fullmatch(
        r"([A-Z])(\d+)([A-Z])",
        str(mutation).strip().upper()
    )

    if not match:
        return None, None, None

    wild = match.group(1)
    position = int(match.group(2))
    mutant = match.group(3)

    return wild, position, mutant


def build_features(row):
    """Build all sequence features for one mutation."""

    uniprot_id = str(row["UniProt_ID"]).strip()
    mutation = str(row["Mutation"]).strip()

    wild, position, mutant = extract_mutation(mutation)

    sequence_file = SEQUENCE_DIR / f"{uniprot_id}.fasta"

    if not sequence_file.exists():
        return None

    sequence = read_fasta(sequence_file)

    if not sequence:
        return None

    if wild is None:
        return None

    if position < 1 or position > len(sequence):
        return None

    actual_residue = sequence[position - 1]

    # Check whether mutation agrees with the downloaded sequence
    residue_match = int(actual_residue == wild)

    features = {
        "UniProt_ID": uniprot_id,
        "Gene": row["Gene"],
        "Mutation": mutation,
        "Protein_Position": position,
        "Label": int(row["Label"]),

        "Protein_Length": len(sequence),

        "Relative_Position": position / len(sequence),

        "Wild_AA": wild,
        "Mutant_AA": mutant,

        "Sequence_Residue": actual_residue,
        "Wild_Residue_Match": residue_match,
    }

    # Global protein composition
    features.update(
        amino_acid_composition(
            sequence,
            "Global_AA"
        )
    )

    # Local sequence window
    window_size = 10

    start = max(0, position - 1 - window_size)
    end = min(
        len(sequence),
        position - 1 + window_size + 1
    )

    local_sequence = sequence[start:end]

    features.update(
        amino_acid_composition(
            local_sequence,
            "Local_AA"
        )
    )

    # Mutation physicochemical properties
    features.update(
        mutation_properties(
            wild,
            mutant
        )
    )

    return features


def main():

    print("Loading labelled mutation dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"UniProt_ID": str}
    )

    print(f"Mutations: {len(df):,}")

    records = []

    skipped = 0

    for index, row in df.iterrows():

        features = build_features(row)

        if features is not None:
            records.append(features)
        else:
            skipped += 1

        if (index + 1) % 1000 == 0:
            print(
                f"Processed: {index + 1:,}/{len(df):,}"
            )

    feature_df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSequence feature extraction complete!")
    print(f"Rows created: {len(feature_df):,}")
    print(f"Rows skipped: {skipped:,}")
    print(f"Number of features: {len(feature_df.columns):,}")

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()