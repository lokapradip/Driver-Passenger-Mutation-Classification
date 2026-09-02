import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "brca_dnn_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "brca_scaler.joblib")

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "brca_labeled_mutations.csv"
)

SEQUENCE_DIR = os.path.join(
    BASE_DIR, "data", "processed", "sequences"
)

STRUCTURE_DIR = os.path.join(
    BASE_DIR, "data", "processed", "structures"
)

FEATURE_FILE = os.path.join(
    BASE_DIR, "data", "processed", "selected_features.txt"
)


# ============================================================
# MODEL
# ============================================================

class DriverPassengerDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.30),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# AMINO ACID PROPERTIES
# ============================================================

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

POLARITY = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3,
    "F": 5.2, "G": 9.0, "H": 10.4, "I": 5.2,
    "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6,
    "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2,
    "T": 8.6, "V": 5.9, "W": 5.4, "Y": 6.2
}

MOLECULAR_WEIGHT = {
    "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13,
    "F": 165.19, "G": 75.07, "H": 155.16, "I": 131.18,
    "K": 146.19, "L": 131.18, "M": 149.21, "N": 132.12,
    "P": 115.13, "Q": 146.15, "R": 174.20, "S": 105.09,
    "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19
}


# ============================================================
# HELPERS
# ============================================================

def load_sequence(uniprot_id):
    path = os.path.join(SEQUENCE_DIR, f"{uniprot_id}.fasta")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequence not found: {uniprot_id}")

    with open(path, "r") as f:
        lines = f.readlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if not line.startswith(">")
    )

    return sequence


def parse_mutation(mutation):
    match = re.match(r"^([A-Z])(\d+)([A-Z])$", mutation.upper())

    if not match:
        raise ValueError(
            "Mutation must look like R175H, e.g. R175H"
        )

    wild = match.group(1)
    position = int(match.group(2))
    mutant = match.group(3)

    return wild, position, mutant


def amino_acid_composition(sequence, prefix):
    length = len(sequence)

    return {
        f"{prefix}_{aa}": sequence.count(aa) / length
        for aa in AA_LIST
    }


def build_sequence_features(sequence, wild, position, mutant):

    if position < 1 or position > len(sequence):
        raise ValueError(
            f"Position {position} is outside protein length {len(sequence)}"
        )

    actual_residue = sequence[position - 1]

    if actual_residue != wild:
        raise ValueError(
            f"Wild-type mismatch: mutation says {wild}{position}{mutant}, "
            f"but sequence contains {actual_residue} at position {position}"
        )

    features = {
        "Protein_Length": len(sequence),
        "Relative_Position": position / len(sequence),
    }

    # Global amino-acid composition
    features.update(
        amino_acid_composition(sequence, "Global_AA")
    )

    # Local sequence window
    window_size = 5

    start = max(0, position - 1 - window_size)
    end = min(len(sequence), position - 1 + window_size + 1)

    local_sequence = sequence[start:end]

    features.update(
        amino_acid_composition(local_sequence, "Local_AA")
    )

    # Mutation properties
    features["Delta_Polarity"] = (
        POLARITY[mutant] - POLARITY[wild]
    )

    features["Delta_Molecular_Weight"] = (
        MOLECULAR_WEIGHT[mutant]
        - MOLECULAR_WEIGHT[wild]
    )

    return features


# ============================================================
# STRUCTURE FEATURES
# ============================================================

def get_structure_features(uniprot_id, position):

    path = os.path.join(
        STRUCTURE_DIR,
        f"AF-{uniprot_id}-F1-model_v4.pdb"
    )

    if not os.path.exists(path):
        # Try alternative naming
        path = os.path.join(
            STRUCTURE_DIR,
            f"{uniprot_id}.pdb"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AlphaFold structure not found for {uniprot_id}"
        )

    target_atoms = []
    all_atoms = []

    with open(path, "r") as f:

        for line in f:

            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                atom_name = line[12:16].strip()
                residue_number = int(line[22:26].strip())

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                b_factor = float(line[60:66])

            except ValueError:
                continue

            atom = (
                residue_number,
                x,
                y,
                z,
                b_factor,
                atom_name
            )

            all_atoms.append(atom)

            if residue_number == position:
                target_atoms.append(atom)

    if not target_atoms:
        raise ValueError(
            f"Residue {position} not found in structure"
        )

    # Prefer CA atom
    ca_atoms = [
        atom for atom in target_atoms
        if atom[5] == "CA"
    ]

    target = ca_atoms[0] if ca_atoms else target_atoms[0]

    tx, ty, tz = target[1], target[2], target[3]

    structure_plddt = target[4]

    residue_coordinates = {}

    for atom in all_atoms:

        residue_number = atom[0]

        if atom[5] == "CA":
            residue_coordinates[residue_number] = atom

    distances = []

    for residue_number, atom in residue_coordinates.items():

        if residue_number == position:
            continue

        x, y, z = atom[1], atom[2], atom[3]

        distance = np.sqrt(
            (x - tx) ** 2 +
            (y - ty) ** 2 +
            (z - tz) ** 2
        )

        if distance <= 8.0:
            distances.append(
                (distance, atom[4])
            )

    if distances:

        neighbor_count = len(distances)

        mean_distance = np.mean(
            [x[0] for x in distances]
        )

        mean_neighbor_plddt = np.mean(
            [x[1] for x in distances]
        )

    else:

        neighbor_count = 0
        mean_distance = 0.0
        mean_neighbor_plddt = 0.0

    # Local pLDDT: nearby residues within 8 Å
    local_plddt_values = [
        structure_plddt
    ]

    local_plddt_values.extend(
        [x[1] for x in distances]
    )

    local_mean_plddt = np.mean(
        local_plddt_values
    )

    return {
        "Structure_pLDDT": structure_plddt,
        "Local_Mean_pLDDT": local_mean_plddt,
        "Neighbor_Count_8A": neighbor_count,
        "Mean_Neighbor_Distance_8A": mean_distance,
        "Mean_Neighbor_pLDDT": mean_neighbor_plddt,
    }


# ============================================================
# FIND UNIPROT
# ============================================================

def find_uniprot(gene, mutation):

    df = pd.read_csv(DATA_PATH)

    matches = df[
        (df["Gene"].astype(str).str.upper() == gene.upper()) &
        (df["Mutation"].astype(str).str.upper() == mutation.upper())
    ]

    if matches.empty:
        raise ValueError(
            f"{gene} {mutation} was not found in the BRCA reference dataset."
        )

    return str(matches.iloc[0]["UniProt_ID"])


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict(gene, mutation):

    print("\n========================================")
    print("BRCA DRIVER / PASSENGER PREDICTION")
    print("========================================")

    print(f"Gene     : {gene}")
    print(f"Mutation : {mutation}")

    # Find UniProt
    uniprot_id = find_uniprot(gene, mutation)

    print(f"UniProt  : {uniprot_id}")

    # Parse mutation
    wild, position, mutant = parse_mutation(mutation)

    # Sequence
    sequence = load_sequence(uniprot_id)

    print(f"Protein length : {len(sequence)}")

    # Sequence features
    features = build_sequence_features(
        sequence,
        wild,
        position,
        mutant
    )

    # Structure features
    structure_features = get_structure_features(
        uniprot_id,
        position
    )

    # Add structure features explicitly.
    # This avoids feature-name/key mismatches during prediction.
    features["Structure_pLDDT"] = float(
        structure_features["Structure_pLDDT"]
    )
    features["Local_Mean_pLDDT"] = float(
        structure_features["Local_Mean_pLDDT"]
    )
    features["Neighbor_Count_8A"] = float(
        structure_features["Neighbor_Count_8A"]
    )
    features["Mean_Neighbor_Distance_8A"] = float(
        structure_features["Mean_Neighbor_Distance_8A"]
    )
    features["Mean_Neighbor_pLDDT"] = float(
        structure_features["Mean_Neighbor_pLDDT"]
    )

    # Load selected features
    with open(FEATURE_FILE, "r", encoding="utf-8-sig") as f:
        selected_features = [
            line.strip().replace("\ufeff", "")
            for line in f
            if line.strip()
        ]

    # Normalize feature names without changing their meaning.
    selected_features = [name.strip() for name in selected_features]

    # Debug information
    print("\nStructure features:")
    for key, value in structure_features.items():
        print(f"  {key}: {value}")

    print("\nSelected features:")
    print(selected_features)

    # Make sure every feature exists
    missing = [
        feature
        for feature in selected_features
        if feature not in features
    ]

    if missing:
        print("\nAvailable feature keys:")
        print(list(features.keys()))
        raise ValueError(
            f"Missing features: {missing}"
        )

    X = np.array(
        [[features[f] for f in selected_features]],
        dtype=np.float32
    )

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    X_scaled = scaler.transform(X)

    # Load model
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    input_size = checkpoint["input_size"]

    model = DriverPassengerDNN(input_size)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Prediction
    with torch.no_grad():

        tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        logit = model(tensor)

        probability = torch.sigmoid(
            logit
        ).item()

    prediction = (
        "Driver"
        if probability >= 0.5
        else "Passenger"
    )

    confidence = (
        probability
        if prediction == "Driver"
        else 1 - probability
    )

    print("\n----------------------------------------")
    print(f"Prediction : {prediction}")
    print(f"Driver probability : {probability:.4f}")
    print(f"Confidence : {confidence:.2%}")
    print("----------------------------------------")

    return prediction, probability


# ============================================================
# INTERACTIVE MODE
# ============================================================

if __name__ == "__main__":

    gene = input("Enter Gene (example: TP53): ").strip()
    mutation = input("Enter Mutation (example: R175H): ").strip()

    try:
        predict(gene, mutation)

    except Exception as e:

        print("\nERROR:")
        print(e)