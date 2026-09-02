import pandas as pd
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "brca_train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "brca_blind_test.csv"

SELECTED_TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "brca_train_selected.csv"
SELECTED_TEST_FILE = PROJECT_ROOT / "data" / "processed" / "brca_blind_test_selected.csv"
FEATURE_LIST_FILE = PROJECT_ROOT / "data" / "processed" / "selected_features.txt"


IDENTIFIER_COLUMNS = [
    "Gene",
    "Mutation",
    "Protein_Position",
    "UniProt_ID",
    "Wild_AA",
    "Mutant_AA",
    "Sequence_Residue",
]


def main():

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    feature_columns = [
        col for col in train_df.columns
        if col not in IDENTIFIER_COLUMNS + ["Label"]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["Label"]

    X_test = test_df[feature_columns]
    y_test = test_df["Label"]

    # Select the 30 most informative features using TRAINING data only
    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=min(30, len(feature_columns))
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = [
        feature_columns[i]
        for i in selector.get_support(indices=True)
    ]

    train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_features
    )

    train_selected["Label"] = y_train.to_numpy()

    test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_features
    )

    test_selected["Label"] = y_test.to_numpy()

    train_selected.to_csv(
        SELECTED_TRAIN_FILE,
        index=False
    )

    test_selected.to_csv(
        SELECTED_TEST_FILE,
        index=False
    )

    with open(FEATURE_LIST_FILE, "w") as f:
        for feature in selected_features:
            f.write(feature + "\n")

    print("Feature selection complete!")
    print(f"Original features: {len(feature_columns)}")
    print(f"Selected features: {len(selected_features)}")

    print("\nSelected features:")
    for feature in selected_features:
        print(feature)

    print(f"\nTraining rows: {len(train_selected)}")
    print(f"Blind test rows: {len(test_selected)}")

    print(f"\nSaved training data: {SELECTED_TRAIN_FILE}")
    print(f"Saved blind test data: {SELECTED_TEST_FILE}")
    print(f"Saved feature list: {FEATURE_LIST_FILE}")


if __name__ == "__main__":
    main()
