import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "brca_train_selected.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "brca_blind_test_selected.csv"

TRAIN_ARFF = PROJECT_ROOT / "data" / "processed" / "brca_train_selected.arff"
TEST_ARFF = PROJECT_ROOT / "data" / "processed" / "brca_blind_test_selected.arff"


def save_arff(df, output_file, relation_name):

    feature_columns = [c for c in df.columns if c != "Label"]

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(f"@RELATION {relation_name}\n\n")

        for col in feature_columns:
            safe_name = col.replace(" ", "_").replace("-", "_")
            f.write(f"@ATTRIBUTE {safe_name} NUMERIC\n")

        f.write("@ATTRIBUTE Label {0,1}\n\n")
        f.write("@DATA\n")

        for _, row in df.iterrows():
            values = [str(row[col]) for col in feature_columns]
            values.append(str(int(row["Label"])))
            f.write(",".join(values) + "\n")


def main():

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    save_arff(train, TRAIN_ARFF, "BRCA_Train")
    save_arff(test, TEST_ARFF, "BRCA_Blind_Test")

    print("Weka files created successfully!")
    print(f"Training ARFF: {TRAIN_ARFF}")
    print(f"Blind test ARFF: {TEST_ARFF}")
    print(f"Training rows: {len(train)}")
    print(f"Blind test rows: {len(test)}")


if __name__ == "__main__":
    main()