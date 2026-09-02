import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "brca_train_selected.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "brca_blind_test_selected.csv"

RESULT_FILE = PROJECT_ROOT / "results" / "classical_baseline.txt"


def main():

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]

    X_test = test_df.drop(columns=["Label"])
    y_test = test_df["Label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])

    print("Training Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print("\nClassical ML baseline results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_FILE, "w") as f:
        f.write("Logistic Regression Baseline\n")
        f.write("============================\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write(f"ROC-AUC  : {auc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"\nResults saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()