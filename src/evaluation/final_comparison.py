import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FINAL EXPERIMENTAL RESULTS
# ============================================================

results = [
    {
        "Model": "Classical Baseline",
        "Accuracy": 0.7394,
        "Precision": 0.7363,
        "Recall": 0.9369,
        "F1-score": 0.8246,
        "ROC-AUC": 0.8456
    },
    {
        "Model": "Weka J48",
        "Accuracy": 0.6925,
        "Precision": 0.6750,
        "Recall": 0.6930,
        "F1-score": 0.6610,
        "ROC-AUC": 0.5850
    },
    {
        "Model": "Sequence-only DNN",
        "Accuracy": 0.6935,
        "Precision": 0.7502,
        "Recall": 0.7739,
        "F1-score": 0.7619,
        "ROC-AUC": 0.7266
    },
    {
        "Model": "Structure-only DNN",
        "Accuracy": 0.7113,
        "Precision": 0.7366,
        "Recall": 0.8690,
        "F1-score": 0.7974,
        "ROC-AUC": 0.7452
    },
    {
        "Model": "Sequence + Structure DNN",
        "Accuracy": 0.7306,
        "Precision": 0.7400,
        "Recall": 0.9063,
        "F1-score": 0.8148,
        "ROC-AUC": 0.7246
    }
]


df = pd.DataFrame(results)

# Save CSV
csv_path = os.path.join(
    RESULTS_DIR,
    "final_model_comparison.csv"
)

df.to_csv(csv_path, index=False)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

txt_path = os.path.join(
    RESULTS_DIR,
    "final_model_comparison.txt"
)

with open(txt_path, "w") as f:

    f.write("FINAL MODEL COMPARISON\n")
    f.write("======================\n\n")

    f.write(df.to_string(index=False))

    f.write("\n\n")
    f.write("KEY FINDINGS\n")
    f.write("============\n")

    f.write(
        "\nThe Structure-only DNN achieved the highest DNN F1-score "
        "(0.7974) among the individual biological information sources.\n"
    )

    f.write(
        "\nThe Sequence + Structure DNN achieved an F1-score of 0.8148 "
        "and recall of 0.9063 on the blind test set.\n"
    )

    f.write(
        "\nThe ablation study demonstrates that sequence and structural "
        "information provide complementary predictive information, "
        "although their combination does not maximize every metric.\n"
    )

    f.write(
        "\nSHAP analysis was used to identify the biological features "
        "that contributed most strongly to the combined DNN predictions.\n"
    )


# ============================================================
# COMPARISON CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(10, 6))

    plt.bar(df["Model"], df[metric])

    plt.ylabel(metric)
    plt.xlabel("Model")
    plt.title(f"Model Comparison — {metric}")

    plt.xticks(
        rotation=25,
        ha="right"
    )

    plt.ylim(0, 1)

    plt.tight_layout()

    filename = metric.lower().replace("-", "_")

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"final_{filename}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nFINAL MODEL COMPARISON COMPLETED")
print("================================")

print("\n")
print(df.to_string(index=False))

print("\nSaved:")
print(csv_path)
print(txt_path)
print("Comparison charts saved in results/")