import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

data = {
    "Model": [
        "Sequence-only DNN",
        "Structure-only DNN",
        "Sequence + Structure DNN"
    ],
    "Accuracy": [0.6935, 0.7113, 0.7306],
    "Precision": [0.7502, 0.7366, 0.7400],
    "Recall": [0.7739, 0.8690, 0.9063],
    "F1-score": [0.7619, 0.7974, 0.8148],
    "ROC-AUC": [0.7266, 0.7452, 0.7246]
}

df = pd.DataFrame(data)

print("\n==============================")
print("ABLATION STUDY")
print("==============================")
print(df.to_string(index=False))

# Save table
df.to_csv(
    "results/ablation_comparison.csv",
    index=False
)

# Save text report
with open(
    "results/ablation_comparison.txt",
    "w"
) as f:
    f.write("BRCA DNN Ablation Study\n")
    f.write("=======================\n\n")
    f.write(df.to_string(index=False))
    f.write("\n")

# Create comparison chart
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

for metric in metrics:
    plt.figure(figsize=(8, 5))

    plt.bar(
        df["Model"],
        df[metric]
    )

    plt.ylabel(metric)
    plt.title(f"Ablation Study - {metric}")
    plt.ylim(0, 1)

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    filename = (
        "results/"
        + metric.lower().replace("-", "_")
        + "_comparison.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

print("\nSaved:")
print("results/ablation_comparison.csv")
print("results/ablation_comparison.txt")
print("Comparison charts saved in results/")