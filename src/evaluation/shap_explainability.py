import os
import sys
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dnn_model import DriverPassengerDNN


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "brca_dnn_model.pt")
TEST_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "brca_blind_test_selected.csv"
)
FEATURE_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "selected_features.txt"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD FEATURES
# ============================================================

with open(FEATURE_PATH, "r") as f:
    feature_columns = [line.strip() for line in f if line.strip()]

print("Number of selected features:", len(feature_columns))


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_PATH)

X_test = test_df[feature_columns].astype(float)
y_test = test_df["Label"].astype(int)

print("Blind-test rows:", len(X_test))


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

input_size = checkpoint["input_size"]

model = DriverPassengerDNN(input_size=input_size)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def predict(X):
    X = np.asarray(X, dtype=np.float32)

    with torch.no_grad():
        tensor = torch.tensor(X)
        output = model(tensor)

        # Handle sigmoid output
        if output.ndim > 1 and output.shape[1] == 1:
            output = output.squeeze(1)

        return output.numpy()


# ============================================================
# SHAP BACKGROUND DATA
# ============================================================

# Use a small representative background dataset
background_size = min(100, len(X_test))

background = X_test.sample(
    n=background_size,
    random_state=42
).values.astype(np.float32)


# ============================================================
# EXPLAIN TEST SAMPLES
# ============================================================

explain_size = min(200, len(X_test))

X_explain = X_test.iloc[:explain_size].values.astype(np.float32)

print("Calculating SHAP values...")
print("Samples explained:", explain_size)

explainer = shap.KernelExplainer(
    predict,
    background
)

shap_values = explainer.shap_values(X_explain)

# SHAP can return a list for some model/output combinations
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance_path = os.path.join(
    RESULTS_DIR,
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\nTOP SHAP FEATURES")
print("=================")

print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_columns,
    show=False
)

plt.tight_layout()

summary_path = os.path.join(
    RESULTS_DIR,
    "shap_summary_plot.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_columns,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_path = os.path.join(
    RESULTS_DIR,
    "shap_feature_importance.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "shap_results.txt"
)

with open(report_path, "w") as f:

    f.write("BRCA DNN SHAP EXPLAINABILITY\n")
    f.write("============================\n\n")

    f.write(
        f"Number of selected features: {len(feature_columns)}\n"
    )

    f.write(
        f"Blind-test samples explained: {explain_size}\n\n"
    )

    f.write("Top 15 features by mean absolute SHAP value:\n\n")

    f.write(
        importance_df.head(15).to_string(index=False)
    )

    f.write("\n")


# ============================================================
# COMPLETE
# ============================================================

print("\nSHAP analysis completed successfully!")

print("Saved:")
print(importance_path)
print(summary_path)
print(bar_path)
print(report_path)