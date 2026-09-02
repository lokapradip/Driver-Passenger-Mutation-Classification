import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Paths
# -----------------------------
SEQUENCE_FILE = "data/processed/brca_sequence_features.csv"
TRAIN_SPLIT = "data/processed/brca_train.csv"
TEST_SPLIT = "data/processed/brca_blind_test.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
sequence_df = pd.read_csv(SEQUENCE_FILE)
train_split = pd.read_csv(TRAIN_SPLIT)
test_split = pd.read_csv(TEST_SPLIT)

# Use the exact protein split already created
train_ids = set(train_split["UniProt_ID"])
test_ids = set(test_split["UniProt_ID"])

train_df = sequence_df[
    sequence_df["UniProt_ID"].isin(train_ids)
].copy()

test_df = sequence_df[
    sequence_df["UniProt_ID"].isin(test_ids)
].copy()

print("Sequence-only training rows:", len(train_df))
print("Sequence-only blind-test rows:", len(test_df))

# -----------------------------
# Target
# -----------------------------
y_train = train_df["Label"].astype(int)
y_test = test_df["Label"].astype(int)

# -----------------------------
# Numeric sequence features only
# -----------------------------
exclude_columns = [
    "Label",
    "UniProt_ID",
    "Gene",
    "Mutation",
    "Wild_AA",
    "Mutant_AA",
    "Sequence_Residue",
]

feature_columns = [
    col
    for col in train_df.columns
    if col not in exclude_columns
    and pd.api.types.is_numeric_dtype(train_df[col])
]

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

print("Sequence features:", len(feature_columns))
print("Features used:")
print(feature_columns)

# -----------------------------
# Standardization
# Fit ONLY on training data
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.float32
).view(-1, 1)

# -----------------------------
# DNN
# -----------------------------
class SequenceOnlyDNN(nn.Module):

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
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = SequenceOnlyDNN(X_train.shape[1])

# -----------------------------
# Class imbalance
# -----------------------------
positive = y_train.sum().item()
negative = len(y_train) - positive

pos_weight = torch.tensor(
    [negative / positive]
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# -----------------------------
# Training
# -----------------------------
epochs = 100
batch_size = 64

model.train()

for epoch in range(epochs):

    indices = torch.randperm(len(X_train))
    total_loss = 0.0

    for start in range(0, len(X_train), batch_size):

        batch_idx = indices[start:start + batch_size]

        xb = X_train[batch_idx]
        yb = y_train[batch_idx]

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:

        avg_loss = total_loss / max(
            1,
            len(X_train) // batch_size
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {avg_loss:.4f}"
        )

# -----------------------------
# Evaluation
# -----------------------------
model.eval()

with torch.no_grad():

    logits = model(X_test)

    probabilities = (
        torch.sigmoid(logits)
        .numpy()
        .ravel()
    )

predictions = (
    probabilities >= 0.5
).astype(int)

y_true = (
    y_test.numpy()
    .ravel()
    .astype(int)
)

accuracy = accuracy_score(
    y_true,
    predictions
)

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

cm = confusion_matrix(
    y_true,
    predictions
)

# -----------------------------
# Results
# -----------------------------
print("\n==============================")
print("SEQUENCE-ONLY DNN RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Save
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
        "features": feature_columns,
    },
    "models/brca_dnn_sequence_only.pt"
)

with open(
    "results/dnn_sequence_only_results.txt",
    "w"
) as f:

    f.write("BRCA Sequence-Only DNN\n")
    f.write("======================\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

print("\nModel saved:")
print("models/brca_dnn_sequence_only.pt")

print("Results saved:")
print("results/dnn_sequence_only_results.txt")