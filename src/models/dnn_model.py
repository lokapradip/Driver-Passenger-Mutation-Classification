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
TRAIN_FILE = "data/processed/brca_train_selected.csv"
TEST_FILE = "data/processed/brca_blind_test_selected.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Label is the target
X_train = train_df.drop(columns=["Label"])
y_train = train_df["Label"].astype(int)

X_test = test_df.drop(columns=["Label"])
y_test = test_df["Label"].astype(int)

# Keep only numeric biological features
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test[X_train.columns]

print("Training shape:", X_train.shape)
print("Blind test shape:", X_test.shape)

# -----------------------------
# Standardization
# Fit ONLY on training data
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -----------------------------
# DNN architecture
# -----------------------------
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
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = DriverPassengerDNN(X_train.shape[1])

# -----------------------------
# Handle class imbalance
# -----------------------------
positive = y_train.sum().item()
negative = len(y_train) - positive

pos_weight = torch.tensor([negative / positive])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        avg_loss = total_loss / max(1, len(X_train) // batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

# -----------------------------
# Blind-test evaluation
# -----------------------------
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits).numpy().ravel()

predictions = (probabilities >= 0.5).astype(int)

y_true = y_test.numpy().ravel().astype(int)

accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, zero_division=0)
recall = recall_score(y_true, predictions, zero_division=0)
f1 = f1_score(y_true, predictions, zero_division=0)
roc_auc = roc_auc_score(y_true, probabilities)

cm = confusion_matrix(y_true, predictions)

# -----------------------------
# Print results
# -----------------------------
print("\n==============================")
print("DNN BLIND TEST RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Save model
# -----------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
    },
    os.path.join(MODEL_DIR, "brca_dnn_model.pt")
)

# -----------------------------
# Save results
# -----------------------------
with open(
    os.path.join(RESULT_DIR, "dnn_results.txt"),
    "w"
) as f:

    f.write("BRCA DNN Blind Test Results\n")
    f.write("===========================\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n")

print("\nModel saved to:", os.path.join(MODEL_DIR, "brca_dnn_model.pt"))
print("Results saved to:", os.path.join(RESULT_DIR, "dnn_results.txt"))