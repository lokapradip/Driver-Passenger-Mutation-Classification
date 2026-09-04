import os
import pandas as pd
import streamlit as st
import torch
import joblib


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR, "data", "processed", "brca_combined_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR, "models", "brca_dnn_model.pt"
)

SCALER_FILE = os.path.join(
    BASE_DIR, "models", "brca_scaler.joblib"
)

FEATURE_FILE = os.path.join(
    BASE_DIR, "data", "processed", "selected_features.txt"
)


# -----------------------------
# DNN Architecture
# -----------------------------
class MutationDNN(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.30),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.30),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_model():

    features = [
        line.strip()
        for line in open(FEATURE_FILE)
        if line.strip()
    ]

    scaler = joblib.load(SCALER_FILE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    model = MutationDNN(len(features))

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, scaler, features


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Cancer Mutation Classifier",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Cancer Driver / Passenger Mutation Classifier")

st.write(
    "Predict whether a supported BRCA missense mutation "
    "is likely to be a Driver or Passenger using a trained "
    "deep neural network."
)

st.divider()


# -----------------------------
# Input
# -----------------------------
gene = st.text_input(
    "Gene",
    placeholder="Example: MUC4"
)

mutation = st.text_input(
    "Mutation",
    placeholder="Example: T1022A"
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔬 Predict Mutation", use_container_width=True):

    if not gene or not mutation:

        st.warning("Please enter both Gene and Mutation.")

    else:

        try:

            df = load_data()
            model, scaler, features = load_model()

            gene_input = gene.strip().upper()
            mutation_input = mutation.strip().upper()

            matches = df[
                (df["Gene"].astype(str).str.upper() == gene_input)
                &
                (df["Mutation"].astype(str).str.upper() == mutation_input)
            ]

            if matches.empty:

                st.error(
                    "This mutation is not available in the "
                    "supported BRCA reference dataset."
                )

            else:

                row = matches.iloc[0]

                X = row[features].to_frame().T

                X_scaled = scaler.transform(X)

                X_tensor = torch.tensor(
                    X_scaled,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    output = model(X_tensor)

                    probability = torch.sigmoid(output).item()

                if probability >= 0.5:

                    prediction = "Driver"

                else:

                    prediction = "Passenger"

                confidence = (
                    probability
                    if prediction == "Driver"
                    else 1 - probability
                )

                st.success(
                    f"Prediction: **{prediction}**"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                st.info(
                    f"Driver probability: {probability * 100:.2f}%"
                )

        except Exception as e:

            st.error(f"Prediction error: {e}")


st.divider()

st.caption(
    "BRCA cancer mutation classification using protein sequence "
    "and AlphaFold-derived structural features."
)

st.caption(
    "Model: Deep Neural Network | "
    "Evaluation: Protein-level leakage-safe split"
)