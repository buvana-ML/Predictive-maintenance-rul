import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# ----------------------------
# LSTM MODEL CLASS
# ----------------------------

class LSTMRULModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMRULModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        output = self.fc(last_output)

        return output.squeeze()


# ----------------------------
# STREAMLIT UI
# ----------------------------

st.title("Predictive Maintenance System")

st.subheader(
    "Remaining Useful Life Prediction using Deep Learning"
)

uploaded_file = st.file_uploader(
    "Upload engine sensor CSV file",
    type=["csv"]
)

st.write("Current directory:", os.getcwd())

if uploaded_file is not None:

    # ----------------------------
    # LOAD CSV
    # ----------------------------

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")

    st.dataframe(data.head())

    st.write("Processing sensor data...")

    # ----------------------------
    # LOAD SCALER
    # ----------------------------

    scaler = joblib.load("../models/scaler.pkl")

    # ----------------------------
    # REMOVE UNUSED COLUMNS
    # ----------------------------

    drop_cols = [
        "setting_3",
        "sensor_1",
        "sensor_5",
        "sensor_6",
        "sensor_10",
        "sensor_16",
        "sensor_18",
        "sensor_19"
    ]

    data = data.drop(
        columns=drop_cols,
        errors="ignore"
    )

    # ----------------------------
    # FEATURE COLUMNS
    # ----------------------------

    feature_cols = list(
        scaler.feature_names_in_
    )

    # ----------------------------
    # SCALE DATA
    # ----------------------------

    data[feature_cols] = scaler.transform(
        data[feature_cols]
    )

    # ----------------------------
    # CREATE LAST 30-CYCLE WINDOW
    # ----------------------------

    WINDOW_SIZE = 30

    features = data[
        feature_cols
    ].values

    if len(features) >= WINDOW_SIZE:

        sequence = features[-WINDOW_SIZE:]

    else:

        padding = np.zeros(
            (
                WINDOW_SIZE - len(features),
                len(feature_cols)
            )
        )

        sequence = np.vstack(
            (
                padding,
                features
            )
        )

    # ----------------------------
    # CONVERT TO TENSOR
    # ----------------------------

    sequence = np.expand_dims(
        sequence,
        axis=0
    )

    sequence_tensor = torch.tensor(
        sequence,
        dtype=torch.float32
    )

    # ----------------------------
    # LOAD MODEL
    # ----------------------------

    input_size = len(feature_cols)

    model = LSTMRULModel(
        input_size=input_size
    )

    model.load_state_dict(
        torch.load(
           "../models/lstm_rul_model.pth",
           map_location="cpu"
        )
    )

    model.eval()

    # ----------------------------
    # PREDICT
    # ----------------------------

    with torch.no_grad():

        prediction = model(
            sequence_tensor
        )

    predicted_rul = prediction.item()

    predicted_rul = max(
        predicted_rul,
        0
    )

    # ----------------------------
    # DISPLAY RESULT
    # ----------------------------

    st.metric(
        "Predicted Remaining Useful Life",
        f"{predicted_rul:.2f} cycles"
    )

    # ----------------------------
    # HEALTH STATUS
    # ----------------------------

    if predicted_rul > 80:

        st.success(
            "Machine Status: Healthy"
        )

    elif predicted_rul > 40:

        st.warning(
            "Machine Status: Warning"
        )

    else:

        st.error(
            "Machine Status: Critical"
        )