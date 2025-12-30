import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------------------
# 1. Load model artifacts
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("earthquake_rf_model.pkl")
    scaler = joblib.load("earthquake_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")  # ['magnitude','depth','cdi','mmi','sig']
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_scaler()

class_to_alert = {
    0: "GREEN",
    1: "YELLOW",
    2: "ORANGE",
    3: "RED"
}

alert_descriptions = {
    "GREEN": "Low impact. No immediate action required.",
    "YELLOW": "Moderate impact. Stay alert and follow updates.",
    "ORANGE": "High impact. Prepare for strong shaking and possible damage.",
    "RED": "Severe impact. Expect very strong shaking and significant damage; follow emergency instructions."
}

# ---------------------------
# 2. Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="Earthquake Alert Predictor",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Earthquake Alert Level Prediction")
st.write("Enter earthquake parameters to predict the **alert level** using a trained Random Forest model.[page:0]")

st.markdown("---")

# Sidebar info
st.sidebar.header("About the Model")
st.sidebar.write("- Algorithm: RandomForestClassifier (500 trees, max_depth=15).[page:0]")
st.sidebar.write("- Features used: magnitude, depth, cdi, mmi, sig.[page:0]")
st.sidebar.write("- Trained on balanced earthquake alert dataset (classes 0–3).[page:0]")

# ---------------------------
# 3. Input widgets
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    magnitude = st.number_input(
        "Magnitude (e.g., 5.0 – 9.5)",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )
    depth = st.number_input(
        "Depth (km)",
        min_value=0.0,
        max_value=700.0,
        value=10.0,
        step=1.0
    )

with col2:
    cdi = st.number_input(
        "CDI (Community Intensity, 0–12)",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.1
    )
    mmi = st.number_input(
        "MMI (Modified Mercalli Intensity, 0–12)",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.1
    )

sig = st.number_input(
    "Significance (sig)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=10.0
)

st.markdown("---")

# ---------------------------
# 4. Prediction logic
# ---------------------------
if st.button("🔍 Predict Alert Level"):
    # Build input DataFrame using the same feature order as training
    input_data = pd.DataFrame([{
        "magnitude": magnitude,
        "depth": depth,
        "cdi": cdi,
        "mmi": mmi,
        "sig": sig
    }])

    # Reorder columns to match training (safety)
    input_data = input_data[feature_names]

    # Scale features
    scaled_input = scaler.transform(input_data)

    # Predict class and probabilities
    pred_class = int(model.predict(scaled_input)[0])
    pred_proba = model.predict_proba(scaled_input)[0]

    alert_label = class_to_alert.get(pred_class, "UNKNOWN")

    st.subheader("Prediction Result")
    st.markdown(f"**Predicted Alert Level:** `{alert_label}`")
    st.write(alert_descriptions.get(alert_label, ""))

    # Show probabilities
    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Alert Level": [class_to_alert[i] for i in range(len(pred_proba))],
        "Probability": np.round(pred_proba, 4)
    })
    st.table(prob_df)

    # Small risk interpretation
    st.info(
        "This prediction is based on the trained model and input parameters; "
        "use it as decision **support**, not as a replacement for official seismic alerts."
    )

# ---------------------------
# 5. Footer
# ---------------------------
st.markdown("---")
st.caption("Earthquake alert prediction model deployed with Streamlit using a Random Forest trained in Colab.[page:0]")
