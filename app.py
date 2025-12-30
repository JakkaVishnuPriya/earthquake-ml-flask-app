import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import uuid

# ---------------------------
# Load model, scaler, features
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("earthquake_rf_model.pkl")
    scaler = joblib.load("earthquake_scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ---------------------------
# Alert mappings & colors
# ---------------------------
alert_to_class = {
    "green": 0,
    "yellow": 1,
    "orange": 2,
    "red": 3
}
class_to_alert = {v: k for k, v in alert_to_class.items()}

ALERT_COLORS = {
    "green": "#10b981",
    "yellow": "#f59e0b",
    "orange": "#f97316",
    "red": "#ef4444"
}

# ---------------------------
# Session (login / signup)
# ---------------------------
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "users" not in st.session_state:
        # demo user store: {username: password}
        st.session_state.users = {"demo_user": "demo123"}  # same demo user as Flask

def signup_block():
    st.subheader("Signup")
    new_user = st.text_input("New username")
    new_pass = st.text_input("New password", type="password")
    if st.button("Register"):
        if not new_user or not new_pass:
            st.error("Username and password cannot be empty.")
        elif new_user in st.session_state.users:
            st.error("Username already exists. Try another.")
        else:
            st.session_state.users[new_user] = new_pass
            st.success(f"User {new_user} registered successfully! Please login.")
            st.info("Go to the Login tab in the sidebar.")

def login_block():
    st.subheader("Login")
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Log in"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials. Please try again.")

def logout_block():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.info("You have been logged out.")

# ---------------------------
# Prediction logic (Flask /make_prediction equivalent)
# ---------------------------
def prediction_page(free_mode=False):
    st.title("🌍 Earthquake Alert Prediction")

    if st.session_state.logged_in and not free_mode:
        st.markdown(f"**Welcome, {st.session_state.username}!**")

    st.markdown(
        "Enter earthquake parameters to predict the alert level using the trained Random Forest model."
    )

    col1, col2 = st.columns(2)

    with col1:
        magnitude = st.number_input(
            "Magnitude",
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
            "CDI (0–12)",
            min_value=0.0,
            max_value=12.0,
            value=5.0,
            step=0.1
        )
        mmi = st.number_input(
            "MMI (0–12)",
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

    if st.button("🔍 Predict"):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prediction_id = str(uuid.uuid4())[:8]

            input_data = pd.DataFrame([{
                "magnitude": magnitude,
                "depth": depth,
                "cdi": cdi,
                "mmi": mmi,
                "sig": sig
            }])

            # same feature ordering as training
            input_data = input_data[feature_names]

            scaled_input = scaler.transform(input_data)
            predicted_class = int(model.predict(scaled_input)[0])
            probabilities = model.predict_proba(scaled_input)[0]

            predicted_alert = class_to_alert[predicted_class]
            predicted_color = ALERT_COLORS[predicted_alert]

            prob_display = {
                class_to_alert[i].capitalize(): f"{p*100:.2f}%"
                for i, p in enumerate(probabilities)
            }

            # Result "card" like Flask result.html
            st.subheader("Prediction Result")
            st.markdown(
                f"""
                <div style="
                    padding:1rem;
                    border-radius:0.5rem;
                    background-color:{predicted_color};
                    color:white;
                    font-weight:bold;
                    text-align:center;">
                    ALERT LEVEL: {predicted_alert.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write(f"Prediction ID: `{prediction_id}`")
            st.write(f"Time: `{current_time}`")
            st.write("Probabilities by alert level:")
            st.json(prob_display)

            # Probability bar chart (matplotlib)
            fig, ax = plt.subplots(figsize=(6, 3))
            alerts = [class_to_alert[i].capitalize() for i in range(len(probabilities))]
            plot_colors = {
                "Green": "#10b981",
                "Yellow": "#f59e0b",
                "Orange": "#f97316",
                "Red": "#ef4444"
            }
            colors = [plot_colors[a] for a in alerts]

            ax.bar(alerts, probabilities, color=colors)
            ax.set_title("Prediction Probabilities", fontsize=12, color="black")
            ax.set_xlabel("Alert Level", fontsize=10, color="black")
            ax.set_ylabel("Probability", fontsize=10, color="black")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", colors="black")
            ax.tick_params(axis="y", colors="black")

            for i, p in enumerate(probabilities):
                ax.text(i, p + 0.02, f"{p*100:.1f}%", ha="center", va="bottom",
                        fontsize=9, color="black")

            st.pyplot(fig)

            st.info(
                "This prediction is for decision support only and does not replace "
                "official earthquake warning systems."
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ---------------------------
# Dashboard (Flask /dashboard)
# ---------------------------
def dashboard_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access the dashboard.")
        return

    st.title("Dashboard")
    st.write(f"Welcome to your dashboard, **{st.session_state.username}**.")
    st.write("Extend this area with history, analytics, etc.")

# ---------------------------
# Main (routes -> pages)
# ---------------------------
def main():
    st.set_page_config(
        page_title="Earthquake Alert App",
        page_icon="🌍",
        layout="centered"
    )

    init_session_state()

    # Sidebar: login / logout / signup + navigation
    st.sidebar.title("User")
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        logout_block()
    else:
        auth_tab = st.sidebar.radio("Auth", ["Login", "Signup"])
        if auth_tab == "Login":
            login_block()
        else:
            signup_block()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["Home", "Free Prediction", "Dashboard"])

    if page == "Home":
        st.title("Earthquake Alert Web App")
        st.write(
            "Use the sidebar to log in or sign up, and go to the prediction pages "
            "to estimate earthquake alert levels."
        )
    elif page == "Free Prediction":
        prediction_page(free_mode=True)
    elif page == "Dashboard":
        dashboard_page()
        st.markdown("---")
        st.write("You can still run predictions from here:")
        prediction_page(free_mode=False)

if __name__ == "__main__":
    main()
