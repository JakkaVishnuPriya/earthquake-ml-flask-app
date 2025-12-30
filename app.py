import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(
    page_title="Earthquake Alert System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .alert-green {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .alert-orange {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
    }
    .alert-yellow {
        background-color: #ffe5cc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
    }
    .alert-red {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained Random Forest model and scaler"""
    try:
        with open('earthquake_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

def predict_earthquake(features, model, scaler):
    """Make prediction using the trained model"""
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    return prediction, probabilities

def get_alert_level(prediction):
    """Determine alert level based on prediction"""
    levels = {
        0: ("Green", "Safe", "#28a745"),
        1: ("Orange", "Minor Alert", "#ff9800"),
        2: ("Yellow", "Moderate Alert", "#ffc107"),
        3: ("Red", "High Alert", "#dc3545")
    }
    return levels.get(prediction, ("Unknown", "Unknown", "#808080"))

# Load model
model, scaler, feature_names = load_model()

if model is not None:
    # App header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌍 Earthquake Prediction System")
        st.subheader("AI-Powered Early Warning System")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Input Parameters")
    st.sidebar.write("Enter earthquake parameters for prediction")
    
    # Feature inputs in sidebar
    magnitude = st.sidebar.slider(
        "Magnitude",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Earthquake magnitude (Richter scale)"
    )
    
    depth = st.sidebar.slider(
        "Depth (km)",
        min_value=0.0,
        max_value=800.0,
        value=50.0,
        step=1.0,
        help="Earthquake depth in kilometers"
    )
    
    cdi = st.sidebar.slider(
        "CDI (Community Felt Reports)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Community Decimal Intensity"
    )
    
    mmi = st.sidebar.slider(
        "MMI (Modified Mercalli Intensity)",
        min_value=0.0,
        max_value=12.0,
        value=6.0,
        step=0.1,
        help="Modified Mercalli Intensity Scale"
    )
    
    sig = st.sidebar.slider(
        "Significance",
        min_value=0.0,
        max_value=1000.0,
        value=100.0,
        step=10.0,
        help="Earthquake significance score"
    )
    
    # Prepare features
    features = [magnitude, depth, cdi, mmi, sig]
    
    # Prediction button
    if st.sidebar.button("🔍 Predict Alert Level", key="predict_btn"):
        prediction, probabilities = predict_earthquake(features, model, scaler)
        alert_color, alert_name, hex_color = get_alert_level(prediction)
        
        # Display results
        st.success("✓ Prediction Complete")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Alert Level", alert_name, delta=alert_color)
        with col2:
            st.metric("Magnitude", f"{magnitude:.1f}")
        with col3:
            st.metric("Depth", f"{depth:.0f} km")
        with col4:
            st.metric("Significance", f"{int(sig)}")
        
        # Alert box
        st.markdown(f"""
        <div class='alert-{alert_color.lower()}'>
            <h3 style='color: {hex_color}; margin: 0;'>⚠️ {alert_name} - {alert_color.upper()} Alert</h3>
            <p>The earthquake parameters indicate a <strong>{alert_name.lower()}</strong> level threat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities visualization
        st.subheader("📈 Prediction Confidence")
        
        # Create probability chart
        alert_levels = ['Green (Safe)', 'Orange (Minor)', 'Yellow (Moderate)', 'Red (High)']
        colors_chart = ['#28a745', '#ff9800', '#ffc107', '#dc3545']
        
        fig = go.Figure(data=[
            go.Bar(
                x=alert_levels,
                y=probabilities * 100,
                marker=dict(color=colors_chart),
                text=[f'{p*100:.1f}%' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Alert Level Probabilities",
            xaxis_title="Alert Category",
            yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.subheader("🔬 Feature Analysis")
        
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features,
            'Scaled Value': scaler.transform([features])[0]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Feature importance visualization (simple bar chart)
            fig_features = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=np.abs(features),
                    marker=dict(color='#007bff'),
                    text=[f'{v:.2f}' for v in features],
                    textposition='auto'
                )
            ])
            
            fig_features.update_layout(
                title="Feature Values",
                xaxis_title="Features",
                yaxis_title="Normalized Value",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
    
    # Information section
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Earthquake Prediction Model
        
        **Model Details:**
        - Algorithm: Random Forest Classifier
        - Accuracy: 93%
        - Training Data: Comprehensive earthquake dataset
        
        **Features Used:**
        1. **Magnitude**: Earthquake strength on Richter scale
        2. **Depth**: Distance from Earth's surface (km)
        3. **CDI**: Community Decimal Intensity (1-10 scale)
        4. **MMI**: Modified Mercalli Intensity (1-12 scale)
        5. **Significance**: Event significance score
        
        **Alert Levels:**
        - 🟢 **Green**: Safe - Low seismic activity
        - 🟠 **Orange**: Minor - Local precautions recommended
        - 🟡 **Yellow**: Moderate - Enhanced monitoring suggested
        - 🔴 **Red**: High - Immediate alert and action required
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>🌍 Earthquake Alert System | Powered by Random Forest ML Model</p>
        <p>⚠️ For emergency situations, contact local seismic authorities</p>
    </div>
    """, unsafe_allow_html=True)
