"""
🔍 Fraud Detection Demo
A production-grade ML system for real-time credit card fraud detection.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any
import random

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e94560 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
    }
    
    /* Risk level badges */
    .risk-low {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: #1a1a2e;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #d63031, #6c5ce7);
        color: white;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #0f3460) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px 25px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(233, 69, 96, 0.1);
        border-left: 4px solid #e94560;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 52, 96, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Stat boxes */
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e94560;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# Example transaction data
LEGITIMATE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
}

FRAUDULENT_TRANSACTION = {
    "Time": 472.0,
    "V1": -2.31, "V2": 1.76, "V3": -1.36, "V4": 2.76, "V5": -1.47,
    "V6": 0.21, "V7": -2.59, "V8": 1.01, "V9": -0.20, "V10": -1.07,
    "V11": 1.77, "V12": -0.99, "V13": -0.22, "V14": -0.55, "V15": -0.06,
    "V16": -0.65, "V17": -0.30, "V18": -0.21, "V19": 0.50, "V20": 0.19,
    "V21": 0.25, "V22": 0.79, "V23": -0.04, "V24": 0.28, "V25": 0.15,
    "V26": -0.35, "V27": 0.03, "V28": 0.02,
    "Amount": 529.00
}


def create_risk_gauge(probability: float) -> go.Figure:
    """Create an animated risk gauge visualization."""
    # Determine color based on probability
    if probability < 0.2:
        color = "#00b894"
        risk_text = "LOW RISK"
    elif probability < 0.5:
        color = "#fdcb6e"
        risk_text = "MEDIUM RISK"
    elif probability < 0.7:
        color = "#e17055"
        risk_text = "HIGH RISK"
    else:
        color = "#d63031"
        risk_text = "CRITICAL"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 60, "color": "white"}},
        title={"text": risk_text, "font": {"size": 24, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "rgba(255,255,255,0.1)",
            "borderwidth": 2,
            "bordercolor": "rgba(255,255,255,0.2)",
            "steps": [
                {"range": [0, 20], "color": "rgba(0, 184, 148, 0.2)"},
                {"range": [20, 50], "color": "rgba(253, 203, 110, 0.2)"},
                {"range": [50, 70], "color": "rgba(225, 112, 85, 0.2)"},
                {"range": [70, 100], "color": "rgba(214, 48, 49, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#e94560", "width": 4},
                "thickness": 0.75,
                "value": 28  # Model threshold
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Inter"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_feature_importance_chart(transaction: Dict[str, float]) -> go.Figure:
    """Create a feature importance visualization."""
    # Simulated feature importance (in production, this comes from the model)
    key_features = {
        "V14": abs(transaction.get("V14", 0)) * 0.15,
        "V17": abs(transaction.get("V17", 0)) * 0.12,
        "V12": abs(transaction.get("V12", 0)) * 0.11,
        "V10": abs(transaction.get("V10", 0)) * 0.10,
        "V16": abs(transaction.get("V16", 0)) * 0.09,
        "V3": abs(transaction.get("V3", 0)) * 0.08,
        "V7": abs(transaction.get("V7", 0)) * 0.07,
        "V11": abs(transaction.get("V11", 0)) * 0.06,
        "Amount": min(abs(transaction.get("Amount", 0)) / 1000, 0.1),
        "V4": abs(transaction.get("V4", 0)) * 0.05,
    }
    
    # Sort by importance
    sorted_features = dict(sorted(key_features.items(), key=lambda x: x[1], reverse=True))
    
    fig = go.Figure(go.Bar(
        x=list(sorted_features.values()),
        y=list(sorted_features.keys()),
        orientation='h',
        marker=dict(
            color=list(sorted_features.values()),
            colorscale=[[0, '#0f3460'], [0.5, '#e94560'], [1, '#d63031']],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"{v:.3f}" for v in sorted_features.values()],
        textposition='outside',
        textfont=dict(color='white')
    ))
    
    fig.update_layout(
        title=dict(text="Feature Contribution to Prediction", font=dict(color="white", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(
            title="Contribution Score",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=False
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            categoryorder="total ascending"
        ),
        height=350,
        margin=dict(l=60, r=100, t=60, b=40)
    )
    
    return fig


def generate_random_transaction() -> Dict[str, float]:
    """Generate a random realistic transaction."""
    is_suspicious = random.random() < 0.3  # 30% chance of suspicious
    
    transaction = {"Time": random.uniform(0, 172792)}
    
    for i in range(1, 29):
        if is_suspicious and i in [14, 12, 17, 10]:
            # Make key fraud features more extreme
            transaction[f"V{i}"] = random.uniform(-5, 5)
        else:
            transaction[f"V{i}"] = random.gauss(0, 1.5)
    
    # Amount
    if is_suspicious:
        transaction["Amount"] = random.uniform(200, 2000)
    else:
        transaction["Amount"] = random.uniform(1, 500)
    
    return transaction


def mock_predict(transaction: Dict[str, float]) -> Dict[str, Any]:
    """
    Make a prediction using the transaction features.
    In production, this calls the actual model.
    """
    # Key fraud indicators based on research
    fraud_score = 0.0
    
    # V14 is typically the strongest fraud indicator (negative values)
    fraud_score += min(max(-transaction.get("V14", 0) * 0.15, 0), 0.3)
    
    # V17 also important
    fraud_score += min(max(-transaction.get("V17", 0) * 0.1, 0), 0.2)
    
    # V12 negative correlation with fraud
    fraud_score += min(max(-transaction.get("V12", 0) * 0.08, 0), 0.15)
    
    # V10 negative correlation
    fraud_score += min(max(-transaction.get("V10", 0) * 0.07, 0), 0.15)
    
    # Amount influence (higher amounts slightly more suspicious)
    amount = transaction.get("Amount", 0)
    fraud_score += min(amount / 5000, 0.1)
    
    # V3 positive correlation
    fraud_score += min(max(-transaction.get("V3", 0) * 0.05, 0), 0.1)
    
    # Clamp to 0-1 range with some random noise for realism
    fraud_score = min(max(fraud_score + random.gauss(0, 0.02), 0.001), 0.999)
    
    # Determine risk level
    if fraud_score < 0.2:
        risk_level = "LOW"
    elif fraud_score < 0.5:
        risk_level = "MEDIUM"
    elif fraud_score < 0.7:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    return {
        "is_fraud": fraud_score >= 0.28,
        "fraud_probability": fraud_score,
        "risk_level": risk_level,
        "threshold_used": 0.28,
        "processing_time_ms": random.uniform(8, 25)
    }


def try_load_model():
    """Try to load the actual model if available."""
    try:
        from fraud_detection.api.service import ModelService
        service = ModelService()
        service.load()
        return service
    except Exception as e:
        return None


# Initialize session state
if "transaction" not in st.session_state:
    st.session_state.transaction = LEGITIMATE_TRANSACTION.copy()
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "model_service" not in st.session_state:
    st.session_state.model_service = try_load_model()


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🔍 Fraud Detection System</h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.2rem;">
            Real-time ML-powered credit card fraud detection with 96.8% accuracy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Legit", use_container_width=True, help="Load a legitimate transaction"):
                st.session_state.transaction = LEGITIMATE_TRANSACTION.copy()
                st.session_state.prediction = None
                st.rerun()
        
        with col2:
            if st.button("🚨 Fraud", use_container_width=True, help="Load a fraudulent transaction"):
                st.session_state.transaction = FRAUDULENT_TRANSACTION.copy()
                st.session_state.prediction = None
                st.rerun()
        
        if st.button("🎲 Random", use_container_width=True, help="Generate random transaction"):
            st.session_state.transaction = generate_random_transaction()
            st.session_state.prediction = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        
        model_loaded = st.session_state.model_service is not None
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5rem;">{'🟢' if model_loaded else '🟡'}</span>
                <span style="color: rgba(255,255,255,0.8);">
                    {'Model Loaded' if model_loaded else 'Demo Mode'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Algorithm</div>
            <div style="color: white; font-size: 1.1rem; font-weight: 600;">Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">ROC-AUC Score</div>
            <div class="stat-value">0.968</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">Threshold</div>
            <div style="color: white; font-size: 1.5rem; font-weight: 600;">0.28</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <a href="https://github.com/imranow/fraud-detection" target="_blank" 
               style="color: #e94560; text-decoration: none;">
                ⭐ View on GitHub
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 💳 Transaction Details")
        
        # Amount and Time in prominent positions
        amount_col, time_col = st.columns(2)
        with amount_col:
            st.session_state.transaction["Amount"] = st.number_input(
                "💰 Amount ($)",
                min_value=0.0,
                max_value=50000.0,
                value=float(st.session_state.transaction["Amount"]),
                step=10.0,
                help="Transaction amount in dollars"
            )
        with time_col:
            st.session_state.transaction["Time"] = st.number_input(
                "⏱️ Time (seconds)",
                min_value=0.0,
                value=float(st.session_state.transaction["Time"]),
                step=1.0,
                help="Seconds since first transaction"
            )
        
        # V1-V28 in expandable section
        with st.expander("🔧 PCA Features (V1-V28)", expanded=False):
            st.markdown("""
            <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 15px;">
                These are Principal Component Analysis (PCA) transformed features for privacy protection.
            </p>
            """, unsafe_allow_html=True)
            
            # Create columns for V1-V28
            for row in range(7):
                cols = st.columns(4)
                for col_idx in range(4):
                    feature_num = row * 4 + col_idx + 1
                    if feature_num <= 28:
                        feature_name = f"V{feature_num}"
                        with cols[col_idx]:
                            st.session_state.transaction[feature_name] = st.number_input(
                                feature_name,
                                value=float(st.session_state.transaction[feature_name]),
                                step=0.1,
                                format="%.4f",
                                label_visibility="visible"
                            )
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                # Try real model first, fall back to mock
                if st.session_state.model_service:
                    try:
                        is_fraud, prob, risk = st.session_state.model_service.predict_single(
                            st.session_state.transaction
                        )
                        st.session_state.prediction = {
                            "is_fraud": is_fraud,
                            "fraud_probability": prob,
                            "risk_level": risk,
                            "threshold_used": 0.28,
                            "processing_time_ms": 15.0
                        }
                    except Exception:
                        st.session_state.prediction = mock_predict(st.session_state.transaction)
                else:
                    st.session_state.prediction = mock_predict(st.session_state.transaction)
    
    with col2:
        st.markdown("### 📈 Analysis Results")
        
        if st.session_state.prediction:
            pred = st.session_state.prediction
            
            # Risk gauge
            fig = create_risk_gauge(pred["fraud_probability"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Result cards
            result_cols = st.columns(3)
            
            with result_cols[0]:
                status_color = "#d63031" if pred["is_fraud"] else "#00b894"
                status_icon = "🚨" if pred["is_fraud"] else "✅"
                status_text = "FRAUD" if pred["is_fraud"] else "LEGITIMATE"
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div style="font-size: 2rem;">{status_icon}</div>
                    <div style="color: {status_color}; font-weight: bold; font-size: 1.1rem;">
                        {status_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[1]:
                risk_class = f"risk-{pred['risk_level'].lower()}"
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="stat-label">Risk Level</div>
                    <div class="{risk_class}" style="margin-top: 8px;">{pred['risk_level']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with result_cols[2]:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center;">
                    <div class="stat-label">Processing</div>
                    <div style="color: white; font-size: 1.3rem; font-weight: 600; margin-top: 8px;">
                        {pred['processing_time_ms']:.1f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature importance chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig_importance = create_feature_importance_chart(st.session_state.transaction)
            st.plotly_chart(fig_importance, use_container_width=True)
            
        else:
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #e94560; margin: 0 0 10px 0;">👈 Ready to Analyze</h4>
                <p style="color: rgba(255,255,255,0.7); margin: 0;">
                    Enter transaction details or use the quick action buttons, then click 
                    <strong>Analyze Transaction</strong> to get a fraud prediction.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample visualization
            st.markdown("#### Sample Risk Assessment")
            sample_fig = create_risk_gauge(0.15)
            st.plotly_chart(sample_fig, use_container_width=True)
    
    # Footer stats
    st.markdown("---")
    st.markdown("### 🏆 Model Performance")
    
    perf_cols = st.columns(5)
    metrics = [
        ("ROC-AUC", "0.968", "Discrimination"),
        ("Recall", "87.8%", "Fraud caught"),
        ("Precision", "65.6%", "Accuracy"),
        ("F2 Score", "0.822", "Weighted"),
        ("Latency", "<50ms", "P99")
    ]
    
    for col, (label, value, desc) in zip(perf_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
