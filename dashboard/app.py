# dashboard/app.py
# Streamlit dashboard for Sepsis Early Warning System

import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import time

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title = "Sepsis Early Warning System",
    page_icon  = "🏥",
    layout     = "wide",
)

# ── API config ─────────────────────────────────────────
API_URL = "https://sepsis-early-warning.onrender.com"

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .risk-red    { background:#FFEBEE; border-left:6px solid #F44336;
                   padding:16px; border-radius:4px; }
    .risk-amber  { background:#FFF3E0; border-left:6px solid #FF9800;
                   padding:16px; border-radius:4px; }
    .risk-yellow { background:#FFFDE7; border-left:6px solid #FFC107;
                   padding:16px; border-radius:4px; }
    .risk-green  { background:#E8F5E9; border-left:6px solid #4CAF50;
                   padding:16px; border-radius:4px; }
    .metric-box  { background:#F5F5F5; padding:12px;
                   border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────
st.title("🏥 Sepsis Early Warning System")
st.markdown("*6-hour early prediction using LSTM neural network*")
st.divider()

# ── Sidebar - Patient Input ────────────────────────────
st.sidebar.header("👤 Patient Information")

patient_id = st.sidebar.text_input(
    "Patient ID", value="P001"
)

st.sidebar.header("📊 Enter Last 6 Hours of Vitals")
st.sidebar.markdown("*Fill in available readings. "
                    "Missing values will be imputed.*")

# Create input for 6 hours
hours_data = []
for hour in range(1, 7):
    with st.sidebar.expander(
        f"Hour {hour}", expanded=(hour == 6)
    ):
        col1, col2 = st.columns(2)
        with col1:
            hr   = st.number_input(
                f"HR (bpm)", 
                min_value=0.0, max_value=300.0,
                value=85.0, key=f"hr_{hour}"
            )
            sbp  = st.number_input(
                f"SBP (mmHg)",
                min_value=0.0, max_value=300.0,
                value=120.0, key=f"sbp_{hour}"
            )
            resp = st.number_input(
                f"Resp (br/min)",
                min_value=0.0, max_value=100.0,
                value=18.0, key=f"resp_{hour}"
            )
            temp = st.number_input(
                f"Temp (°C)",
                min_value=0.0, max_value=45.0,
                value=37.0, key=f"temp_{hour}"
            )
        with col2:
            o2sat = st.number_input(
                f"O2Sat (%)",
                min_value=0.0, max_value=100.0,
                value=98.0, key=f"o2sat_{hour}"
            )
            map_  = st.number_input(
                f"MAP (mmHg)",
                min_value=0.0, max_value=300.0,
                value=80.0, key=f"map_{hour}"
            )
            dbp   = st.number_input(
                f"DBP (mmHg)",
                min_value=0.0, max_value=300.0,
                value=60.0, key=f"dbp_{hour}"
            )

        st.markdown("**Lab Values (if available):**")
        col3, col4 = st.columns(2)
        with col3:
            lactate = st.number_input(
                "Lactate", min_value=0.0,
                max_value=50.0, value=0.0,
                key=f"lac_{hour}"
            )
            wbc = st.number_input(
                "WBC", min_value=0.0,
                max_value=500.0, value=0.0,
                key=f"wbc_{hour}"
            )
            creat = st.number_input(
                "Creatinine", min_value=0.0,
                max_value=50.0, value=0.0,
                key=f"creat_{hour}"
            )
        with col4:
            glucose = st.number_input(
                "Glucose", min_value=0.0,
                max_value=1000.0, value=0.0,
                key=f"gluc_{hour}"
            )
            ph = st.number_input(
                "pH", min_value=0.0,
                max_value=10.0, value=0.0,
                key=f"ph_{hour}"
            )
            hgb = st.number_input(
                "Hgb", min_value=0.0,
                max_value=50.0, value=0.0,
                key=f"hgb_{hour}"
            )

        hours_data.append({
            "HR": hr, "O2Sat": o2sat,
            "SBP": sbp, "MAP": map_,
            "DBP": dbp, "Resp": resp,
            "Temp": temp,
            "Lactate":    lactate if lactate > 0 else None,
            "WBC":        wbc     if wbc     > 0 else None,
            "Creatinine": creat   if creat   > 0 else None,
            "Glucose":    glucose if glucose > 0 else None,
            "pH":         ph      if ph      > 0 else None,
            "Hgb":        hgb     if hgb     > 0 else None,
            "Age":        65.0,
            "Gender":     1.0,
            "HospAdmTime":-2.0,
            "ICULOS":     float(hour),
        })

# ── Predict button ─────────────────────────────────────
predict_btn = st.sidebar.button(
    "🔍 Assess Sepsis Risk",
    type="primary",
    use_container_width=True
)

# ── Main dashboard ─────────────────────────────────────
if predict_btn:
    with st.spinner("Analysing patient data..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "patient_id": patient_id,
                    "readings":   hours_data
                },
                timeout=10
            )
            result = response.json()

            # ── Alert Banner ───────────────────────────
            level = result['alert_level']
            score = result['risk_score']
            msg   = result['alert_message']

            css_class = {
                "RED":    "risk-red",
                "AMBER":  "risk-amber",
                "YELLOW": "risk-yellow",
                "GREEN":  "risk-green"
            }.get(level, "risk-green")

            emoji = {
                "RED":    "🔴",
                "AMBER":  "🟠",
                "YELLOW": "🟡",
                "GREEN":  "🟢"
            }.get(level, "🟢")

            st.markdown(
                f'<div class="{css_class}">'
                f'<h2>{emoji} {level} ALERT</h2>'
                f'<p><strong>{msg}</strong></p>'
                f'<p>Patient: {patient_id} | '
                f'Risk Score: {score:.1%} | '
                f'Time: {datetime.now().strftime("%H:%M:%S")}'
                f'</p></div>',
                unsafe_allow_html=True
            )

            st.divider()

            # ── Metrics row ────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Risk Score",
                    f"{score:.1%}",
                    delta=None
                )
            with col2:
                st.metric(
                    "Alert Level", level
                )
            with col3:
                st.metric(
                    "Sepsis Alert",
                    "YES" if result['sepsis_in_6h'] 
                    else "NO"
                )
            with col4:
                st.metric(
                    "Hours Analysed",
                    result['hours_of_data']
                )

            st.divider()

            # ── Vitals trend charts ────────────────────
            st.subheader("📈 Vital Signs Trend (Last 6 Hours)")

            vitals_df = pd.DataFrame([{
                'Hour':  i + 1,
                'HR':    h['HR'],
                'O2Sat': h['O2Sat'],
                'SBP':   h['SBP'],
                'Resp':  h['Resp'],
                'Temp':  h['Temp'],
            } for i, h in enumerate(hours_data)])

            fig, axes = plt.subplots(
                1, 5, figsize=(18, 3)
            )

            configs = [
                ('HR',    'Heart Rate',    'bpm',
                 '#E53935', (60, 100)),
                ('O2Sat', 'O2 Saturation', '%',
                 '#1E88E5', (95, 100)),
                ('SBP',   'Systolic BP',   'mmHg',
                 '#43A047', (90, 140)),
                ('Resp',  'Resp Rate',     'br/min',
                 '#FB8C00', (12, 20)),
                ('Temp',  'Temperature',   '°C',
                 '#8E24AA', (36.1, 37.2)),
            ]

            for ax, (col, label, unit, 
                     color, normal) in zip(axes, configs):
                ax.plot(
                    vitals_df['Hour'],
                    vitals_df[col],
                    color=color, linewidth=2.5,
                    marker='o', markersize=6
                )
                ax.axhspan(
                    normal[0], normal[1],
                    alpha=0.15, color='green',
                    label='Normal range'
                )
                ax.set_title(
                    f'{label}\n({unit})',
                    fontsize=10, fontweight='bold'
                )
                ax.set_xlabel('Hour')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(range(1, 7))

            plt.tight_layout()
            st.pyplot(fig)

            # ── SHAP Explanation ───────────────────────
            st.divider()
            st.subheader("🔍 What Drove This Prediction?")
            st.markdown(
                "*Based on SHAP analysis — "
                "which features most influenced "
                "the risk score*"
            )

            shap_col1, shap_col2 = st.columns(2)

            with shap_col1:
                st.markdown("**🔴 Risk-Increasing Factors**")
                risk_factors = result.get(
                    'top_risk_factors', []
                )
                if risk_factors:
                    for factor in risk_factors:
                        contribution = factor['contribution']
                        display      = factor['display_name']
                        shap_val     = factor['shap_value']

                        # Colour bar based on contribution
                        bar_width = min(
                            abs(shap_val) * 1000, 100
                        )
                        st.markdown(
                            f"""
                            <div style="margin-bottom:8px">
                                <div style="display:flex;
                                    justify-content:space-between">
                                    <span>{display}</span>
                                    <strong style="color:#F44336">
                                        {contribution}
                                    </strong>
                                </div>
                                <div style="background:#eee;
                                    border-radius:4px;height:8px">
                                    <div style="background:#F44336;
                                        width:{bar_width}%;
                                        height:8px;
                                        border-radius:4px">
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No risk-increasing factors detected")

            with shap_col2:
                st.markdown("**🟢 Protective Factors**")
                protective = result.get(
                    'protective_factors', []
                )
                if protective:
                    for factor in protective:
                        contribution = factor['contribution']
                        display      = factor['display_name']
                        shap_val     = factor['shap_value']

                        bar_width = min(
                            abs(shap_val) * 1000, 100
                        )
                        st.markdown(
                            f"""
                            <div style="margin-bottom:8px">
                                <div style="display:flex;
                                    justify-content:space-between">
                                    <span>{display}</span>
                                    <strong style="color:#4CAF50">
                                        {contribution}
                                    </strong>
                                </div>
                                <div style="background:#eee;
                                    border-radius:4px;height:8px">
                                    <div style="background:#4CAF50;
                                        width:{bar_width}%;
                                        height:8px;
                                        border-radius:4px">
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No protective factors detected")


            # ── Clinical notes ─────────────────────────
            st.divider()
            st.subheader("📋 Clinical Notes")

            # Identify abnormal vitals
            last = hours_data[-1]
            warnings = []

            if last['HR'] and last['HR'] > 100:
                warnings.append(
                    f"⚠️ Tachycardia: HR = {last['HR']} bpm"
                )
            if last['O2Sat'] and last['O2Sat'] < 95:
                warnings.append(
                    f"⚠️ Low O2Sat: {last['O2Sat']}%"
                )
            if last['SBP'] and last['SBP'] < 90:
                warnings.append(
                    f"⚠️ Hypotension: SBP = {last['SBP']} mmHg"
                )
            if last['Resp'] and last['Resp'] > 20:
                warnings.append(
                    f"⚠️ Tachypnoea: Resp = {last['Resp']} br/min"
                )
            if last['Temp'] and last['Temp'] > 38.3:
                warnings.append(
                    f"⚠️ Fever: Temp = {last['Temp']}°C"
                )
            if last['Lactate'] and last['Lactate'] > 2.0:
                warnings.append(
                    f"⚠️ Elevated Lactate: {last['Lactate']} mmol/L"
                )

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success(
                    "✅ All vitals within normal ranges"
                )

            # ── Model info ─────────────────────────────
            with st.expander("ℹ️ Model Information"):
                st.markdown(f"""
                **Model:** LSTM Neural Network  
                **AUROC:** 0.7796  
                **Threshold:** {result['threshold_used']}  
                **Prediction window:** 6 hours  
                **Training data:** 40,336 ICU patients  
                **Dataset:** PhysioNet Sepsis Challenge 2019  
                """)

        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to API. "
                "Make sure the API is running on port 8000."
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:
    # ── Default state ──────────────────────────────────
    st.info(
        "👈 Enter patient vitals in the sidebar "
        "and click **Assess Sepsis Risk** to get a prediction."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧠 How it works
        Enter 6 consecutive hours of patient vitals.
        The LSTM analyses trends across time to predict
        sepsis risk 6 hours before clinical onset.
        """)
    with col2:
        st.markdown("""
        ### 🎯 Alert Levels
        - 🔴 **RED** — Immediate review required
        - 🟠 **AMBER** — Increased monitoring
        - 🟡 **YELLOW** — Continue monitoring
        - 🟢 **GREEN** — No immediate action
        """)
    with col3:
        st.markdown("""
        ### 📊 Model Performance
        - **AUROC:** 0.7796
        - **Sensitivity:** 80.9%
        - **Dataset:** 40,336 patients
        - **Prediction:** 6h early warning
        """)