import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 🌈 Global Background Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom, #f4f9ff, #ffffff);
        }
        .main {
            background: linear-gradient(to bottom, #f4f9ff, #ffffff);
        }
    </style>
""", unsafe_allow_html=True)

# --------------------- Preprocessing Function ---------------------
def preprocess_input(df):
    df = df.copy()

    mapping_substance = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
    mapping_unit = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
    mapping_source = {'commodity': 0, 'industry': 1}

    df['Substance'] = df['Substance'].map(mapping_substance)
    df['Unit'] = df['Unit'].map(mapping_unit)
    df['Source'] = df['Source'].map(mapping_source)

    df['Year'] = 2018  # Default year
    return df

# --------------------- Load Model & Scaler ---------------------
model = joblib.load(r'C:\Users\T M\Desktop\LR_model.pkl')
scaler = joblib.load(r'C:\Users\T M\Desktop\scaler.pkl')

# --------------------- App Title & Description ---------------------
st.markdown("<h2 style='color:#0066cc;'>🌍 Supply Chain Emissions Prediction</h2>", unsafe_allow_html=True)
st.markdown("""
<div style='color:#333333; font-size:16px;'>This app predicts <b>Supply Chain Emission Factors with Margins</b> based on DQ metrics and other inputs.</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --------------------- Form Inputs ---------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        substance = st.selectbox("🧪 Substance", ['carbon dioxide', 'methane', 'nitrous oxide', 'other GHGs'])
        unit = st.selectbox("📏 Unit", ['kg/2018 USD, purchaser price', 'kg CO2e/2018 USD, purchaser price'])
        source = st.selectbox("🏭 Source", ['commodity', 'industry'])
        supply_wo_margin = st.number_input("🔻 Supply Chain Emission Factors (No Margin)", min_value=0.0)
        margin = st.number_input("➕ Margin of Emission Factors", min_value=0.0)

    with col2:
        dq_reliability = st.slider("✔️ DQ Reliability", 0.0, 1.0, help="How reliable the data is on a scale of 0 to 1.")
        dq_temporal = st.slider("🕒 DQ Temporal Correlation", 0.0, 1.0, help="How well the data reflects the current time.")
        dq_geo = st.slider("🌎 DQ Geographical Correlation", 0.0, 1.0, help="How regionally accurate the data is.")
        dq_tech = st.slider("⚙️ DQ Technological Correlation", 0.0, 1.0, help="How well the data reflects current technology.")
        dq_data = st.slider("📊 DQ Data Collection", 0.0, 1.0, help="Quality of data collection practices.")

    submit = st.form_submit_button("🔍 Predict")

# --------------------- Prediction Logic ---------------------
    if submit:
        input_data = {
            'Substance': substance,
            'Unit': unit,
            'Source': source,
            'Supply Chain Emission Factors without Margins': supply_wo_margin,
            'Margins of Supply Chain Emission Factors': margin,
            'DQ ReliabilityScore of Factors without Margins': dq_reliability,
            'DQ TemporalCorrelation of Factors without Margins': dq_temporal,
            'DQ GeographicalCorrelation of Factors without Margins': dq_geo,
            'DQ TechnologicalCorrelation of Factors without Margins': dq_tech,
            'DQ DataCollection of Factors without Margins': dq_data
        }

        input_df = preprocess_input(pd.DataFrame([input_data]))
        expected_columns = scaler.feature_names_in_
        input_df = input_df[expected_columns]

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        # Score classifier
        def classify_score(value):
            if value >= 0.7:
                return "🟢 Good", "#29c94e"
            elif value >= 0.4:
                return "🟡 Moderate", "#e7bf45"
            else:
                return "🔴 Poor", "#dc3545"

        # Styled prediction result
        st.markdown("---")
        st.markdown(f"""
        <div style='padding: 25px; background: linear-gradient(to right, #e0f7fa, #ffffff); 
                     border-left: 8px solid darkblue; border-radius: 10px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1);'>
            <h4 style='color:darkblue;'>✅ Predicted Emission Factor with Margin:</h4>
            <h3 style='color:green;'>⚡ {prediction[0]:.4f}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Compact DQ Metric Summary badges
        st.markdown("### 🧾 DQ Metric Summary")
        dq_data_quality = {
            "DQ Reliability": dq_reliability,
            "DQ Temporal Correlation": dq_temporal,
            "DQ Geographical Correlation": dq_geo,
            "DQ Technological Correlation": dq_tech,
            "DQ Data Collection": dq_data
        }

        for label, score in dq_data_quality.items():
            status, bg_color = classify_score(score)
            st.markdown(
                f"""
                <div style='
                    padding: 6px 10px;
                    background-color: {bg_color};
                    color: white;
                    border-radius: 5px;
                    display: inline-block;
                    margin: 4px 4px 4px 0;
                    font-size: 12px;
                '>
                    {label}: <b>{status}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
