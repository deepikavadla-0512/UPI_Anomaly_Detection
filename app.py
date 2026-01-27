import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="UPI Anomaly Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 UPI Transaction Anomaly Detection Dashboard")
st.markdown("**AI-powered system to detect suspicious UPI transactions using Autoencoders**")

# =============================
# Column Definitions
# =============================
categorical_cols = ["transaction_type", "location", "device_type", "transaction_status"]
numerical_cols = ["log_amount", "hour_of_day"]

# =============================
# Load Model & Preprocessing Objects
# =============================
@st.cache_resource
def load_objects():
    model = load_model("upi_autoencoder_model.keras", compile=False)
    ct = joblib.load("ct.pkl")
    scaler = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, ct, scaler, threshold

model, ct, scaler, threshold = load_objects()
st.success("✅ Model & preprocessing files loaded successfully")

# =============================
# File Upload
# =============================
uploaded_file = st.file_uploader("📂 Upload UPI Transactions CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Feature Engineering
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log1p(df["amount"])

    # Transform & Predict
    X = ct.transform(df[categorical_cols + numerical_cols])
    X_scaled = scaler.transform(X)
    with st.spinner("🔍 Detecting anomalies..."):
        preds = model.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - preds), axis=1)

    df["reconstruction_error"] = mse

    # =============================
    # Threshold Slider
    # =============================
    threshold_slider = st.sidebar.slider(
        "Adjust Anomaly Threshold",
        float(df["reconstruction_error"].min()),
        float(df["reconstruction_error"].max()),
        float(threshold)
    )

    df["predicted_label"] = (df["reconstruction_error"] > threshold_slider).astype(int)
    df["alert_flag"] = df["reconstruction_error"] > threshold_slider * 1.5

    # =============================
    # Sidebar Filters
    # =============================
    st.sidebar.header("🔎 Filters")
    tx_type = st.sidebar.multiselect("Transaction Type", df["transaction_type"].unique(), default=df["transaction_type"].unique())
    device = st.sidebar.multiselect("Device Type", df["device_type"].unique(), default=df["device_type"].unique())
    status = st.sidebar.multiselect("Transaction Status", df["transaction_status"].unique(), default=df["transaction_status"].unique())
    anomaly_filter = st.sidebar.radio("View", ["All Transactions", "Only Anomalies"])

    filtered_df = df[(df["transaction_type"].isin(tx_type)) &
                     (df["device_type"].isin(device)) &
                     (df["transaction_status"].isin(status))]
    if anomaly_filter == "Only Anomalies":
        filtered_df = filtered_df[filtered_df["predicted_label"] == 1]

    # =============================
    # KPI Metrics
    # =============================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(filtered_df))
    col2.metric("Anomalies Detected", int(filtered_df["predicted_label"].sum()))
    col3.metric("High-Risk Alerts", int(filtered_df["alert_flag"].sum()))

    # =============================
    # Tabs
    # =============================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "⚠️ Anomalies", "🚨 Alerts", "📈 Model Evaluation"])

    # -----------------------------
    # Dashboard Tab
    # -----------------------------
    with tab1:
        st.subheader("Reconstruction Error Distribution")
        fig1 = px.histogram(
            filtered_df,
            x="reconstruction_error",
            color="predicted_label",
            nbins=40,
            color_discrete_map={0: "green", 1: "red"},
            labels={"predicted_label": "Anomaly"}
        )
        fig1.add_vline(x=threshold_slider, line_dash="dash", line_color="black")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Amount vs Anomaly Score")
        fig2 = px.scatter(
            filtered_df,
            x="amount",
            y="reconstruction_error",
            color="predicted_label",
            hover_data=["transaction_id"],
            color_discrete_map={0: "blue", 1: "red"}
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Risk score and timeline
        df["risk_score"] = (df["reconstruction_error"] / df["reconstruction_error"].max()) * 100
        fig_time = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="reconstruction_error",
            title="Reconstruction Error Over Time"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # -----------------------------
    # Anomalies Table Tab
    # -----------------------------
    with tab2:
        anomalies = filtered_df[filtered_df["predicted_label"] == 1]
        with st.expander("Show Detected Anomalies Table"):
            st.dataframe(anomalies.sort_values("reconstruction_error", ascending=False), use_container_width=True)

    # -----------------------------
    # Alerts Tab
    # -----------------------------
    with tab3:
        alerts = filtered_df[filtered_df["alert_flag"] == 1]
        with st.expander("Show High-Risk Alerts"):
            if not alerts.empty:
                st.error(f"🚨 {len(alerts)} HIGH-RISK TRANSACTIONS DETECTED")
                st.dataframe(alerts, use_container_width=True)
            else:
                st.success("✅ No critical alerts found")

    # -----------------------------
    # Model Evaluation Tab
    # -----------------------------
    with tab4:
        st.subheader("📈 Model Performance")
        if "label" in df.columns:
            cm = confusion_matrix(df["label"], df["predicted_label"])
            fig_cm = ff.create_annotated_heatmap(
                cm.tolist(),
                x=["Predicted Normal", "Predicted Anomaly"],
                y=["Actual Normal", "Actual Anomaly"],
                colorscale="Blues"
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

            report = classification_report(df["label"], df["predicted_label"], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            roc = roc_auc_score(df["label"], df["predicted_label"])
            st.metric("ROC-AUC Score", round(roc, 4))
        else:
            st.warning("⚠️ Ground truth labels not available")

    # -----------------------------
    # Download Anomaly Report
    # -----------------------------
    st.subheader("📥 Download Anomaly Report")
    report_df = filtered_df[filtered_df["predicted_label"] == 1]
    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "UPI_Anomaly_Report.csv", "text/csv")

    # -----------------------------
    # Live Transaction Simulation
    # -----------------------------
    st.write("### 🖊️ Simulate a New Transaction")
    with st.form("txn_form"):
        amount = st.number_input("Amount (₹)", min_value=0.0, value=5000.0)
        transaction_type = st.selectbox("Transaction Type", df["transaction_type"].unique())
        location = st.selectbox("Location", df["location"].unique())
        device_type = st.selectbox("Device Type", df["device_type"].unique())
        transaction_status = st.selectbox("Transaction Status", df["transaction_status"].unique())
        submit = st.form_submit_button("Check Transaction")

    if submit:
        new_txn = pd.DataFrame({
            "transaction_type": [transaction_type],
            "location": [location],
            "device_type": [device_type],
            "transaction_status": [transaction_status],
            "amount": [amount],
            "timestamp": [pd.Timestamp.now()]
        })
        new_txn["hour_of_day"] = new_txn["timestamp"].dt.hour
        new_txn["log_amount"] = np.log1p(new_txn["amount"])

        X_new = ct.transform(new_txn[categorical_cols + numerical_cols])
        X_new = scaler.transform(X_new)
        pred_new = model.predict(X_new)
        mse_new = np.mean(np.square(X_new - pred_new), axis=1)[0]
        label = "⚠️ Anomalous Transaction" if mse_new > threshold_slider else "✅ Normal Transaction"

        st.write(f"**Result:** {label}")
        st.write(f"**Reconstruction Error:** {mse_new:.6f}")

else:
    st.info("👆 Please upload a CSV file to start anomaly detection")
