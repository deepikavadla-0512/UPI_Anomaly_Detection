# 💳 UPI Transaction Anomaly Detection

[💳 Live Demo](https://upi-anomaly-detection.onrender.com/)

**AI-powered system to detect suspicious UPI transactions using Autoencoders.**  
This project simulates UPI transaction data and detects anomalies (potential fraud) using a trained autoencoder model. It includes interactive dashboards, live transaction simulation, and downloadable reports.

---

## 🌟 Features

- **Anomaly Detection:** Detect suspicious transactions based on learned normal behavior using Autoencoders.
- **Interactive Dashboard:** View reconstruction errors, anomaly scores, and transaction trends.
- **High-Risk Alerts:** Automatically flags high-risk transactions.
- **Live Transaction Simulation:** Test new transactions instantly to see if they are anomalous.
- **Download Reports:** Export detected anomalies as CSV for reporting or analysis.
- **Professional Metrics:** Confusion matrix, classification report, and ROC-AUC for evaluation.

## 🖥️ Live Demo

Experience the app live:

[💳 Open Live App](https://your-render-link.onrender.com)

---

## 📊 Screenshots

![Dashboard Screenshot](<img width="1920" height="1080" alt="Screenshot" src="https://github.com/user-attachments/assets/ee2817b2-e6e6-48b0-899f-565db962d4cc" />
<img width="1920" height="1080" alt="Screenshot" src="https://github.com/user-attachments/assets/ee2817b2-e6e6-48b0-899f-565db962d4cc" />

)
*Before uploading of Dataset.*

![Dashboard Screenshot](<img width="1920" height="1080" alt="Screenshot1" src="https://github.com/user-attachments/assets/c1dcf5ed-2c51-4022-bea8-4ce4621c9cba" />
<img width="1920" height="1080" alt="Screenshot1" src="https://github.com/user-attachments/assets/c1dcf5ed-2c51-4022-bea8-4ce4621c9cba" />
)  
*Reconstruction error distribution.*

![Anomalies Table](<img width="1920" height="1080" alt="Screenshot5" src="https://github.com/user-attachments/assets/43bb2678-22b8-4bb0-9f3c-ea0b46080707" />
<img width="1920" height="1080" alt="Screenshot5" src="https://github.com/user-attachments/assets/43bb2678-22b8-4bb0-9f3c-ea0b46080707" />
)  
*Confusion matrix.*

![Alerts](<img width="1920" height="1080" alt="Screenshot4" src="https://github.com/user-attachments/assets/ef366b48-792a-4f32-9c15-6cad5ebde612" />
<img width="1920" height="1080" alt="Screenshot4" src="https://github.com/user-attachments/assets/ef366b48-792a-4f32-9c15-6cad5ebde612" />
)  
*High-risk transaction alerts.*

> Replace screenshots with your own from the deployed app for best presentation.

---

## 🛠️ Technologies Used

- **Frontend & Deployment:** Streamlit, Render
- **Backend & ML:** Python, TensorFlow, Keras, scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** Plotly, Plotly Figure Factory
- **Model Persistence:** joblib

---

## ⚙️ Setup Instructions

Follow these steps to run the app locally:

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/UPI_Anomaly_Detection.git
cd UPI_Anomaly_Detection
Create a virtual environment

python -m venv venv
Activate the environment

Windows

venv\Scripts\activate
Mac/Linux

source venv/bin/activate
Install dependencies

pip install -r requirements.txt
Run the app

streamlit run app.py
Open in browser
The app will open automatically, or visit http://localhost:8501.

📁 Project Structure
UPI_Anomaly_Detection/
│
├── app.py                  # Streamlit application
├── upi_autoencoder_model.keras  # Trained Autoencoder model
├── ct.pkl                  # ColumnTransformer for preprocessing
├── scaler.pkl              # MinMaxScaler object
├── threshold.pkl           # Reconstruction error threshold
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── screenshots/            # Dashboard screenshots
🚀 How It Works
Upload CSV: Upload a dataset of UPI transactions.

Preprocessing: Features are engineered (hour_of_day, log_amount) and encoded.

Anomaly Detection: Autoencoder predicts reconstruction errors; transactions above threshold are anomalies.

Dashboard & Alerts: Interactive charts, anomalies table, and high-risk alert flags.

Simulation: Test new transactions in real-time to check risk.

📈 Evaluation Metrics
If ground truth labels are available, the app computes:

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

ROC-AUC Score

💡 Notes
Use CSV files with the same columns as in training (transaction_id, sender_upi, receiver_upi, amount, transaction_type, location, device_type, transaction_status, timestamp, label (optional)).

Live simulation uses the same preprocessing and model for instant anomaly detection.

Dashboard and visualizations are fully interactive using Plotly.





