# AI-Driven Oil Spill Identification and Monitoring System

This repository hosts a production-grade, end-to-end Machine Learning web application designed to identify and analyze oil spills from satellite or aerial imagery. Using **Transfer Learning with MobileNetV2** and **Computer Vision (OpenCV)**, the system predicts the presence of oil spills, estimates spill area coverage, classifies threat severity, and leverages **Grad-CAM** gradient maps to explain AI classification decisions.

---

## Key Features

1. **Secure User Authentication**: Integrated login/registration portals with salted and hashed credentials stored in an SQLite backend.
2. **Interactive Analytics Dashboard**: Live metric tiles displaying total records, oil spill case ratios, and Plotly visualization charts.
3. **Observation Preprocessing**: Automatic Gaussian noise reduction and CLAHE (Contrast Limited Adaptive Histogram Equalization) processing.
4. **AI Inference & Grad-CAM Explainability**: High-confidence binary classification backed by gradient-weighted class activation mapping (Grad-CAM) to highlight region focus.
5. **Spill Metrics Estimation**: Estimates the percentage of the observation footprint affected using HSV/Grayscale adaptive segmentation, mapping results to threat levels (Low, Medium, High, Very High).
6. **PDF Compliance Reporting**: Instant compile-and-download functionality using ReportLab to export structural, sign-off ready compliance documents containing metadata, metrics, and original-vs-heatmap comparisons.
7. **History Log Auditing**: Operator logs queryable via search parameters and date-range inputs.

---

## System Architecture

```
                       [Satellite Image Upload]
                                  │
                                  ▼
                     [Pre-processing & Contrast]
                                  │
                                  ▼
                [TensorFlow MobileNetV2 Classifier]
                       /                      \
                      ▼                        ▼
               [Spill Presence]         [Grad-CAM Gradients]
                      │                        │
                      ▼                        ▼
            [Area % & Severity]          [Heatmap Overlay]
                      \                        /
                       ▼                      ▼
                     [Save to SQLite database (DB)]
                                  │
                                  ▼
                    [Standardized PDF Report]
```

---

## File Structure

```
OilSpillAI/
├── config.json                     # Paths, model parameters, thresholds
├── requirements.txt                # System library dependencies
├── generate_dataset.py             # Script to generate synthetic training data
├── train.py                        # Transfer learning model training and evaluation
├── predict.py                      # Offline predictions & core predictor class
├── database.db                     # SQLite database (created on first run)
├── utils/
│   ├── database.py                 # User authentication & history logging database scripts
│   ├── gradcam.py                  # Grad-CAM class activation mapping
│   ├── preprocess.py               # Gaussian noise filters, CLAHE, & percentage calculation
│   └── report.py                   # ReportLab PDF generator
├── models/                         # Trained model storage & evaluation curves
│   └── saved_model.keras
└── streamlit_app/
    ├── app.py                      # Streamlit application UI and routing
    └── assets/                     # Output assets (predictions and reports)
```

---

## Step-by-Step Execution Guide

### 1. Prerequisite Setup
Ensure Python 3.12+ is installed. Clone this repository or locate the directory, and run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
Since actual satellite datasets can be multi-gigabyte or proprietary, build a synthetic training set simulating dark metallic slick contours blended over wave-patterned ocean surfaces:

```bash
python generate_dataset.py
```
This initializes the `dataset/train` and `dataset/test` splits automatically.

### 3. Model Training & Evaluation
Trigger the transfer learning script to load MobileNetV2 with pre-trained ImageNet weights, fit it to the synthetic data, compile performance metrics, and save evaluation curves:

```bash
python train.py
```
Outputs:
- Trained model saved at `models/saved_model.keras`
- Performance evaluations stored in `models/` (`metrics.json`, `accuracy_curve.png`, `loss_curve.png`, `confusion_matrix.png`, `roc_curve.png`)

### 4. Running the Web Portal
Launch the Streamlit web dashboard locally:

```bash
streamlit run streamlit_app/app.py
```

*Default Admin Credentials:*
- **Username:** `admin`
- **Password:** `admin123`

---

## Future Enhancements
- **Live Satellite Feeds**: Integration with Sentinel-2 or Landsat-8 APIs.
- **Drone Telemetry**: API interfaces to receive real-time aerial streaming telemetry.
- **Automated Alerts**: Email (SendGrid) and SMS (Twilio) triggers upon high-severity events.
- **Dockerization**: Complete containerization for deployment in Kubernetes environments.
