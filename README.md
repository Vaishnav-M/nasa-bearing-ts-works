# NASA Bearing Time Series Anomaly Detection

## AI/ML Engineer (Fresher) Assignment - Time Series Anomaly Detection for IoT Sensors

**Author:** Vaishnav M  
**Date:** November 2025  
**Dataset:** NASA IMS Bearing Dataset (Run-to-Failure)  
**Duration:** ~8 hours

> � **Technical Report**: See [ASSIGNMENT_SUMMARY.md](ASSIGNMENT_SUMMARY.md) for complete 2-3 page technical summary

---

## 📋 Project Overview

This project implements an end-to-end machine learning solution for detecting anomalies in time series sensor data from bearing vibration signals. The system compares **3 anomaly detection approaches** to identify equipment failure patterns in real-world NASA bearing data.

### Business Context
Manufacturing facilities use IoT sensors to monitor critical equipment. Early detection of anomalies in vibration signals can prevent catastrophic failures and enable predictive maintenance, reducing downtime and maintenance costs by 30-40%.

---

## 🚀 Quick Start

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Launch Jupyter
jupyter notebook

# 3. Open: notebooks/nasa_bearing_validation.ipynb
# 4. Click: Kernel → Restart & Run All
```

**Runtime:** ~30 minutes (GPU) or ~1 hour (CPU)

---

### Prerequisites
- Python 3.10+
- CUDA 11.2 + cuDNN 8.1 (for GPU acceleration)
- 8GB+ RAM recommended

### Installation

```powershell
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python validate_project.py
```

---

## 📁 Project Structure

```
nasa-bearing-ts-works/
├── notebooks/
│   └── nasa_bearing_validation.ipynb    # Main analysis notebook (101 cells)
├── src/
│   ├── nasa_data_loader.py              # Data loading & statistical features
│   ├── feature_engineering.py           # Time series feature engineering
│   └── models/
│       ├── statistical_models.py        # Isolation Forest & LOF
│       └── lstm_autoencoder.py          # Deep learning model
├── data/
│   ├── raw/bearing_dataset/             # NASA dataset (3 test sets)
│   └── processed/                       # Processed features
├── outputs/
│   ├── models/                          # Saved model files (*.h5, *.npy)
│   ├── plots/                           # Visualizations (15+)
│   └── results/                         # Evaluation metrics (CSV)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── ASSIGNMENT_SUMMARY.md                # 2-3 page technical report
```

---

## 📊 Dataset Information

### NASA IMS Bearing Dataset

**Source:** NSF I/UCR Center for Intelligent Maintenance Systems (IMS), University of Cincinnati

**📥 Download Dataset:**
- **Kaggle:** [NASA Bearing Dataset](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) ← **USED**
- **Size:** ~6GB (excluded from git - download separately)

**Installation:**
```bash
# Download from Kaggle and extract to:
data/raw/bearing_dataset/
```

### Dataset Overview

| Set | Duration | Files | Channels | Bearings | Failure Type | Usage |
|-----|----------|-------|----------|----------|--------------|-------|
| **Set 1** | Oct 22 - Nov 25, 2003 | 2,156 | 8 | 1-4 | Inner race (B3), Roller (B4) | **Primary Training** ✅ |
| **Set 2** | Feb 12-19, 2004 | 984 | 4 | 1-4 | Outer race (B1) | Cross-validation |
| **Set 3** | Mar 4 - Apr 4, 2004 | 4,448 | 4 | 3 | Outer race (B3) | Cross-validation |

### Technical Specifications

- **Sampling Rate:** 20,480 Hz (20 kHz)
- **Data Points per File:** 20,480 (1-second snapshots)
- **Recording Interval:** Every 10 minutes (initially), increasing to 5 minutes near failure
- **Test Rig Configuration:**
  - **Speed:** 2000 RPM (constant)
  - **Load:** 6000 lbs radial load
  - **Bearings:** 4 Rexnord ZA-2115 double row bearings
  - **Sensors:** 8 accelerometers (2 per bearing - horizontal & vertical)
  
### What Happened in Each Test

**Set 1 (Used in This Project):**
- **Bearing 3:** Inner race defect detected after 2,156 recordings
- **Bearing 4:** Roller element defect (secondary failure)
- **Duration:** 35 days of continuous operation
- **Failure Mode:** Gradual degradation with 15.6% RMS increase in final 10%

**Set 2:**
- **Bearing 1:** Outer race failure after 984 recordings
- **Duration:** 7 days
- **Note:** Shorter test, abrupt failure pattern

**Set 3:**
- **Bearing 3:** Outer race failure after 4,448 recordings  
- **Duration:** 31 days
- **Note:** Longest test, most data points

### Data Format

- **File Format:** Binary files (no extension)
- **Naming:** Timestamp format `YYYY.MM.DD.HH.MM.SS`
- **Structure:** Each file contains 8 channels × 20,480 samples
- **Channels:** `['Channel_1', 'Channel_2', ..., 'Channel_8']`
- **Loading:** Custom `NASABearingDataLoader` class in `src/nasa_data_loader.py`

---

## 📈 What You Get

- **3 trained models**: Isolation Forest, LOF, LSTM Autoencoder
- **Performance metrics**: F1, Precision, Recall, ROC-AUC
- **10+ visualizations**: Training curves, confusion matrices, ROC curves
- **Cross-validation**: Original (Set 1) + Optimized (Set 2/3)
- **Saved models**: Ready for deployment

---

## 🔧 Troubleshooting

**GPU not working?**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Out of memory?**
- Reduce `BATCH_SIZE = 16` (in config cell)
- Reduce `SEQUENCE_LENGTH = 30`

---

## 🔬 Technical Approach

### Model Implementations

**1. Isolation Forest** - Statistical/Unsupervised
- Builds ensemble of random decision trees
- Anomalies require fewer splits to isolate
- Fast training (0.22s), good for baseline

**2. Local Outlier Factor** - Density-based
- Compares local density to k-nearest neighbors
- Captures local anomalies in high dimensions
- Fast training (0.15s), complementary to IF

**3. LSTM Autoencoder** - Deep Learning 🏆
- **Architecture**: Encoder→Bottleneck→Decoder
- **Training**: Only on normal data (first 90%)
- **Detection**: High reconstruction error = anomaly
- **Winner**: F1=0.43 (3.5x better than statistical)

### Results Summary

| Metric | Isolation Forest | LOF | LSTM Autoencoder |
|--------|-----------------|-----|------------------|
| **F1-Score** | 0.12 | 0.12 | **0.43** ✅ |
| **Precision** | 0.11 | 0.11 | **0.40** |
| **Recall** | 0.13 | 0.13 | **0.47** |
| **ROC-AUC** | 0.54 | 0.54 | **0.70** |
| **Training Time** | 0.22s | 0.15s | 33.47s |

---

## ✅ Assignment Checklist

- [x] **Data Preparation & Exploration** - Comprehensive EDA with visualizations
- [x] **Feature Engineering** - 96 engineered features with justification
- [x] **Two Detection Approaches**:
  - [x] Statistical/Unsupervised (Isolation Forest & LOF)
  - [x] Deep Learning (LSTM Autoencoder)
- [x] **Model Evaluation** - Precision, Recall, F1, ROC-AUC metrics
- [x] **Deliverables**:
  - [x] Well-documented Jupyter notebook (101 cells)
  - [x] Summary document (2-3 pages)
  - [x] 15+ visualizations
  - [x] README with run instructions
- [x] **Code Quality** - Production-ready with error handling and logging

---

## 📚 Documentation

- **[ASSIGNMENT_SUMMARY.md](ASSIGNMENT_SUMMARY.md)** - Complete 2-3 page technical report with detailed analysis

---

## ⚠️ Limitations & Future Work

**Limitations:**
- Approximate labels (10% heuristic, not ground truth)
- Single failure mode (inner race defects only)
- Fixed conditions (2000 RPM, constant load)

**Future Improvements:**
- Cross-validation on Set 2 & 3 (outer race failures)
- Hyperparameter optimization (Optuna/Ray Tune)
- Attention mechanisms for interpretability
- Transfer learning across bearing types
- Remaining Useful Life (RUL) prediction

---

## 📄 License

This project is for educational and assignment purposes.

---

## 📧 Contact

**Vaishnav M**  
**Email:** vaishnavmsanthosh@hotmail.com    
---

**Last Updated:** November 7, 2025  
**Status:** ✅ Assignment Complete - Ready for Review
