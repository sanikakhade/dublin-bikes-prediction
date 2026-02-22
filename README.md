# 🚴 Short-Term Bike Availability Prediction — Dublin Bikes

A machine learning project that predicts the number of available bikes at Dublin Bikes stations for the next timestamp, using 2.9 million real-world records from the Smart Dublin open data platform (Q1 2019).

> 📚 Individual Academic Project | MSc in Artificial Intelligence | National College of Ireland (2025)

---

## 📌 Problem Statement

> *"Can we accurately predict the number of bikes available at a Dublin Bikes station at the next timestamp using historical real-time data?"*

Dublin Bikes users often face two frustrating problems — arriving at a station to find no bikes available, or finding no empty stands to return a bike. This project builds a predictive ML model to help users plan their commute in advance and support smarter bike redistribution for operators.

---

## 🗂️ Dataset

| Detail | Info |
|---|---|
| **Source** | [Smart Dublin — Dublinbikes API](https://data.smartdublin.ie/dataset/dublinbikes-api) |
| **Period** | Q1 2019 (January – March) |
| **Records** | ~2.9 million rows |
| **Update Frequency** | Every 5 minutes per station |
| **Key Attributes** | Station ID, Available Bikes, Bike Stands, Available Bike Stands, Status, Latitude, Longitude, Timestamp |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning, feature engineering |
| **NumPy** | Numerical operations |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Matplotlib** | Time-series line charts |
| **Seaborn** | Heatmaps and EDA visualizations |
| **Jupyter Notebook** | Interactive development & documentation |

---

## ⚙️ Project Pipeline

```
Raw CSV (2.9M rows)
    │
    ▼
Data Loading & EDA
    │
    ▼
Preprocessing (datetime parsing, sorting by station + time)
    │
    ▼
Feature Engineering (11 → 23 features)
    │
    ▼
ML Model Training (Dummy / Linear Regression / Neural Network / Random Forest)
    │
    ▼
Evaluation (RMSE, MAE, R²)
    │
    ▼
Feature Importance Analysis (RF Importance + Permutation Importance)
```

---

## 🔧 Feature Engineering

Starting from 11 raw columns, the dataset was expanded to **23 features**:

| Feature | Description |
|---|---|
| `hour`, `day`, `weekday`, `month`, `weekofyear` | Temporal decomposition from timestamp |
| `is_morning_rush` | Binary flag — hour between 7–10 |
| `is_evening_rush` | Binary flag — hour between 16–19 |
| `is_weekend` | Binary flag — Saturday or Sunday |
| `month_part` | Beginning / Middle / End of month |
| `lag_1` | Available bikes at previous timestamp (t-1) |
| `target_next` | Available bikes at next timestamp (prediction target) |

> Rows with NaN values from lag/target shifting (~220 records) were dropped.

---

## 📊 Exploratory Visualizations

| Chart | Insight |
|---|---|
| 📈 Hourly Time-Series Line Chart | Clear temporal rhythms with dips during peak commute hours |
| 🌡️ Day × Hour Heatmap | Highest availability at night (8PM–6AM); lowest during daytime |

---

## 🤖 Models Trained

| Model | Purpose |
|---|---|
| **Dummy Mean Predictor** | Baseline reference |
| **Linear Regression** | Simple interpretable model for continuous prediction |
| **Neural Network (Dense / Feedforward)** | Captures non-linear temporal relationships |
| **Random Forest Regressor** | Ensemble method; trained on subsampled data due to 2.9M row size |

Data split: **80% training / 20% testing** (random split across all stations)

Preprocessing: `StandardScaler` for numerical features, `OneHotEncoder` for categoricals via `ColumnTransformer`

---

## 📈 Model Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Dummy Mean Predictor | 10.866 | 9.113 | ~0.00 |
| Linear Regression | 1.011 | 0.442 | **0.9913** |
| Neural Network | **0.986** | **0.482** | **0.9918** |
| Random Forest | 2.449 | 1.893 | 0.9492 |

✅ **Best Model: Neural Network** — R² of 0.9918, RMSE of 0.986

---

## 🔑 Feature Importance

Both **Random Forest Feature Importance** and **Permutation Importance (NN)** consistently ranked:

1. `AVAILABLE BIKES` — current availability (most dominant predictor)
2. `lag_1` — bikes available at previous timestamp

This confirms the system is **highly autoregressive** — the best predictor of the next bike count is the current and previous count.

---

## 🚀 How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/sanikakhade/dublin-bikes-prediction.git
cd dublin-bikes-prediction
```

### Step 2 — Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 3 — Download the dataset
Visit [Smart Dublin](https://data.smartdublin.ie/dataset/dublinbikes-api) and download the Q1 2019 CSV file. Place it in the `/data/` directory.

### Step 4 — Launch the notebook
```bash
jupyter notebook bike_analysis.ipynb
```

### Step 5 — Run all cells in order
1. Data loading & EDA
2. Preprocessing & feature engineering
3. Model training & evaluation
4. Feature importance analysis

---

## 📁 Project Structure

```
dublin-bikes-prediction/
│
├── bike_analysis.ipynb                    # Main analysis & ML notebook
├── data/
│   └── dublinbikes_20190101_20190401.csv  # Raw dataset (download separately)
│
└── README.md
```

---

## 🔑 Key Findings

- 🏆 Neural Network achieved the best performance with **R² = 0.9918** and **RMSE = 0.986**
- 📐 Linear Regression performed remarkably well (**R² = 0.991**), indicating mostly linear relationships
- ⏱️ `lag_1` and current `AVAILABLE BIKES` were the strongest predictors across all models
- 🌙 Bike availability is highest between **8PM–6AM** and lowest during morning/evening peak hours
- 🔄 The Dublin Bikes system is **highly autoregressive** — availability changes gradually over short intervals

---

## 🔮 Future Work

- Extend dataset to full multi-year data for seasonal pattern capture (tourism, holidays)
- Incorporate external features such as weather (rainfall, temperature)
- Build a real-time interactive prediction dashboard using Plotly/Dash
- Explore LSTM/time-series deep learning models for longer-horizon forecasting

---

## 📜 License

This project is for academic purposes only. Dataset sourced from [Smart Dublin](https://data.smartdublin.ie) under open data license.

---
