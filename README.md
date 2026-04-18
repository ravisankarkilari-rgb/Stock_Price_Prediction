# 📈 Stock Price Prediction Using ARIMA and LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![University](https://img.shields.io/badge/JAIN-University-orange?style=flat-square)
![Course](https://img.shields.io/badge/23BSCSMA61-Time%20Series-purple?style=flat-square)

> Experiential Learning Activity — B.Tech CSE-AIML | Time Series Analysis and Forecasting Techniques (23BSCSMA61) | JAIN (Deemed-To-Be University)

---

## 📌 Project Overview

This project predicts future stock prices for 6 major NASDAQ-listed companies using two complementary time-series models:

- **ARIMA** (AutoRegressive Integrated Moving Average) — classical statistical model
- **LSTM** (Long Short-Term Memory) — deep learning neural network

The system generates a **14-day price forecast**, compares model accuracy, and produces a **BUY / SELL / HOLD investment signal** based on ensemble averaging of both models.

---

## 🏦 Stocks Analysed

| Ticker | Company | Price (Mar 2026) | Sector |
|--------|---------|-----------------|--------|
| AAPL | Apple Inc. | $220.83 | Consumer Electronics |
| MSFT | Microsoft Corporation | $404.00 | Cloud / Software |
| GOOGL | Alphabet Inc. | $166.82 | Internet / AI |
| TSLA | Tesla Inc. | $386.33 | Electric Vehicles |
| NVDA | NVIDIA Corporation | $176.32 | Semiconductors / AI |
| AMZN | Amazon.com Inc. | $208.67 | E-Commerce / Cloud |

---

## 🗂️ Project Structure

```
STOCK_PRICE_PREDICTION_ML/
│
├── main.py               ← Full Python prediction code (ARIMA + LSTM)
├── requirements.txt      ← All required libraries
├── README.md             ← This file
│
├── index.html            ← StockSight AI web dashboard
├── style.css             ← Dashboard styling
└── app.js                ← Dashboard logic (Chart.js + simulations)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/KLEELAKRISHNA424/STOCK_PRICE_PREDICTION_ML.git
cd STOCK_PRICE_PREDICTION_ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Python prediction model
```bash
python main.py
```

### 4. Open the Web Dashboard
Open `index.html` in your browser directly, or use **VS Code Live Server**.

---

## 📊 Model Results (AAPL)

| Metric | ARIMA | LSTM |
|--------|-------|------|
| RMSE ($) | 2.84 | **2.31** |
| MAE ($) | 2.17 | **1.89** |
| MAPE (%) | 1.02 | **0.87** |
| R² Score | 0.9312 | **0.9518** |

> ✅ LSTM outperformed ARIMA on all metrics

---

## 📅 14-Day Forecast (AAPL — Base: $220.83)

| Day | Date | ARIMA ($) | LSTM ($) | Avg ($) | Change |
|-----|------|-----------|----------|---------|--------|
| D1  | 2026-03-17 | 221.64 | 221.32 | 221.48 | +0.29% |
| D7  | 2026-03-25 | 225.78 | 225.89 | 225.84 | +2.27% |
| D14 | 2026-04-03 | 229.72 | 230.24 | 229.98 | +4.16% |

### 🟢 Signal: **BUY** — Predicted gain of +4.16% exceeds the +3% threshold

---

## 🌐 Web Dashboard — StockSight AI

An interactive web dashboard built with HTML5, CSS3, and JavaScript:

- 🔄 Switch between all 6 stocks instantly
- 📈 Interactive Chart.js forecast chart with confidence bands
- 🤖 Animated ARIMA and LSTM training progress bars
- 📋 Day-by-day 14-day forecast table
- ⚖️ Model accuracy comparison (RMSE, MAE, MAPE, R²)
- 🟢 BUY / 🔴 SELL / 🟡 HOLD signal with confidence score

---

## 🧮 Mathematical Background

### ARIMA Equation
```
(1 - phi_1*B - phi_2*B²)(1-B)^d * Y_t = (1 + theta_1*B + theta_2*B²) * epsilon_t
```
Project uses **ARIMA(2,1,2)** — p=2, d=1, q=2

### LSTM Gates
```
Forget Gate : f_t = sigmoid(W_f . [h_{t-1}, x_t] + b_f)
Input  Gate : i_t = sigmoid(W_i . [h_{t-1}, x_t] + b_i)
Output Gate : o_t = sigmoid(W_o . [h_{t-1}, x_t] + b_o)
Cell State  : C_t = f_t * C_{t-1} + i_t * tanh(W_c . [h_{t-1}, x_t] + b_c)
```
Configuration: **2 layers, 64 units, 20% Dropout, 60-day lookback, Adam optimiser**

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Primary language |
| yfinance | Stock data from Yahoo Finance |
| statsmodels | ARIMA model + ADF test |
| scikit-learn | Metrics + MinMaxScaler |
| pandas / numpy | Data handling |
| matplotlib | Charts and plots |
| Chart.js | Interactive web charts |
| HTML5 / CSS3 / JS | Web dashboard |

---

## 👥 Team Members

| Name | USN | Contribution |
|------|-----|-------------|
| K Leela Krishna | 23BTRCL017 | Project Lead, LSTM Model Design |
| K Ravi Sankar | 23BTRCL198 | ARIMA Model, Statistical Analysis |
| M Jeevan | 23BTRCL194 | Data Collection, Preprocessing |
| G Sai Krishna | 23BTRCL189 | Web Dashboard Development |
| K Balaram | 23BTRCL201 | Model Evaluation, Report Writing |
| G Jathin | 23BTRCL243 | Signal Generation, Testing |

---

## 👨‍🏫 Submitted To

**Prof. Chaitali Dey**
Assistant Professor, Department of Data Analytics and Mathematical Sciences
Faculty of Engineering & Technology, JAIN (Deemed-To-Be University)

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. The forecasts are generated using simulated models and should **not** be used for real investment decisions. Past performance does not guarantee future results.

---


## 🌐 Live Demo
👉 https://kleelakrishna424.github.io/STOCK_PRICE_PREDICTION_ML/
