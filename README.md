# 📱 Used iPhone Price Predictor

An end-to-end Machine Learning web application for predicting the estimated price of used iPhones in the Egyptian market.

The project uses a trained **Random Forest Regression** model and provides a user-friendly **Flask web application** where users can enter iPhone specifications and receive an estimated price in Egyptian Pounds (EGP).

---

## 🚀 Project Overview

Buying or selling a used iPhone can be difficult because the price depends on several factors such as:

- iPhone model
- RAM
- Storage capacity
- Color
- Battery health
- Warranty status

This project applies Machine Learning to these features to estimate the expected price of a used iPhone.

The complete workflow includes:

**Data → Preprocessing → Encoding → Model Training → Evaluation → Flask Web Application → Price Prediction**

---

## ✨ Features

- 📱 Supports multiple iPhone models
- 💾 Storage options dynamically related to the selected model
- 🎨 Available colors based on the selected model
- 🔋 Battery health input
- 🛡️ Warranty selection
- 🤖 Random Forest Regression model
- 💰 Price prediction in Egyptian Pounds (EGP)
- 🌐 Interactive Flask web interface
- 🖼️ iPhone image displayed with the prediction
- 📊 Model evaluation using MAE, RMSE, and R²
- 🔄 End-to-end Machine Learning pipeline

---

## 🧠 Machine Learning Model

The project uses:

**Random Forest Regressor**

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
