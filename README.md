# 🚀 Cryptocurrency Price Prediction using LSTM Neural Networks

![LSTM Model](https://miro.medium.com/max/1400/1*xmjflMIoy2qPhP6F-2JBoA.png)  
*A project to predict cryptocurrency prices using LSTM & Twitter Sentiment Analysis*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![LSTM](https://img.shields.io/badge/LSTM-Neural%20Networks-green)](https://en.wikipedia.org/wiki/Long_short-term_memory)  
[![Flask](https://img.shields.io/badge/Flask-Web%20App-red)](https://flask.palletsprojects.com/)  

---

## 📌 **Project Overview**
This project utilizes **Long Short-Term Memory (LSTM)** neural networks to predict cryptocurrency prices (**Bitcoin, Ethereum, Ripple**). The model captures **temporal patterns** in time-series data, tackling the high volatility of the crypto market.  

### **🔹 Model Performance**
📈 **Mean Absolute Error (MAE):**  
✅ **Bitcoin** - **2.42%**  
✅ **Ethereum** - **4.19%**  
✅ **Ripple** - **5.88%**  

---

## 🔍 **Key Features**
✅ **📊 Data Preprocessing:** Cleaned & normalized historical price data for effective model learning.  
✅ **💬 Twitter Sentiment Analysis:** Used **RoBERTa**, a transformer model, to classify tweets as **positive, neutral, or negative**, improving predictions.  
✅ **🌐 Web Application:** Built a **Flask-based dashboard** for real-time crypto price forecasting.  

---

## 🏗 **Project Architecture**
```mermaid
graph TD;
    A[Raw Data] -->|Preprocessing| B[Cleaned Data]
    B -->|LSTM Model| C[Predicted Prices]
    C -->|Web Dashboard| D[User Interface]
    C -->|Evaluation| E[Performance Metrics]
```

---
