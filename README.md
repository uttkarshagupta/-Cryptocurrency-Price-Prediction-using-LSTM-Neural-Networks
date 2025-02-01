# -Cryptocurrency-Price-Prediction-using-LSTM-Neural-Networks
This project utilized Long Short-Term Memory (LSTM) neural networks to predict prices for cryptocurrencies like Bitcoin, Ethereum, and Ripple, tackling the inherent volatility of the cryptocurrency market. LSTM was chosen for its ability to capture temporal patterns in time-series data, resulting in predictions with low error rates:
Bitcoin: 2.42% Mean Absolute Error (MAE)
Ethereum: 4.19% MAE
Ripple: 5.88% MAE
Key highlights include:

Data Preprocessing: Cleaned and normalized historical price data, scaling critical features for effective model learning.
Twitter Sentiment Analysis: Explored the potential of using RoBERTa, a pre-trained transformer model, to classify tweets into positive, neutral, and negative sentiments as a future feature to improve prediction accuracy.
Web Application: Developed a Flask-based web interface to enable users to interact with the system, view historical price trends, and access real-time cryptocurrency price forecasts.
The project demonstrates an innovative approach by integrating time-series forecasting with the potential use of social media sentiment analysis to improve accuracy. Future work aims to expand the dataset, include more cryptocurrencies, and fully integrate sentiment analysis into the prediction model.
