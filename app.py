from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)


model_bitcoin_crypto = pickle.load(open('Models/Bitcoin.pkl', 'rb'))
model_ethereum_crypto= pickle.load(open('Models/Ethereum.pkl', 'rb'))
model_ripple_crypto = pickle.load(open('Models/Ripple.pkl', 'rb'))

bitcoin_data = pd.read_csv('Bitcoin_data.csv')
ethereum_data = pd.read_csv('Ethereum_data.csv')
ripple_data = pd.read_csv('Ripple_data.csv')

bitcoin_pred_df = pd.read_csv('Bitcoin_pred.csv').tail(5)
ethereum_pred_df = pd.read_csv('Ethereum_pred.csv').tail(5)
ripple_pred_df = pd.read_csv('Ripple_pred.csv').tail(5)


mae_bitcoin = 2.4296390123270317
mae_ethereum = 4.19018730067933
mae_ripple = 5.881519760666388


def calculate_dates(crypto_data, n_timesteps):
    min_required_date = crypto_data['Date'].iloc[n_timesteps - 1]
    return crypto_data[crypto_data['Date'] >= min_required_date]['Date'].tolist()

available_dates_bitcoin = calculate_dates(bitcoin_data, 90)
available_dates_ethereum = calculate_dates(ethereum_data, 60)
available_dates_ripple = calculate_dates(ripple_data, 120)


def feature_extraction(crypto_data, date, n_timesteps):
    
    selected_rows = crypto_data[crypto_data['Date'] <= date].tail(n_timesteps)
    
    if len(selected_rows) < n_timesteps:
        raise ValueError(f"Not enough data available for {n_timesteps} timesteps before the selected date.")
    
    
    features = selected_rows[['Close']].values  
    

    if np.isnan(features).any():
        raise ValueError("The data contains NaN values.")
    
    return features.astype(np.float32)

# Function to predict the next 5 days' prices
def prediction_next(model, features):
    predictions = []
    current_input = features.reshape(1, features.shape[0], features.shape[1])
    for _ in range(5):
        predicted_price = model.predict(current_input)[0][0]
        predictions.append(predicted_price * 10000)
        
        
        predicted_price_reshaped = np.full((1, 1, current_input.shape[2]), predicted_price)
        current_input = np.append(current_input[:, 1:, :], predicted_price_reshaped, axis=1)
        
    return predictions

@app.route('/')
def home():
    return render_template('index.html',
                           available_dates_bitcoin=available_dates_bitcoin,
                           available_dates_ethereum=available_dates_ethereum,
                           available_dates_ripple=available_dates_ripple,
                           cryptocurrency_choice="")

@app.route('/view_crypto', methods=['POST'])
def view_crypto():
    cryptocurrency_choice = request.form['crypto']
    
    if cryptocurrency_choice == 'Bitcoin':
        close_prices = bitcoin_pred_df['Close'].tolist()
        predicted_prices = bitcoin_pred_df['Predictions'].tolist()
        available_dates = available_dates_bitcoin
        mae = mae_bitcoin
    elif cryptocurrency_choice == 'Ethereum':
        close_prices = ethereum_pred_df['Close'].tolist()
        predicted_prices = ethereum_pred_df['Predictions'].tolist()
        available_dates = available_dates_ethereum
        mae = mae_ethereum
    elif cryptocurrency_choice == 'Ripple':
        close_prices = ripple_pred_df['Close'].tolist()
        predicted_prices = ripple_pred_df['Predictions'].tolist()
        available_dates = available_dates_ripple
        mae = mae_ripple
    else:
        return "Invalid selection."

    return render_template('index.html',
                           cryptocurrency_choice=cryptocurrency_choice,
                           close_prices=close_prices,
                           predicted_prices=predicted_prices,
                           available_dates=available_dates,
                           mae=mae,
                           zip=zip)

@app.route('/predict', methods=['POST'])
def predict():
    cryptocurrency_choice = request.form['crypto']
    selected_date = request.form['date']
    
    if cryptocurrency_choice == 'Bitcoin':
        model = model_bitcoin_crypto
        data = bitcoin_data
        close_prices = bitcoin_pred_df['Close'].tolist()
        predicted_prices = bitcoin_pred_df['Predictions'].tolist()
        available_dates = available_dates_bitcoin
        n_timesteps = 90
        mae = mae_bitcoin
    elif cryptocurrency_choice == 'Ethereum':
        model = model_ethereum
        data = ethereum_data
        close_prices = ethereum_pred_df['Close'].tolist()
        predicted_prices = ethereum_pred_df['Predictions'].tolist()
        available_dates = available_dates_ethereum
        n_timesteps = 60
        mae = mae_ethereum
    elif cryptocurrency_choice == 'Ripple':
        model = model_ripple_crypto
        data = ripple_data
        close_prices = ripple_pred_df['Close'].tolist()
        predicted_prices = ripple_pred_df['Predictions'].tolist()
        available_dates = available_dates_ripple
        n_timesteps = 120
        mae = mae_ripple
    else:
        return "Invalid selection."
    
    try:
        features = feature_extraction(data, selected_date, n_timesteps)
        predictions = prediction_next(model, features)

        return render_template('index.html',
                               predictions=predictions,
                               selected_date=selected_date,
                               available_dates=available_dates,
                               cryptocurrency_choice=cryptocurrency_choice,
                               close_prices=close_prices,
                               predicted_prices=predicted_prices,
                               mae=mae,
                               zip=zip,
                               enumerate=enumerate)  
    except ValueError as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
