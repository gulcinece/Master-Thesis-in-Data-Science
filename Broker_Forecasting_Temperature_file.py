# Code to load the saved model for future use
import joblib
import json
import keras
from keras.losses import MeanSquaredError
from keras.models import load_model, model_from_json

import pandas as pd
import numpy as np
from collections import deque
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta


# Global variables for MQTT connection
initialization = False
BROKER = 'localhost'
PORT = 1883
SENSOR_TOPIC = 'MB1Y/Temperature'
FORECAST = 'MB1Y/Temperature/Forecast'

lookback = 10
number_of_future_forecasts = 10
temperatures = deque(maxlen=lookback)
timestamps = deque(maxlen=lookback)

## Load the LSTM model for temperature forecasting
## Function to load the model, scaler, and configuration
## timestamp: The timestamp of the saved model (e.g., "20240816_143022")
## save_dir: Directory where models are saved
##Dictionary containing model, scaler, and configuration
def load_temperature_lstm_model(timestamp, save_dir='/Users/gulcinecesasmaz/Desktop/Master_Studies/MDBlue_Data/Saved_LSTMModel_Temperature_Univariate'):
    
    # Load the complete model with custom loss function
    model_path = f"{save_dir}/temperature_lstm_model_{timestamp}.h5"
    loaded_model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
    print(f" Model loaded from: {model_path}")
    
    # Load the scaler
    scaler_path = f"{save_dir}/temperature_scaler_{timestamp}.joblib"
    loaded_scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")
    
    # Load configuration
    config_path = f"{save_dir}/temperature_model_config_{timestamp}.json"
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    print(f" Configuration loaded from: {config_path}")
    
    # Load data info
    data_info_path = f"{save_dir}/temperature_data_info_{timestamp}.json"
    with open(data_info_path, 'r') as info_file:
        data_info = json.load(info_file)
    print(f" Data info loaded from: {data_info_path}")
    
    return {
        'model': loaded_model,
        'scaler': loaded_scaler,
        'config': config,
        'data_info': data_info
    }

def predict_with_loaded_model(model, scaler, new_data, lookback=10):
## Forecasting 10 days into the future
##Used loaded model to predict future values based on the last 'lookback' days of data
## model: Loaded Keras model
## scaler: Loaded MinMaxScaler
##new_data: New temperature data (pandas Series or array)
##lookback: Number of previous timesteps to use for prediction

    
    # Normalize new data
    scaled_new_data = scaler.transform(new_data.reshape(-1, 1))
    last_input = scaled_new_data.reshape(1, lookback, 1)
    # Make predictions

    future_forecasts = np.zeros((number_of_future_forecasts, 1))
    for i in range(number_of_future_forecasts):
        future_prediction = model.predict(last_input)
        future_forecasts[i] = future_prediction
        # Rolling forecast: update last_input with the new prediction
        last_input = np.append(last_input[:, 1:, :], future_prediction.reshape(1, 1, 1), axis=1)

    #print("Future forecasts shape:", future_forecasts.shape)
    # Inverse transform future forecasts
    future_forecasts = scaler.inverse_transform(future_forecasts)
    return future_forecasts

# Load the model and other components
loaded_components = load_temperature_lstm_model("20250816_194127")
model = loaded_components['model']
scaler = loaded_components['scaler']
config = loaded_components['config']
data_info = loaded_components['data_info']


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(SENSOR_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = data.get('temperature')
        ts = data.get('timestamp')

        temperatures.append(temp)
        timestamps.append(ts)
        if(len(temperatures) >= lookback):
            # Prepare data for prediction
            reordered_temperature_data = np.array(list(temperatures)[-lookback:])
            reordered_timestamp_data = list(timestamps)[-lookback:]
            
            # Get the last timestamp and create 10 future timestamps
            last_timestamp = reordered_timestamp_data[-1]
            
            # Convert to datetime if it's a string
            if isinstance(last_timestamp, str):
                last_datetime = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
            else:
                last_datetime = last_timestamp
            
            # Generate future timestamps (1 day apart each)
            future_timestamps = []
            for i in range(1, number_of_future_forecasts+1):  # 1 to 10 days ahead
                future_date = last_datetime + timedelta(days=i)
                future_timestamps.append(future_date.isoformat())

            # print(f"Future timestamps: {future_timestamps}")

            forecasts = predict_with_loaded_model(model, scaler, reordered_temperature_data, lookback)
            
            message = {
                "sensor_id": 1,
                "sensor_timestamp": reordered_timestamp_data,
                "sensor_data": reordered_temperature_data.tolist(),
                "future_timestamps": future_timestamps,
                "forecasts": forecasts.flatten().tolist()
            }
            # print(f"Publishing message {message}")
            client.publish(FORECAST, json.dumps(message))
        else:
            print("Not enough data for prediction yet.")
            print(f"Received message: {data} on topic {msg.topic}")
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()