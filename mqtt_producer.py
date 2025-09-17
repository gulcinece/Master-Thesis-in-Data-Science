import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime
import pandas as pd

## To make it look like real-time, this Python script reads temperature data from a CSV file and publishes each entry to an MQTT broker with a 1-second interval.
## 1 second interval is used here for demonstration purposes; in a real-world scenario, this could be adjusted based on actual data frequency.
## every time you run this script, it will start publishing from the beginning of the CSV file. It connects to the MQTT broker,
## publishes each temperature reading along with its timestamp, and then waits for 1 second before publishing the next reading.
# MQTT broker settings
BROKER_ADDRESS = "localhost"
PORT = 1883
TOPIC = "MB1Y/Temperature"

dataset1=pd.read_csv('/Users/gulcinecesasmaz/Desktop/Master_Studies/MDBLUE_DATA/Dataset1_MD_1Y_all.csv')
dataset=dataset1.drop(columns=['dk'])
df=dataset
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.set_index('timestamps')

# Select only 'Temperature' and 'timestamps' columns, set 'timestamps' as index
temperature = dataset[['Temperature', 'timestamps']]
temperature['timestamps'] = pd.to_datetime(temperature['timestamps'])
temperature = temperature.set_index('timestamps')


# Callback when the client connects to the broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
    else:
        print(f"Connection failed with code {rc}")

# Create MQTT client instance
client = mqtt.Client()
client.on_connect = on_connect

# Connect to broker
try:
    client.connect(BROKER_ADDRESS, PORT)
except Exception as e:
    print(f"Could not connect to broker: {e}")
    exit(1)

# Start the loop in a non-blocking way
client.loop_start()

try:
    for idx, row in temperature.iterrows():
        timestamp = idx
        value = row['Temperature']
        message = {
            "sensor_id": 1,
            "timestamp": timestamp.isoformat(),
            "temperature": value
        }
        print(f"Publishing message {idx}: {message}")
        client.publish(TOPIC, json.dumps(message))
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping the producer...")
    client.loop_stop()
    client.disconnect()

