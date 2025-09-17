import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import json
from collections import deque

## It is assumed that the incoming MQTT messages are JSON formatted with 'temperature' and 'timestamp' fields.
## This script is for real-time plotting of temperature data received via MQTT. One of the plots is shared in thesis.
BROKER = 'localhost'
PORT = 1883
TOPIC = 'MB1Y/Temperature'

N = 100
temperatures = deque(maxlen=N)
timestamps = deque(maxlen=N)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')
ax.set_xlabel('timestamp')
ax.set_ylabel('Temperature')
ax.set_title('Real-time Temperature Plot')

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        temp = data.get('temperature')
        ts = data.get('timestamp')
        if temp is not None:
            temperatures.append(temp)
            timestamps.append(ts if ts else len(temperatures))
            line.set_xdata(range(len(temperatures)))
            line.set_ydata(list(temperatures))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
        print(f"Received message: {data} on topic {msg.topic}")
    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
plt.show(block=False)
client.loop_forever()
