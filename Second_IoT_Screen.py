import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
import json
from datetime import datetime
import pandas as pd
import numpy as np
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

## MQTT Configuration and Thresholds values are implemented in this python file. It issues alerts to the user.
## Issues are for measured and forecasting values.
## The alert is shown on the screen and the background color of the plot changes according to the alert level.
# MQTT Configuration
BROKER = 'localhost'
PORT = 1883
TOPIC = 'MB1Y/Temperature/Forecast'

LOW_WARNING_THRESHOLD = 18.0
LOW_ERROR_THRESHOLD = 10.0
HIGH_WARNING_THRESHOLD = 25.0
HIGH_ERROR_THRESHOLD = 30.0

# Data storage
data_queue = queue.Queue()
plot_data = {
    'sensor_data': [],
    'sensor_timestamps': [],
    'forecasts': [],
    'future_timestamps': [],
    'sensor_id': None
}

# MQTT client setup
mqtt_client = mqtt.Client()

class RealTimeForecastPlotter:
    def __init__(self):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.suptitle('Real-Time Temperature Sensor Data & Forecasts', fontsize=16)
        
        # Initialize empty line objects
        self.line_sensor, = self.ax.plot([], [], 'b-', linewidth=2.5, 
                                        label='Sensor Data', marker='o', markersize=4)
        self.line_forecast, = self.ax.plot([], [], 'r--', linewidth=2.5, 
                                          label='Forecasts', marker='s', markersize=5)
        
        # Add threshold lines
        self.line_high_error = self.ax.axhline(y=HIGH_ERROR_THRESHOLD, color='red', 
                                              linestyle='-', linewidth=2, alpha=0.8, 
                                              label=f'High Error ({HIGH_ERROR_THRESHOLD}°C)')
        self.line_high_warning = self.ax.axhline(y=HIGH_WARNING_THRESHOLD, color='orange', 
                                                linestyle='--', linewidth=2, alpha=0.8, 
                                                label=f'High Warning ({HIGH_WARNING_THRESHOLD}°C)')
        self.line_low_warning = self.ax.axhline(y=LOW_WARNING_THRESHOLD, color='orange', 
                                               linestyle='--', linewidth=2, alpha=0.8, 
                                               label=f'Low Warning ({LOW_WARNING_THRESHOLD}°C)')
        self.line_low_error = self.ax.axhline(y=LOW_ERROR_THRESHOLD, color='red', 
                                             linestyle='-', linewidth=2, alpha=0.8, 
                                             label=f'Low Error ({LOW_ERROR_THRESHOLD}°C)')
        
        # Configure the plot
        self.ax.set_xlabel('Time', fontsize=12)
        self.ax.set_ylabel('Temperature (°C)', fontsize=12)
        self.ax.legend(loc='upper left', fontsize=9)
        self.ax.grid(True, alpha=0.3)
        
        # Format x-axis for time display
        self.ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
        self.fig.autofmt_xdate()
        
        # Set initial axis limits
        now = datetime.now()
        self.ax.set_xlim(now, now)
        self.ax.set_ylim(5, 35)  # Expanded to show thresholds
        
        # Text for displaying information
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     fontsize=10, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Alert text for threshold violations
        self.alert_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes, 
                                      fontsize=12, verticalalignment='bottom', fontweight='bold',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Connection line between sensor data and forecasts
        self.connection_line, = self.ax.plot([], [], 'r--', linewidth=2.5, markersize=5)
        
        # Initialize background color state
        self.current_background_color = 'white'
        
    def check_thresholds(self):
        """Check threshold violations and return alert status and message"""
        alert_status = 'normal'  # 'normal', 'warning', 'error'
        alert_messages = []
        
        try:
            # Get values to check: last sensor value + all forecast values
            values_to_check = []
            
            # Add last sensor value if available
            if plot_data['sensor_data']:
                last_sensor_value = plot_data['sensor_data'][-1]
                values_to_check.append(('Current Sensor', last_sensor_value))
            
            # Add all forecast values
            if plot_data['forecasts']:
                for i, forecast_value in enumerate(plot_data['forecasts']):
                    values_to_check.append(('Forecast', forecast_value))
            
            if not values_to_check:
                return alert_status, []
            
            # Check for ERROR thresholds first
            error_violations = []
            warning_violations = []
            
            for value_type, value in values_to_check:
                if value > HIGH_ERROR_THRESHOLD:
                    error_violations.append(f"{value_type}: {value:.2f}°C > {HIGH_ERROR_THRESHOLD}°C")
                elif value < LOW_ERROR_THRESHOLD:
                    error_violations.append(f"{value_type}: {value:.2f}°C < {LOW_ERROR_THRESHOLD}°C")
                elif value > HIGH_WARNING_THRESHOLD:
                    warning_violations.append(f"{value_type}: {value:.2f}°C > {HIGH_WARNING_THRESHOLD}°C")
                elif value < LOW_WARNING_THRESHOLD:
                    warning_violations.append(f"{value_type}: {value:.2f}°C < {LOW_WARNING_THRESHOLD}°C")
            
            # Determine alert status and messages
            if error_violations:
                alert_status = 'error'
                alert_messages.append("ERROR THRESHOLD EXCEEDED!")
                alert_messages.extend(error_violations[:3])  # Show max 3 violations
                if len(error_violations) > 3:
                    alert_messages.append(f"... and {len(error_violations) - 3} more violations")
            elif warning_violations:
                alert_status = 'warning'
                alert_messages.append("WARNING THRESHOLD EXCEEDED!")
                alert_messages.extend(warning_violations[:3])  # Show max 3 violations
                if len(warning_violations) > 3:
                    alert_messages.append(f"... and {len(warning_violations) - 3} more violations")
            
            return alert_status, alert_messages
            
        except Exception as e:
            print(f"Error checking thresholds: {e}")
            return 'normal', []
    
    def update_background_color(self, alert_status):
        """Update background color based on alert status"""
        color_map = {
            'normal': 'white',
            'warning': '#FFFACD',  # Light yellow (LemonChiffon)
            'error': '#FFB6C1'     # Light pink (LightPink)
        }
        
        new_color = color_map.get(alert_status, 'white')
        
        if new_color != self.current_background_color:
            self.fig.patch.set_facecolor(new_color)
            self.ax.set_facecolor(new_color)
            self.current_background_color = new_color
    
    def update_plot(self, frame):
        """Update the plot with new data"""
        try:
            # Process all available data from queue
            while not data_queue.empty():
                try:
                    new_data = data_queue.get_nowait()
                    self.process_new_data(new_data)
                except queue.Empty:
                    break
            
            # Update the plot with current data
            self.redraw_plot()
            
            # Check thresholds and update alerts
            alert_status, alert_messages = self.check_thresholds()
            self.update_background_color(alert_status)
            self.update_alert_text(alert_status, alert_messages)
                
        except Exception as e:
            print(f"Error updating plot: {e}")
        
        return self.line_sensor, self.line_forecast, self.connection_line, self.info_text, self.alert_text
    
    def process_new_data(self, data):
        """Process new MQTT data and update plot_data"""
        global plot_data
        
        try:
            # Extract data from MQTT message
            sensor_data = data.get('sensor_data', [])
            sensor_timestamps = data.get('sensor_timestamp', [])  # Note: your key might be 'sensor_timestamp'
            forecasts = data.get('forecasts', [])
            future_timestamps = data.get('future_timestamps', [])
            sensor_id = data.get('sensor_id')
            
            print(f"📊 Processing data from sensor {sensor_id}")
            print(f"   Sensor data points: {len(sensor_data)}")
            print(f"   Forecast points: {len(forecasts)}")
            
            # Convert timestamps to datetime objects
            if sensor_timestamps:
                sensor_timestamps_dt = [pd.to_datetime(ts) for ts in sensor_timestamps]
            else:
                sensor_timestamps_dt = []
            
            if future_timestamps:
                future_timestamps_dt = [pd.to_datetime(ts) for ts in future_timestamps]
            else:
                future_timestamps_dt = []
            
            # Update plot_data
            plot_data['sensor_data'] = sensor_data
            plot_data['sensor_timestamps'] = sensor_timestamps_dt
            plot_data['forecasts'] = forecasts
            plot_data['future_timestamps'] = future_timestamps_dt
            plot_data['sensor_id'] = sensor_id
            
        except Exception as e:
            print(f"Error processing new data: {e}")
    
    def redraw_plot(self):
        """Redraw the entire plot with current data"""
        try:
            # Clear previous data
            self.line_sensor.set_data([], [])
            self.line_forecast.set_data([], [])
            self.connection_line.set_data([], [])
            
            # Plot sensor data
            if plot_data['sensor_data'] and plot_data['sensor_timestamps']:
                self.line_sensor.set_data(plot_data['sensor_timestamps'], plot_data['sensor_data'])
            
            # Plot forecasts
            if plot_data['forecasts'] and plot_data['future_timestamps']:
                self.line_forecast.set_data(plot_data['future_timestamps'], plot_data['forecasts'])
            
            # Draw connection line between last sensor point and first forecast point
            if (plot_data['sensor_data'] and plot_data['sensor_timestamps'] and 
                plot_data['forecasts'] and plot_data['future_timestamps']):
                
                last_sensor_time = plot_data['sensor_timestamps'][-1]
                last_sensor_value = plot_data['sensor_data'][-1]
                first_forecast_time = plot_data['future_timestamps'][0]
                first_forecast_value = plot_data['forecasts'][0]
                
                self.connection_line.set_data([last_sensor_time, first_forecast_time],
                                             [last_sensor_value, first_forecast_value])
            
            # Auto-scale the plot
            self.auto_scale()
            
            # Update info text
            self.update_info_text()
            
        except Exception as e:
            print(f"Error redrawing plot: {e}")
    
    def auto_scale(self):
        """Auto-scale the plot axes"""
        try:
            all_timestamps = []
            all_values = []
            
            # Collect all timestamps and values
            if plot_data['sensor_timestamps'] and plot_data['sensor_data']:
                all_timestamps.extend(plot_data['sensor_timestamps'])
                all_values.extend(plot_data['sensor_data'])
            
            if plot_data['future_timestamps'] and plot_data['forecasts']:
                all_timestamps.extend(plot_data['future_timestamps'])
                all_values.extend(plot_data['forecasts'])
            
            if all_timestamps and all_values:
                # X-axis scaling
                time_min = min(all_timestamps)
                time_max = max(all_timestamps)
                time_range = time_max - time_min
                padding = time_range * 0.05  # 5% padding
                self.ax.set_xlim(time_min - padding, time_max + padding)
                
                # Y-axis scaling - ensure thresholds are visible
                value_min = min(all_values)
                value_max = max(all_values)
                
                # Include thresholds in the range
                y_min = min(value_min, LOW_ERROR_THRESHOLD) - 2
                y_max = max(value_max, HIGH_ERROR_THRESHOLD) + 2
                
                self.ax.set_ylim(y_min, y_max)
                
        except Exception as e:
            print(f"Error auto-scaling: {e}")
    
    def update_info_text(self):
        """Update the information text box"""
        try:
            info_lines = []
            
            if plot_data['sensor_id'] is not None:
                info_lines.append(f"Sensor ID: {plot_data['sensor_id']}")
            
            if plot_data['sensor_data']:
                latest_temp = plot_data['sensor_data'][-1]
                info_lines.append(f"Latest Temperature: {latest_temp:.2f}°C")
                info_lines.append(f"Sensor Data Points: {len(plot_data['sensor_data'])}")
            
            if plot_data['forecasts']:
                avg_forecast = np.mean(plot_data['forecasts'])
                info_lines.append(f"Forecast Points: {len(plot_data['forecasts'])}")
                info_lines.append(f"Avg Forecast: {avg_forecast:.2f}°C")
                
                # Calculate trend
                if len(plot_data['forecasts']) >= 2:
                    trend_change = plot_data['forecasts'][-1] - plot_data['forecasts'][0]
                    trend = "↗️ Rising" if trend_change > 0.5 else "↘️ Falling" if trend_change < -0.5 else "→ Stable"
                    info_lines.append(f"Trend: {trend} ({trend_change:+.2f}°C)")
            
            if plot_data['sensor_timestamps']:
                latest_time = plot_data['sensor_timestamps'][-1].strftime('%m-%d %H:%M:%S')
                info_lines.append(f"Last Update: {latest_time}")
            
            # Add threshold info
            info_lines.append("")
            info_lines.append("Thresholds:")
            info_lines.append(f"Error: {LOW_ERROR_THRESHOLD}°C - {HIGH_ERROR_THRESHOLD}°C")
            info_lines.append(f"Warning: {LOW_WARNING_THRESHOLD}°C - {HIGH_WARNING_THRESHOLD}°C")
            
            self.info_text.set_text('\n'.join(info_lines))
            
        except Exception as e:
            print(f"Error updating info text: {e}")
    
    def update_alert_text(self, alert_status, alert_messages):
        """Update the alert text based on threshold violations"""
        try:
            if alert_messages:
                text_color = 'red' if alert_status == 'error' else 'orange'
                bg_color = '#FFE4E1' if alert_status == 'error' else '#FFF8DC'  # MistyRose or Cornsilk
                
                self.alert_text.set_text('\n'.join(alert_messages))
                self.alert_text.set_color(text_color)
                self.alert_text.set_bbox(dict(boxstyle='round', facecolor=bg_color, alpha=0.9))
                self.alert_text.set_visible(True)
            else:
                self.alert_text.set_visible(False)
                
        except Exception as e:
            print(f"Error updating alert text: {e}")

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    """Callback for when the client receives a CONNACK response from the server"""
    if rc == 0:
        print(f"Connected to MQTT broker at {BROKER}:{PORT}")
        client.subscribe(TOPIC)
        print(f"Subscribed to topic: {TOPIC}")
        print(f"Threshold configuration:")
        print(f"  Low Error: < {LOW_ERROR_THRESHOLD}°C")
        print(f"  Low Warning: < {LOW_WARNING_THRESHOLD}°C")
        print(f"  High Warning: > {HIGH_WARNING_THRESHOLD}°C")
        print(f"  High Error: > {HIGH_ERROR_THRESHOLD}°C")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for when a PUBLISH message is received from the server"""
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        print(f"Received message on topic '{msg.topic}'")
        print(f"Message preview: sensor_id={data.get('sensor_id')}, "
              f"sensor_data_len={len(data.get('sensor_data', []))}, "
              f"forecasts_len={len(data.get('forecasts', []))}")
        
        # Add data to queue for processing
        data_queue.put(data)
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    # Setup MQTT connection
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    print(f"Connecting to MQTT broker at {BROKER}:{PORT}...")
    mqtt_client.connect(BROKER, PORT, 60)
    mqtt_client.loop_start()
    
    # Create plotter instance
    plotter = RealTimeForecastPlotter()
    
    # Setup animation
    ani = animation.FuncAnimation(
        plotter.fig, 
        plotter.update_plot, 
        interval=500,  # Update every 500ms
        blit=False,
        cache_frame_data=False
    )

    try:
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("Cleanup completed")

if __name__ == "__main__":
    main()