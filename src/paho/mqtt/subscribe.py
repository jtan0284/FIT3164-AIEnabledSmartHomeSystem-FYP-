import paho.mqtt.client as paho
import threading
from . import Model_training  # Import your Model_training class from __init__.py
import datetime

# Global model instance
model = None
temperature = None
humidity = None

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully")
    else:
        print(f"Failed to connect, return code {rc}")

def on_subscribe(client, userdata, mid, granted_qos):
    print(f"Subscribed successfully with QoS {granted_qos}")

def on_message(client, userdata, msg):
    global temperature, humidity
    try:
        # Determine if the message is for temperature or humidity based on the topic
        if "temperature" in msg.topic:
            temperature_data = msg.payload.decode("utf-8").strip('b').strip("'")
            temperature = float(temperature_data)
            print(f"Received temperature: {temperature}")

        elif "humidity" in msg.topic:
            humidity_data = msg.payload.decode("utf-8").strip('b').strip("'")
            humidity = float(humidity_data)
            print(f"Received humidity: {humidity}")

        # Once both temperature and humidity are available, process them
        if temperature is not None and humidity is not None:
            insert_time = datetime.datetime.now().strftime("%H:%M:%S")
            model.preprocessing(temperature, humidity, insert_time)  # Pass both values to preprocessing
            
            # Reset values after processing
            temperature = None
            humidity = None

    except ValueError as e:
        print(f"Error processing the message: {e}")

# Start MQTT client in a separate thread
def start_mqtt():
    client = paho.Client()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    client.connect("broker.hivemq.com", 1883)
    client.subscribe("md506_temperature/#", qos=0)
    client.subscribe("md506_humidity/#", qos=0)
    client.loop_forever()

if __name__ == "__main__":
    # Initialize Model_training instance (no need for user preference)
    model = Model_training()

    # Start the MQTT client in a separate thread
    mqtt_thread = threading.Thread(target=start_mqtt)
    mqtt_thread.start()
