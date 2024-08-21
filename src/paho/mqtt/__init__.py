import paho.mqtt.client as mqtt
import numpy as np
from sklearn.linear_model import LinearRegression
import json  # To parse JSON messages
__version__ = "2.1.1.dev0"


class MQTTException(Exception):
    pass

import paho.mqtt.client as mqtt

class MyMQTTClient:
    def __init__(self, broker, port=1883, keepalive=60):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(broker, port, keepalive)
        self.messages = []
        self.data = []
        self.target = []

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        client.subscribe("$SYS/#")

    def on_message(self, client, userdata, msg):
        message = {"topic": msg.topic, "payload": msg.payload.decode()}
        self.messages.append(message)  # Store the message
        print(message)
        self.preprocessing()

    def start(self):
        self.client.loop_forever()

    def preprocessing(self):
        data_str = self.messages.payload.decode()
        data_array = json.loads(data_str)  # Parse the JSON string into a Python object

        for entry in data_array:
            self.data.append([entry[0],entry[1]])  # Treat temperature as the feature
            actions = self.encode_user_action(entry)
            self.targets.append(actions)  # User action is the target variable

    def encode_user_action(self, action):
        # Encode the user action as a numerical value
        if action == "Increase":
            return 1
        elif action == "Decrease":
            return -1
        elif action == "Do Nothing":
            return 0
        else:
            raise ValueError(f"Unknown user action: {action}")
        
    def regression(self):
        X = np.array(self.data)
        y = np.array(self.targets)
        model = LinearRegression()
        model.fit(X, y)
        print(f"Regression coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")

# Example usage
mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
mqtt_client.start()
