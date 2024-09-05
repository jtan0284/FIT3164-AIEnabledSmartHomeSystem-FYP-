import paho.mqtt.client as mqtt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import json  # To parse JSON messages
import pandas as pd  # To read the Excel file
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
        self.reg_model = None
        self.dataframe = []

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        client.subscribe("$SYS/#")

    def on_message(self, client, userdata, msg):
        message = {"topic": msg.topic, "payload": msg.payload.decode()}
        self.messages.append(message)  # Store the message
        print(message)
        self.preprocessing(message['payload'])

    def start(self):
        self.client.loop_forever()

    def preprocessing(self, payload):
        data_array = json.loads(payload)  # Parse the JSON string into a Python object

        for entry in data_array:
            self.data.append([entry[0],entry[1],entry[2],entry[3]])  # Treat temperature as the feature
            actions = self.encode_user_action(entry[4])
            self.target.append(actions)  # User action is the target variable\
            self.dataframe.append([entry[0],entry[1],entry[2],entry[3],actions])

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
        
    def simulate_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)  # Read the Excel file into a DataFrame
        df = df.dropna()
    
        for _, row in df.iterrows():
            # Convert the row to a JSON message format
            message = json.dumps([[row['Humidity9am'], row['Humidity3pm'], row['Temp9am'], row['Temp3pm'],row['UserAction']]])
            self.preprocessing(message)

        self.regression()

        
    def regression(self):
        X = np.array(self.data)
        y = np.array(self.target)
        model = LinearRegression()
        model.fit(X, y)
        print(f"Regression coefficients: {model.coef_}")
        print(f"Intercept: {model.intercept_}")


if __name__ == "__main__":
    # Example broker, you should replace this with the actual broker address you intend to use
    # mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
    mqtt_client = MyMQTTClient(broker="broker.hivemq.com")

    # Simulate data from an Excel file for testing
    mqtt_client.simulate_from_csv(r"C:\Users\ethan\OneDrive\Documents\VScode\archive\weatherAUS.csv")
    

# # Example usage
# mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
# mqtt_client.start()
