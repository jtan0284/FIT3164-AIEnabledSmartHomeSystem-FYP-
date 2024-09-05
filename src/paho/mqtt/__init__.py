import paho.mqtt.client as mqtt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import json  # To parse JSON messages
import pandas as pd  # To read the Excel file
__version__ = "2.1.1.dev0"

# necessary packages for decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

# necessary packages for gradient boosting model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

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
        self.dataframe = []
        
        # model variables 
        self.reg_model = None
        self.tree_model = None
        self.gb_model = None

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

        # self.regression()
        # self.decision_tree()
        self.gradient_boosting()

        
    def regression(self):
        X = np.array(self.data)
        y = np.array(self.target)
        self.reg_model = LinearRegression()
        self.reg_model.fit(X, y)
        print(f"Regression coefficients (Temperature, Humidity): {self.reg_model.coef_}")
        print(f"Intercept: {self.reg_model.intercept_}")
        predictions = self.reg_model.predict(X)
        self.plot_regression_tree()
        return
    
    def plot_regression_tree(self):
        X = np.array(self.data)  # Features: temperature and humidity
        y = np.array(self.target)  # Target: encoded user actions
         # Plotting the results
        plt.figure(figsize=(10, 6))

        # Make predictions using the regression model
        predictions = self.reg_model.predict(X)

        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Scatter plot of actual data
        plt.scatter(X[:, 0], y, color='blue', label='Actual Data')

        # Plot predicted values
        plt.scatter(X[:, 0], predictions, color='red', label='Predicted Data')

        plt.xlabel('Temperature')
        plt.ylabel('User Action (Encoded)')
        plt.title('Linear Regression: Temperature vs. User Action')
        plt.legend()
        plt.show()
        return
    
    def decision_tree(self):
        X = np.array(self.data)  # Features: temperature and humidity
        y = np.array(self.target)  # Target: encoded user actions
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        self.tree_model = DecisionTreeClassifier()
        self.tree_model.fit(X_train, y_train)

        print(f"Decision Tree model trained successfully.")
        # Make predictions
        predictions = self.tree_model.predict(X_test)

        # Evaluate the model
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(classification_report(y_test, predictions))

        self.plot_decision_tree()
        return 
    
    def plot_decision_tree(self):
        plt.figure(figsize=(20, 10))  
        plot_tree(self.tree_model, feature_names=['Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm'], class_names=['Decrease', 'Do Nothing', 'Increase'], filled=True)
        plt.show()

    def gradient_boosting(self):
        X = np.array(self.data)  # Features: temperature and humidity
        y = np.array(self.target)  # Target: encoded user actions

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Gradient Boosting model
        self.gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.gb_model.fit(X_train, y_train)

        print(f"Gradient Boosting model trained successfully.")
        
        # Make predictions
        predictions = self.gb_model.predict(X_test)

        # Evaluate the model
        print(f"Accuracy: {accuracy_score(y_test, predictions)}")
        print(classification_report(y_test, predictions))
        return

if __name__ == "__main__":
    # Example broker, you should replace this with the actual broker address you intend to use
    # mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
    mqtt_client = MyMQTTClient(broker="broker.hivemq.com")

    # Simulate data from an Excel file for testing
    mqtt_client.simulate_from_csv(r"C:\Users\ethan\OneDrive\Documents\VScode\archive\weatherAUS.csv")
    

# # Example usage
# mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
# mqtt_client.start()
