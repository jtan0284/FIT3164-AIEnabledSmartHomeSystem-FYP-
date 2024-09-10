import paho.mqtt.client as paho
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
from sklearn.metrics import roc_curve, auc

# nexeassary packages for checking dataset imbalance
import numpy as np
from collections import Counter

# neceasary packages for date time
import datetime 

class MQTTException(Exception):
    pass

import paho.mqtt.client as mqtt

class Model_training:
    def __init__(self,payload, preference, insert_time):
        self.messages = []
        self.data = []
        self.target = []
        self.dataframe = []
        self.preference = preference
        self.payload = payload
        self.insert_time = datetime.datetime.strptime(insert_time, "%d %H:%M:%S")  # Ensure start_time is a datetime object

        self.preprocessing(self.payload)
        
        # model variables 
        self.reg_model = None
        self.tree_model = None
        self.gb_model = None

    def preprocessing(self, payload):

        current_time = datetime.datetime.now().strftime("%d %H:%M:%S")  # Get current time
        # Extract numeric values and append to self.data
        for temp in payload:
            number_value = float(temp.strip("b'"))  # Remove 'b' and quotes, then convert to float
            
                # Check if the current time matches the insert_time condition
            if current_time == self.insert_time:
                # Insert user_preference at the specific time
                self.data.append([current_time, number_value, self.preference])
            else:
                # Insert without user_preference (could use None or some other placeholder)
                self.data.append([current_time, number_value, None])
            current_time += datetime.timedelta(minutes=1)

        # Convert the data into a pandas DataFrame
        self.dataframe = pd.DataFrame(self.data, columns=['Timestamp', 'Temperature', 'User_preferencce'])
        print(self.dataframe)
        
    def temperature_control(self):
        for rows in self.dataframe:
            while[rows][1] != self.preference:
                if [rows][i] > self.preference:
                    action = "decrease"
                elif [rows][i] < self.preference:
                    action = "increase"
                else:
                    action = "do nothing"
                print(action)

    def encode_user_action(self, action):
        if action == "Increase":  # Adjust this to match the exact label in your dataset
            return 1
        elif action == "Decrease":
            return -1
        elif action == "Do Nothing":
            return 0
        else:
            raise ValueError(f"Unknown user action: {action}")
        
    def simulate_from_csv(self, csv_file):
        df = pd.read_csv(csv_file)  # Read the Excel file into a DataFrame

        # Drop rows with NA values in specific columns only
        df = df.dropna(subset=['Humidity9am', 'Humidity3pm', 'Temp9am', 'Temp3pm'])

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
        self.plot_roc_curve()
        return

    def plot_roc_curve(self):
        # Ensure the model has been trained
        if self.gb_model is None:
            print("Model has not been trained yet.")
            return

        X = np.array(self.data)  # Features: temperature and humidity
        y = np.array(self.target)  # Target: encoded user actions
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the predicted probabilities
        y_pred_prob = self.gb_model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and ROC area, specifying pos_label as -1
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random performance
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

        return


if __name__ == "__main__":
    # Example broker, you should replace this with the actual broker address you intend to use
    # mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
    # mqtt_client = MyMQTTClient(broker="broker.hivemq.com")
    # mqtt_client = MyMQTTClient("broker.hivemq.com")

    # Simulate data from an Excel file for testing
    # mqtt_client.simulate_from_csv(r"C:\Users\ethan\OneDrive\Documents\VScode\archive\weatherAUS.csv")
    # # Load the dataset
    # df = pd.read_csv(r"C:\Users\ethan\OneDrive\Documents\VScode\archive\weatherAUS.csv")
    temperature_data = [
    "b'21.8'", "b'28.2'", "b'33.1'", "b'25.9'", "b'23.4'", 
    "b'30.0'", "b'32.5'", "b'26.7'", "b'24.8'", "b'27.1'", 
    "b'29.6'", "b'22.0'", "b'31.9'", "b'34.7'", "b'20.3'", 
    "b'28.4'", "b'21.2'", "b'30.8'", "b'26.5'", "b'22.7'", 
    "b'27.9'", "b'23.1'", "b'25.4'", "b'32.3'", "b'21.6'", 
    "b'33.5'", "b'29.2'", "b'24.6'", "b'34.1'", "b'22.5'"
    ]

    insert_time = datetime.datetime.now().strftime("%d %H:%M:%S")
    preference = 28.9

    model_test = Model_training(temperature_data, preference, insert_time)

    # # Check unique values in the UserAction column
    # print(df['UserAction'].unique())

    # # Check the count of each UserAction
    # print(df['UserAction'].value_counts())

# # Example usage
# mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
# mqtt_client.start()
