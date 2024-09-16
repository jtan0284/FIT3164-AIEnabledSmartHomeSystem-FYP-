import paho.mqtt.client as paho
import paho.mqtt.client as mqtt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc

# neceassary packages for checking dataset imbalance
import numpy as np
from collections import Counter

# neceasary packages for date time
import datetime 

# neceassary packages for intergration with HTML
from flask import Flask, request, render_template

app = Flask(__name__)

class MQTTException(Exception):
    pass

import paho.mqtt.client as mqtt

class Model_training:
    def __init__(self, preference):
        self.messages = []
        self.data = []
        self.target = []
        self.dataframe = []
        self.preference = preference
        self.payload = None
        self.insert_time = None

        # # self.regression()
        # self.tests()
        # print(self.dataframe)
        # self.gradient_boosting()
        
        # model variables 
        self.reg_model = None
        self.tree_model = None
        self.gb_model = None

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/set_temperature', methods=['POST'])
    def set_temperature(self):
        temperature = float(request.form['temperature'])
        user_preference = 24.0  # You can set this based on user input
        
        action = self.gb_model.temperature_control(temperature)
        
        return f"<h2>Action: {action} the temperature!</h2>"

    def preprocessing(self, payload, insert_time,window_size=100):

        current_time = datetime.datetime.now().strftime("%H:%M:%S")  # Get current time

        # Convert both current_time and self.insert_time to datetime objects
        current_time = datetime.datetime.strptime(current_time, "%H:%M:%S")
        insert_time = datetime.datetime.strptime(insert_time, "%H:%M:%S")

        # Extract numeric values and append to self.data
        temperature = float(payload.strip("b'"))  # Remove 'b' and quotes, then convert to float
        action = self.temperature_control(temperature)

        new_row = [current_time, temperature, self.preference if current_time == insert_time else None, action]

        # Add the new row to the data
        self.data.append(new_row)
      
         # Keep only the most recent 'window_size' data points (sliding window approach)
        if len(self.data) > window_size:
            self.data = self.data[-window_size:]  # Keep only the last 'window_size' entries

        # Convert the data into a pandas DataFrame
        self.dataframe = pd.DataFrame(self.data, columns=['Timestamp', 'Temperature', 'User_preference', 'action'])

        print("Updated DataFrame (Sliding Window):\n", self.dataframe.tail())  # Optional log

        if len(self.dataframe) > 0:
            self.gradient_boosting()
        
    def tests(self):
        # Define the updates as (row_index, value) pairs
        updates = [
            (5, 22.1),
            (6, 24.6),
            (7, 26.9),
            (8, 23.0),
            (15, 27.2)
        ]

        # Get the column index for 'User_preference'
        if 'User_preference' in self.dataframe.columns:
            col_index = self.dataframe.columns.get_loc('User_preference')
            
            for row, value in updates:
                if row < len(self.dataframe):
                    self.dataframe.iloc[row, col_index] = value
                    print(f"Assigned {value} to row {row} in 'User_preference'")

    def temperature_control(self, temperature):
        if temperature > self.preference:
            action = "decrease"
        elif temperature < self.preference:
            action = "increase"
        else:
            action = "do nothing"
        return(action)

    def encode_user_action(self, action):
        if action == "Increase":  # Adjust this to match the exact label in your dataset
            return 1
        elif action == "Decrease":
            return -1
        elif action == "Do Nothing":
            return 0
        else:
            raise ValueError(f"Unknown user action: {action}")

    def regression(self):
        X = self.dataframe[['Temperature']]
        y = self.dataframe['User_preference']

        # Separate the rows where User_preference is not NaN for training
        X_train = X[~y.isna()]  # Training data (non-NaN User_preference)
        y_train = y.dropna()    # Corresponding User preferences

        # Create and fit the Linear Regression model
        self.reg_model = LinearRegression()
        self.reg_model.fit(X_train, y_train)

        # Predict the User_preference for rows where it is NaN (missing)
        X_test = X[y.isna()]  # Temperature values where User_preference is NaN
        if not X_test.empty:  # Ensure there are rows to predict
            y_pred = self.reg_model.predict(X_test)

            # Fill in the missing User_preference values with the predicted values
            self.dataframe.loc[y.isna(), 'User_preference'] = y_pred
            print("Predicted user preferences for missing values:", y_pred)

        # Optionally evaluate the model (if you want to test performance)
        # Predict user preferences for X_train to see model performance on known data
        y_train_pred = self.reg_model.predict(X_train)
        mse = mean_squared_error(y_train, y_train_pred)
        print(f"Mean Squared Error on training data: {mse}")

        # Visualize or proceed with the complete dataframe
        self.plot_regression_tree()  # Assuming this method exists to visualize the results

        return
    
    def plot_regression_tree(self):
        X = self.dataframe[['Temperature']]  # Temperature as feature
        y = self.dataframe['User_preference']  # User preference as target
        
        # Drop NaN values in User_preference to match what was used in training
        X_valid = X[~y.isna()]
        y_valid = y.dropna()

        # Plotting the results
        plt.figure(figsize=(10, 6))

        # Make predictions using the regression model on non-NaN User_preference rows
        predictions = self.reg_model.predict(X_valid)

        # Scatter plot of actual data
        plt.scatter(X_valid['Temperature'], y_valid, color='blue', label='Actual Data')

        # Scatter plot of predicted values
        plt.scatter(X_valid['Temperature'], predictions, color='red', label='Predicted Data')

        # Add labels and title
        plt.xlabel('Temperature')
        plt.ylabel('User Preference')
        plt.title('Linear Regression: Temperature vs. User Preference')
        plt.legend()

        # Show the plot
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
        # Features: Using 'Temperature' (and optionally 'Humidity' if available)
        # Adjust the feature selection to match your actual DataFrame structure
        X = self.dataframe[['Temperature']]  # Add 'Humidity' if you want to use it
        y = self.dataframe['User_preference']  # Target: encoded user actions or preferences

        # Remove rows where User_preference is NaN
        X = X[~y.isna()]
        y = y.dropna()

        # Convert to numpy arrays if necessary
        X = X.to_numpy()  # Convert X to NumPy array
        y = y.to_numpy()  # Convert y to NumPy array (encoded user actions)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Gradient Boosting model
        self.gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.gb_model.fit(X_train, y_train)

        print(f"Gradient Boosting model trained successfully.")
            
        # Make predictions
        predictions = self.gb_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        # Plot ROC curve
        self.plot_roc_curve()

        return

    def plot_roc_curve(self):
        X = self.dataframe[['Temperature']]  # Features: temperature (add others if needed)
        y = self.dataframe['User_preference']  # Target: continuous user preferences
        
        # Remove NaN values to match training data
        X = X[~y.isna()]
        y = y.dropna()

        # Convert to numpy arrays if necessary
        X = X.to_numpy()
        y = y.to_numpy()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Get the predicted values for the test set
        y_pred = self.gb_model.predict(X_test)

        # Plot actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label='Ideal Line')
        
        plt.xlabel('Actual User Preferences')
        plt.ylabel('Predicted User Preferences')
        plt.title('Predicted vs Actual User Preferences')
        plt.legend(loc="upper left")
        plt.show()

        return


if __name__ == "__main__":
    temperature_data = [
    "b'21.8'", "b'28.2'", "b'33.1'", "b'25.9'", "b'23.4'", 
    "b'30.0'", "b'32.5'", "b'26.7'", "b'24.8'", "b'27.1'", 
    "b'29.6'", "b'22.0'", "b'31.9'", "b'34.7'", "b'20.3'", 
    "b'28.4'", "b'21.2'", "b'30.8'", "b'26.5'", "b'22.7'", 
    "b'27.9'", "b'23.1'", "b'25.4'", "b'32.3'", "b'21.6'", 
    "b'33.5'", "b'29.2'", "b'24.6'", "b'34.1'", "b'22.5'"
    ]

    insert_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Prompt the user to enter their temperature preference
    preference = float(input("Please enter your temperature preference: "))

    model_test = Model_training(preference)

    # # Check unique values in the UserAction column
    # print(df['UserAction'].unique())22


    # # Check the count of each UserAction
    # print(df['UserAction'].value_counts())

# # Example usage
# mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
# mqtt_client.start()
