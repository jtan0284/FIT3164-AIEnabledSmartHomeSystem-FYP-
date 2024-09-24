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

# neceasary packages for model buidling for missing values
from sklearn.impute import KNNImputer

app = Flask(__name__)

class MQTTException(Exception):
    pass

import paho.mqtt.client as mqtt

class Model_training:
    def __init__(self):
        self.messages = []
        self.data = []
        self.target = []
        # self.preference = 22.5
        # Initialize an array of 24 lists (one for each hour of the day)
        self.hourly_data = [[] for _ in range(24)]
        self.current_hour = datetime.datetime.now().hour  # Get the current hour (0-23)

        # self.temperature = temperature
        # self.humidity = humidity
        # self.insert_time = insert_time

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

    def preprocessing(self, temperature, humidity, insert_time,pref_temperature, pref_humidity):
        """
        Process incoming temperature and humidity data, store it in the correct hour array, 
        and trigger training once the hour changes.
        """
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        current_hour = datetime.datetime.now().hour

        if pref_temperature is None:
            pref_temperature = np.nan

        if pref_humidity is None:
            pref_humidity = np.nan

        
        # Store the data entry (temperature, humidity, and timestamp)
        self.data.append([current_time, temperature, humidity,pref_temperature ,pref_humidity])
        
        # If the hour has changed, trigger model training and reset data
        if current_hour != self.current_hour:
             # Convert data to DataFrame for model training
            df = pd.DataFrame(self.data, columns=['Timestamp', 'Temperature', 'Humidity','Preferred Temperature','Preferred Humidty'])
            self.gradient_boosting(df)
            self.data = []  # Reset the data for the new hour
            self.current_hour = current_hour
        
        # For real-time predictions
        action = self.temperature_control(temperature)
        print(f"Action based on temperature: {action}")


    def temperature_control(self, temperature):
        if temperature > self.preference:
            action = "decrease"
        elif temperature < self.preference:
            action = "increase"
        else:
            action = "do nothing"
        return(action)

    def gradient_boosting(self, data):
        data = self.handle_missing(data)        

        X = data[['Temperature','Humidity']]  
        y_temp = data['Preferred_Temperature']  # Target for temperature prediction
        y_humid = data['Preferred_Humidity'] # Target for humidity prediction

        # Convert to numpy arrays if necessary
        X = X.to_numpy()  # Convert X to NumPy array
        y_temp = y_temp.to_numpy()  # Convert y_temp to NumPy array (encoded user preferences)

        # Split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)

        # Initialize and train the Gradient Boosting Regressor for temperature prediction
        self.gb_model_temp = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.gb_model_temp.fit(X_train, y_train_temp)

        print(f"Gradient Boosting model trained successfully for temperature prediction.")
                
        # Make temperature predictions on the test set
        predictions_temp = self.gb_model_temp.predict(X_test)

        # Evaluate the model using Mean Squared Error (MSE) for temperature prediction
        mse_temp = mean_squared_error(y_test_temp, predictions_temp)
        print(f"Mean Squared Error (Temperature): {mse_temp}")

        # Gradient boosting for humidity data 
        y_humid = y_humid.to_numpy()
        X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(X, y_humid, test_size=0.2, random_state=42)

        self.gb_model_humid = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.gb_model_humid.fit(X_train_hum, y_train_hum)
        print(f"Gradient Boosting model trained successfully for humidity prediction.")

        # Make humidity predictions
        predictions_humid = self.gb_model_humid.predict(X_test_hum)
        mse_humid = mean_squared_error(y_test_hum, predictions_humid)
        print(f"Mean Squared Error (Humidity): {mse_humid}")

        # Plot Predicted vs Actual values for temperature
        self.plot_predicted_vs_actual(y_test_temp, predictions_temp, "Temperature")

        return

    def plot_predicted_vs_actual(self, y_true, y_pred, label):
        """
        Plot the predicted vs actual user preferences.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, color='blue', label=f'Predicted vs Actual {label}')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--', lw=2, label='Ideal Line')
        
        plt.xlabel(f'Actual {label}')
        plt.ylabel(f'Predicted {label}')
        plt.title(f'Predicted vs Actual {label}')
        plt.legend(loc="upper left")
        plt.show()

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
