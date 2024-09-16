import paho.mqtt.client as paho

# importing model class from __init__ file 
from mqtt import Model_training

# neceasary packages for date time
import datetime 

class MQTTClient:
    def __init__(self, broker, port, topic, ai_model):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.ai_model = ai_model

        # Create the MQTT client instance
        self.client = paho.Client()

        # Bind the callback methods
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message

    def connect(self):
        """Connects the MQTT client to the broker."""
        print(f"Connecting to MQTT broker {self.broker} on port {self.port}...")
        self.client.connect(self.broker, self.port)

    def subscribe(self):
        """Subscribes to the specified MQTT topic."""
        print(f"Subscribing to topic {self.topic}...")
        self.client.subscribe(self.topic, qos=1)

    def start(self):
        """Start the MQTT client loop."""
        print("Starting MQTT client loop...")
        self.client.loop_forever()

    def on_subscribe(self, client, userdata, mid, granted_qos):
        """Callback function for successful subscription."""
        print(f"Subscribed: {mid}, QoS: {granted_qos}")

    def on_message(self, client, userdata, msg):
        """Callback function to handle incoming MQTT messages."""
        data = msg.payload.decode('utf-8')
        print(f"Received message: {data} on topic {msg.topic} with QoS {msg.qos}")
        insert_time = datetime.datetime.now().strftime("%H:%M:%S")
        # Process the data using the AIModel
        result = self.ai_model.preprocessing(payload=data, insert_time= insert_time)
        
        # Print the result of the AI model's processing
        print(f"AI Model processed result: {result}")

# This block ensures the following code is executed only if the script is run directly
if __name__ == "__main__":
    # Create an instance of the AIModel
    ai_model = Model_training()

    # Instantiate the MQTTClient class with broker details and the AIModel instance
    mqtt_client = MQTTClient(
        broker='broker.mqttdashboard.com',
        port=1883,
        topic='encyclopedia/#',
        ai_model=ai_model
    )

    # Connect to the broker, subscribe to the topic, and start the loop
    mqtt_client.connect()
    mqtt_client.subscribe()
    mqtt_client.start()