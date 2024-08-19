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

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected with result code {reason_code}")
        client.subscribe("$SYS/#")

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))

    def start(self):
        self.client.loop_forever()

# Example usage
mqtt_client = MyMQTTClient(broker="mqtt.eclipseprojects.io")
mqtt_client.start()
