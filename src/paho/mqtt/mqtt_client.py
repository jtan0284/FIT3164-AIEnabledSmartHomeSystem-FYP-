import paho.mqtt.client as paho

class MQTT_connect:
    def __init__(self):
        self.client = paho.Client()  
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.client.connect('broker.hivemq.com', 1883)
        self.client.subscribe('mds06_temperature/#', qos=0)
        self.client.loop_forever()

    def on_subscribe(client, userdata, mid, granted_qos):
        print("Subscribed: "+str(mid)+" "+str(granted_qos))

    def on_message(client, userdata, msg):
        print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))    

if __name__ == "__main__":
    connect = MQTT_connect()
