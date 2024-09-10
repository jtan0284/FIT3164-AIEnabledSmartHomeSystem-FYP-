import paho.mqtt.client as paho

class MQTT_connect:
    def __init__(self):
        self.client = paho.Client()  
        self.client.on_subscribe = self.on_subscribe
        self.client.on_message = self.on_message
        self.client.connect('broker.hivemq.com', 1883)
        self.client.subscribe('mds06_temperature/#', qos=0)
        self.client.loop_forever()
        self.payload_array = [None] * 100
        self.index = 0 
        

    def on_subscribe(self,client, userdata, mid, granted_qos):
        print("Subscribed: "+str(mid)+" "+str(granted_qos))

    def on_message(self,client, userdata, msg):
        print(msg.topic+" "+str(msg.qos)+" "+str(msg.payload))

        payload = msg.payload.decode("utf-8")  # Decode the payload to a string
        self.payload_array[self.index] = payload  # Store the payload
        self.index = (self.index + 1) % len(self.paylaod_array)  # Move to the next index, wrap around when reaching the end
    
        print(f"Updated Array: {self.payload_array}")    


if __name__ == "__main__":
    client = MQTT_connect()
    payload = client.payload_array

    for i in payload:
        print(i)
