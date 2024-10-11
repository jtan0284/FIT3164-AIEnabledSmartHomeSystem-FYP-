import socket
import time
import paho.mqtt.client as paho

# MQTT Function
def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))

def on_message(client, userdata, msg):
    global last_sent  # Declare last_sent as global
    print(msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
    payload = msg.payload.decode('utf-8')
    if last_sent != payload:
        last_sent = payload
        sock.sendto(last_sent.encode(), (UDP_IP, UDP_PORT))

# UDP
UDP_IP = "192.168.0.2"  # IP of RX
UDP_PORT = 12345  # Port number
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create UDP socket
last_sent = ""  # Initialize last_sent variable

# MQTT
client = paho.Client()
client.on_subscribe = on_subscribe
client.on_message = on_message
client.connect('broker.hivemq.com', 1883)
client.subscribe('mds06/aitotx', qos=1)
client.loop_forever()
