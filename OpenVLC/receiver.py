# -*- coding: utf-8 -*-
import socket
import time
import paho.mqtt.client as paho

# MQTT Function
def on_publish(client, userdata, mid):
    print("mid: "+str(mid))

UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 12345 # Port number

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create UDP socket
sock.bind((UDP_IP, UDP_PORT)) # Bind to address

# MQTT (code placed before while loop to make sure MQTT initialization is complete)
client = paho.Client()
client.on_publish = on_publish
client.connect('broker.hivemq.com', 1883)
client.loop_start()

while True:
    data, addr = sock.recvfrom(1024) # Buffer size is 1024 bytes
    print("Received message:", data.decode()) # Print received string
    payload = data.decode()
    (rc, mid) = client.publish('mds06/rxtoesp', str(payload), qos=1)
