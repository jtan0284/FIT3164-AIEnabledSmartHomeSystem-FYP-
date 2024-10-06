# publish
import paho.mqtt.client as paho
import time

def on_publish(client, userdata, mid):
    print("mid: "+str(mid))
 
client = paho.Client()
client.on_publish = on_publish
client.connect('broker.hivemq.com', 1883)
client.loop_start()
i = 0

while True:
    i += 1
    message = "decrease temperature " + str(i)
    (rc, mid) = client.publish('mds06/aitotx', str(message), qos=1)
    time.sleep(1)