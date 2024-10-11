#include <Arduino.h>
#include <esp_task_wdt.h>
#include <ArduinoMqttClient.h>
#include <WiFi.h>
#include <RGBLed.h>
#include <AM2302-Sensor.h>
#include "TinyIRSender.hpp"
#include <IRremote.hpp>

#if !defined(ARDUINO_ESP32C3_DEV) // This is due to a bug in RISC-V compiler, which requires unused function sections :-(.
#define DISABLE_CODE_FOR_RECEIVER // Disables static receiver code like receive timer ISR handler and static IRReceiver and irparams data. Saves 450 bytes program memory and 269 bytes RAM if receiving functions are not required.
#endif

#include "TinyIRSender.hpp"
#include <IRremote.hpp>  // Include the IRremote library

//-----------------------------------------------------------------------------------------------------
#define RGB_RED_pin           D2
#define RGB_GREEN_pin         D1
#define RGB_BLUE_pin          D0
#define SENSOR_PIN            D5

#define IR1                   D3
#define IR2                   D4
//-----------------------------------------------------------------------------------------------------

// IR part
uint8_t sAddress = 0;
uint8_t sCommand = 0;
uint8_t sRepeats = 0;

AM2302::AM2302_Sensor am2302{SENSOR_PIN};


const char* ssid = "Galaxy S21 Ultra 5G"; //replace this with your WiFi network name
const char* password = "fypmds06"; //replace this with your WiFi network password


WiFiClient wifiClient;
MqttClient mqttClient(wifiClient);

const char broker[] = "broker.hivemq.com";
int port = 1883;

unsigned long previousMillis = 0;
unsigned long connectionCounter = 0;
unsigned long rebootCount = 0; 

RGBLed led(RGB_RED_pin, RGB_GREEN_pin, RGB_BLUE_pin, RGBLed::COMMON_ANODE);

esp_task_wdt_config_t wdt_config = {
  .timeout_ms = 30000,
  .idle_core_mask = 0,
  .trigger_panic = true
};

esp_err_t _status;

void hard_restart() {
  _status = esp_task_wdt_reconfigure(&wdt_config);
  if(_status == ESP_OK) Serial.println("Initialization was successful");
  else if(_status == ESP_ERR_INVALID_STATE) Serial.println("Already initialized");
  else Serial.println("Failed to initialize TWDT ");
  esp_task_wdt_add(NULL);
  while(true);
}

//###################################################################################################
//
//###################################################################################################
void wifiConnection(){
  Serial.println();
  Serial.print("Connecting to ");
  Serial.print(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
    connectionCounter++;
    if(connectionCounter == 50){
        Serial.println();
        Serial.print("Connecting to ");
        Serial.print(ssid);
        WiFi.begin(ssid, password);
        connectionCounter = 0;
        rebootCount++; 
        if(rebootCount == 10){  
          hard_restart();
        }
    }
  }
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  Serial.print("Attempting to connect to the MQTT broker: ");
  Serial.println(broker);

  if (!mqttClient.connect(broker, port)) {
    Serial.print("MQTT connection failed! Error code = ");
    Serial.println(mqttClient.connectError());
    while (!mqttClient.connect(broker, port)){
      led.flash(RGBLed::RED, 250, 200);
      led.flash(RGBLed::RED, 250, 200);
      led.off();
      delay(1000);
    }
  }

  Serial.println("You're connected to the MQTT broker!");
  Serial.println();

    // set the message receive callback
  mqttClient.onMessage(onMqttMessage);

  // subscribe to a topic
  // mqttClient.subscribe("mds06_OnOff");
  // mqttClient.subscribe("mds06_tempControl");
  mqttClient.subscribe("mds06/rxtoesp");
  Serial.println("Ready for command from RX");
}

void setup() {
    
  Serial.begin(115200);
  // led.off();
  while (!Serial);

  for (uint8_t i=0; i<3; i++){
     led.brightness(RGBLed::RED, 100);
     delay(300);
     led.brightness(RGBLed::GREEN, 100);
     delay(300);
     led.brightness(RGBLed::BLUE, 100);
     delay(300);
  }
  led.off();
  wifiConnection();

  Serial.print(F("\n >>> AM2302-Sensor_Example <<<\n\n"));

  // set pin and check for sensor
  if (am2302.begin()) {
      // this delay is needed to receive valid data,
      // when the loop directly read again
      delay(3000);
  }
  else {
      while (true) {
        Serial.println("Error: sensor check. => Please check sensor connection!");
        delay(10000);
      }
  }

  Serial.print(F("Send IR signals at pin "));
  Serial.print(IR1);
  Serial.print(F(" & "));
  Serial.println(IR2);
}

void loop() {
  // put your main code here, to run repeatedly:
  auto status = am2302.read();
  if (status == 0){
    float temperature = am2302.get_Temperature();
    float humidity = am2302.get_Humidity();
    Serial.println();
    Serial.print("Temperature: ");
    Serial.println(temperature, 1);

    Serial.print("Humidity:    ");
    Serial.println(humidity, 1);

    mqttClient.beginMessage("mds06_temperature");
    mqttClient.print(temperature,1);
    mqttClient.endMessage();

    mqttClient.beginMessage("mds06_humidity");
    mqttClient.print(humidity,1);
    mqttClient.endMessage();
  }

  // Keep MQTT alive to receive commands from RX
  mqttClient.poll();
  led.flash(RGBLed::GREEN, 250, 200);

  delay(3000);
}

void onMqttMessage(int messageSize) {
  // use the Stream interface to print the contents

  // check if there is message from mqtt (mds06/rxtoesp)
  // do a switch case code to send different IR signal based from different case (string in this case)

  if(mqttClient.messageTopic() == "mds06/rxtoesp"){
      Serial.print(F("Command from RX: "));
      String command;
      // use the Stream interface to print the contents
      while (mqttClient.available()) {
          char msg = (char) mqttClient.read();
          String msgstr = String(msg);
          command += msgstr;
      }
      Serial.println(command);
      irAction(command);
  }
  else {
      // for other MQTT topics if available
  }
}

void irAction(String command){

  // uint8_t sAddress = 0x02;
  // uint8_t sCommand = 0x34;
  // uint8_t sRepeats = 2;
  bool valid = true;

  // IR actions depending on instructions
  if (command == "increase temperature, increase humidity") {
    Serial.println(F("Sending IR to increase temperature, increase humidity"));
    Serial.flush();
    sAddress = 0x01, sCommand = 0x31, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "increase temperature, decrease humidity") {
    Serial.println(F("Sending IR to increase temperature, decrease humidity"));
    Serial.flush();
    sAddress = 0x02, sCommand = 0x32, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "increase temperature, do nothing") {
    Serial.println(F("Sending IR to increase temperature, do nothing"));
    Serial.flush();
    sAddress = 0x03, sCommand = 0x33, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "decrease temperature, increase humidity") {
    Serial.println(F("Sending IR to decrease temperature, increase humidity"));
    Serial.flush();
    sAddress = 0x04, sCommand = 0x34, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "decrease temperature, decrease humidity") {
    Serial.println(F("Sending IR to decrease temperature, decrease humidity"));
    Serial.flush();
    sAddress = 0x05, sCommand = 0x35, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "decrease temperature, do nothing") {
    Serial.println(F("Sending IR to decrease temperature, do nothing"));
    Serial.flush();
    sAddress = 0x06, sCommand = 0x36, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "do nothing, increase humidity") {
    Serial.println(F("Sending IR to do nothing, increase humidity"));
    Serial.flush();
    sAddress = 0x07, sCommand = 0x37, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "do nothing, decrease humidity") {
    Serial.println(F("Sending IR to do nothing, decrease humidity"));
    Serial.flush();
    sAddress = 0x08, sCommand = 0x38, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else if (command == "do nothing, do nothing") {
    Serial.println(F("Sending IR to do nothing, do nothing"));
    Serial.flush();
    sAddress = 0x09, sCommand = 0x39, sRepeats = 1;
    sendNEC(IR1, sAddress, sCommand, sRepeats);
    sendNEC(IR2, sAddress, sCommand, sRepeats);
  } else {
    Serial.println(F("Unknown command!"));
    valid = false;
  }

  if (valid){
      Serial.print(F("Send now:"));
      Serial.print(F(" address=0x"));
      Serial.print(sAddress, HEX);
      Serial.print(F(" command=0x"));
      Serial.print(sCommand, HEX);
      Serial.print(F(" repeats="));
      Serial.print(sRepeats);
      Serial.println();
      led.flash(RGBLed::GREEN, 150, 100);
      led.flash(RGBLed::GREEN, 150, 100);
  } else {
      Serial.print(F("No IR signal transmitted."));
  }
}