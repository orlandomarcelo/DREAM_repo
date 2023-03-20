#include <Arduino.h>

const int ledPin = 22;

void setup() {
  pinMode(LEDC_HS_SIG_OUT0_IDX, OUTPUT);
  Serial.begin(921600);
  Serial.println("Hello from the setup");
}

void loop() {
  delay(1000);
  Serial.println("Hello from the loop");
  delay(1000);
  digitalWrite(ledPin,HIGH);
  delay(1000);
  digitalWrite(ledPin,LOW);
}