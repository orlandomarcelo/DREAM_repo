// the number of the LED pin
const int ledPin = 18;

// setting PWM properties
const int freq = 5000;
const int ledChannel = 0;
const int resolution = 16;

void setup() {
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    int dutyCycle = Serial.parseInt();
    ledcWrite(ledChannel, dutyCycle);
  }
}