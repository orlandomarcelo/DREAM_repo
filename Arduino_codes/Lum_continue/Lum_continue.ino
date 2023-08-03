#include <math.h>

// the number of the LED pin
const int ledPin = 2;

// the number of the flash pin
const int  flashPin = 5;

// setting PWM properties
const int freq = 997 ; //refresh rate
const int ledChannel = 0;
const int resolution = 10;
int max_amp = pow(2, resolution) - 1;

// setting the actinic light properties
float amp_fact = 0.52; // Amplitude of modulation a fraction of max intensity

int PWM_value = 0;
int trigger = 1;


void setup() {
  Serial.begin(115200);
  pinMode(flashPin, INPUT);
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
}

void loop() {
  //trigger = digitalRead(flashPin);
  if (trigger == 1){
    PWM_value = round(amp_fact * max_amp);
    ledcWrite(ledChannel, PWM_value);
    delay(30000);   
    }
  ledcWrite(ledChannel, 0);
  }
