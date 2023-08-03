#include <math.h>

// the number of the LED pin
const int ledPin = 2;

// the number of the flash pin
const int  flashPin = 5;

// setting PWM properties
const int freq = 1000 ; //refresh rate
const int ledChannel = 0;
const int resolution = 16;
int max_amp = pow(2, resolution) - 1;

// setting the actinic light properties
float offset_fact = 0.85; // Offset as a fraction of max intensity
float dark_pulse_time = 5; // Dark pulse time in seconds
int cont_light = round(offset_fact * max_amp);

unsigned long t = 0;
int PWM_value = 0;
unsigned long startMillis = 0;
int trigger = 1;


void setup() {
  Serial.begin(115200);
  pinMode(flashPin, INPUT);
  dark_pulse_time = dark_pulse_time * 1000;
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
}

void loop() {
  PWM_value = cont_light;
  ledcWrite(ledChannel, PWM_value);
  trigger = digitalRead(flashPin);
  if (trigger == 1){
    startMillis = millis();
    t = 0;
    while (t < dark_pulse_time ){
      Serial.println(trigger);
      t = (millis() - startMillis);
      PWM_value = 0;
      ledcWrite(ledChannel, PWM_value);
      delay(1/freq);
      
    }
    ledcWrite(ledChannel, cont_light);
  }
}
