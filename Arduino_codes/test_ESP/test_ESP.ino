# include <math.h>

// the number of the LED pin
const int ledPin = 18;

// the number of the flash pin
const int  flashPin = 4;

// setting PWM properties
const int freq = 50000; //refresh rate
const int ledChannel = 0;
const int resolution = 16;
int max_amp = pow(2, resolution) - 1 ;

// setting the actinic light properties
float frequency = 1; // Hz
float offset_fact = 0.5; // Offset as a fraction of max intensity
float amp_fact = 0.5; // Amplitude of modulation a fraction of max intensity
float max_time = 10; // Experiment time

//IMPORTANT: 
// (amp_fact + offset_fact <= 1) and (amp_fact <= offset_fact=)

float t = 0;
int PWM_value = 0;
unsigned long startMillis;
int buttonState = 0; 

void setup() {
  // configure LED PWM functionalitites
  ledcSetup(ledChannel, freq, resolution);
  
  // attach the channel to the GPIO to be controlled
  ledcAttachPin(ledPin, ledChannel);
}

void loop() {
  buttonState = digitalRead(flashPin);
  if (buttonState == 0){
    while (t < max_time ){
      if (t == 0){
        startMillis = millis();
      }
      t = (millis() - startMillis)/1000;
      PWM_value = round((amp_fact * max_amp * sin(2 * PI * frequency * t) + offset_fact * max_amp));
      ledcWrite(ledChannel, PWM_value);
      delay(1000/freq);
    }
  }
}